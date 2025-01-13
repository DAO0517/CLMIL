import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn, einsum
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics.pairwise import cosine_similarity
import math
from torch_geometric.nn import GCNConv
import numpy as np
import warnings
# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, 128),
                                nn.ReLU(),
                                nn.Linear(128, out_size))

    def forward(self, feats):
        x = self.fc(feats)
        return feats, x


class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()

        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(feature_size, output_class)

    def forward(self, x):
        device = x.device
        feats = self.feature_extractor(x)  # N x K
        c = self.fc(feats.view(feats.shape[0], -1))  # N x C
        return feats.view(feats.shape[0], -1), c


class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True):  # K, L, N
        super(BClassifier, self).__init__()

        if nonlinear:
            self.lin = nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU())
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.Tanh())
        else:
            self.lin = nn.Identity()
            self.q = nn.Linear(input_size, 128)
        self.v = nn.Sequential(
            nn.Dropout(dropout_v),
            nn.Linear(input_size, input_size)
        )

        # 1D convolutional layer that can handle multiple class (including binary)
        # still not settled whether conv or fc
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)

    def forward(self, feats, c, label=None, epoch=None):  # N x K, N x C
        device = feats.device
        feats = self.lin(feats)

        length = feats.size(0)
        V = self.v(feats)
        Q = self.q(feats).view(feats.shape[0], -1)

        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True)

        num_tiles = int(length * 0.05)
        temp_feats = feats[m_indices][0: num_tiles]

        m_feats = temp_feats.mean(dim=0)

        # temp_feats_norm = F.normalize(temp_feats, dim=2)
        # dist = torch.matmul(temp_feats_norm.permute(1, 0, 2),
        #                     temp_feats_norm.permute(1, 2, 0)).mean(dim=2, keepdim=True).permute(1, 0, 2)
        # dist = F.softmax(dist, dim=0)
        # m_feats = (dist * temp_feats).sum(dim=0)
        ###############################

        q_max = self.q(m_feats)  # compute queries of critical instances, q_max in shape C x Q

        A = torch.mm(Q, q_max.transpose(0, 1))

        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), dim=0)
        B = torch.mm(A.transpose(0, 1), V)  # compute bag representation, B in shape C x V

        B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V

        C = self.fcc(B)  # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B, m_feats


def exists(val):
    return val is not None


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim=512, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim=512, context_dim=512, heads=4, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Sequential(nn.Linear(query_dim, inner_dim, bias=False))
        self.to_q2 = nn.Sequential(nn.Linear(query_dim, inner_dim, bias=False))
        self.to_k = nn.Sequential(nn.Linear(context_dim, inner_dim, bias=False))
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x1, x2, context, mask=None):
        h = self.heads

        q = self.to_q(x1)
        q_2 = self.to_q2(x2)

        # k, v = self.to_kv(context).chunk(2, dim=-1)
        k, v = self.to_k(context), self.to_v(context)
        q, q_2, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, q_2, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        sim_2 = einsum('b i d, b j d -> b i j', q_2, k) * self.scale

        # attention, what we cannot get enough of
        # attn = (sim.softmax(dim=-1) + sim_2.softmax(dim=-1)) / 2

        # attn = sim_2.softmax(dim=-1)
        attn = 0.5* sim.softmax(dim=-1) + 0.5 * sim_2.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class Self_Attention(nn.Module):
    def __init__(self, query_dim=512, context_dim=512, heads=4, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Sequential(nn.Linear(query_dim, inner_dim, bias=False))
        self.to_q2 = nn.Sequential(nn.Linear(query_dim, inner_dim, bias=False))
        self.to_k = nn.Sequential(nn.Linear(context_dim, inner_dim, bias=False))
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x1, mask=None):
        h = self.heads

        q = self.to_q(x1)

        # k, v = self.to_kv(context).chunk(2, dim=-1)
        k, v = self.to_k(x1), self.to_v(x1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class GCNNet(nn.Module):
    def __init__(self, k_num=3):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(512, 256)
        self.conv2 = GCNConv(256, 128)
        self.conv3 = GCNConv(128, 128)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=1)  # 输入通道数为三层GCN的输出通道数之和
        self.k_neighbors = k_num

    def construct_edges(self,x):
        # 定义阈值
        similarity_threshold = 0.9  # 你可以根据需要调整相似度的阈值
        # 计算余弦相似度矩阵
        similarity_matrix = cosine_similarity(x)
        # 构建邻接矩阵，只有在余弦相似度大于阈值时才建立边
        adjacency_matrix = np.zeros_like(similarity_matrix)
        # 使用KNN找到每个样本的K个最近邻
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors, algorithm='auto').fit(x)
        _, indices = nbrs.kneighbors(x)
        for i in range(len(x)):
            for j in indices[i]:
                if similarity_matrix[i, j] > similarity_threshold:
                    adjacency_matrix[i, j] = 1

        edges = np.column_stack(np.where(adjacency_matrix == 1))
        edges_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edges_tensor

    def forward(self, x):
        x_numpy = x.detach().cpu().numpy()
        edge_index = self.construct_edges(x_numpy)
        edge_index = torch.tensor(edge_index).cuda()
        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.conv2(x1, edge_index))
        x3 = F.relu(self.conv3(x2, edge_index))
        x_concat = torch.cat([x1, x2, x3], dim=1)
        x_concat = torch.unsqueeze(torch.unsqueeze(x_concat, -1), -1)
        x_out = self.conv4(x_concat)
        x_out = torch.squeeze(x_out)
        return x_out


class MILNet(nn.Module):
    def __init__(self, i_classifier, num_classes=None, margin=0, p1=0, p2=0, CAMEYLON=False):
        super(MILNet, self).__init__()
        self.patch_selcet_dict = {}
        self.i_classifier = i_classifier
        self.num_classes = num_classes
        self.margin = margin
        self.p1, self.p2 = p1, p2
        self.CAMEYLON = CAMEYLON

        # self.lin = nn.Sequential(nn.Linear(512*2, 512), nn.LayerNorm(512), nn.ReLU())
        self.lin = nn.Sequential(nn.Linear(512 * 2, 512),
                                 nn.LayerNorm(512),
                                 nn.ReLU(inplace=True),   #inplace = True ,会改变输入数据的值,节省反复申请与释放内存的空间与时间,只是将原来的地址传递,效率更好
                                 nn.Linear(512, 512)
                                 )


        self.LN_0_1 = nn.LayerNorm(512)
        self.LN_0_2 = nn.LayerNorm(512)
        self.LN_0_3 = nn.LayerNorm(512)

        self.LN_1_1 = nn.LayerNorm(512)
        self.LN_2_1 = nn.LayerNorm(512)
        self.LN_2_2 = nn.LayerNorm(512)

        self.LN_3_1 = nn.LayerNorm(512)
        self.LN_3_2 = nn.LayerNorm(512)

        self.attention = Attention()
        self.ffn = FeedForward(dim=512)

        self.attention2 = Self_Attention()
        self.ffn2 = FeedForward(dim=512)

        self.classification_layer_conv = nn.Conv1d(num_classes, num_classes, kernel_size=512, bias=False)
        self.classification_layer_fc = nn.Linear(512, num_classes, bias=False)

        self.dropout = nn.Dropout(0.1)
        self.GCNNET = GCNNet(k_num=3)
        # initialize_weights(self)


    def forward(self, x):  #, image_paths, label

        length = x.size(0)
        b = x.detach().cpu().numpy()
        num_tiles1, num_tiles2 = math.ceil(length * self.p1), math.ceil(length * self.p2)  # p1 0.05 p2 0.01
        feats, classes = self.i_classifier(x) # 实例分类器
        feats = self.lin(feats)  # n_all,512
        # int是向上取整，有可能后面返回值为0，后面就计算不了，ceil是向上取整
        # num_tiles1, num_tiles2 = int(length * self.p1), int(length * self.p2)

        classes = F.softmax(classes, dim=1)  # (2744, 3)各个样本的概率

        temp, m_indices = torch.sort(classes, 0, descending=True)   # temp（2744，3）概率 m_indices 包含了排序后的值在原始 classes 中的索引
        mean, var = torch.mean(temp[0: num_tiles1], dim=0), torch.std(temp[0: num_tiles1], dim=0)  # 形状为（3，）
        mean_max_index, var_min_index = torch.argmax(mean), torch.argmin(var)

        cf = mean - var  # 长度为n_class[0.3638, 0.3211, 0.3274]
        cf_index = torch.topk(cf, k=2)[1] # [0,2] 获取到的元素在原数组中的位置标号,k=2 is enough

        feats_numpy = feats.detach().cpu().numpy()
        m_indices_numpy = m_indices.detach().cpu().numpy()
        similarity = []
        # for i in range(self.num_classes):
        #     similarity_matrix = cosine_similarity(feats_numpy[m_indices_numpy[0: num_tiles1, i]])
        #     similarity.append(similarity_matrix.mean())
        #   # [0,2] 获取到的元素在原数组中的位置标号,k=2 is enough
        for i in range(self.num_classes):
            inner_product_matrix = np.dot(feats_numpy[m_indices_numpy[0: num_tiles1, i]],
                                          feats_numpy[m_indices_numpy[0: num_tiles1, i]].T)
            similarity.append(inner_product_matrix.mean())

        similarity_index = torch.topk(torch.tensor(similarity), k=2)[1]

        if self.num_classes > 2:
            if cf[cf_index[0]] > cf[cf_index[1]] and cf_index[0]==similarity_index[0].cuda():
                m_feats = []
                g_feats = []
                for i in range(self.num_classes):   # num_tiles1的数量多于num_tiles2
                    if i == cf_index[0]:   # 预测概率最高的一类
                        m_feats.append(feats[m_indices[0: num_tiles1, i]].mean(dim=0, keepdim=True))
                        g_feat = self.GCNNET(feats[m_indices[0: num_tiles1, i]])
                        g_feats.append(g_feat.mean(dim=0, keepdim=True))
                        # self.patch_selcet_dict['class_'+str(i)] = [path for path in image_paths[m_indices[0: num_tiles1, i]]]
                    else:  # 预测概率小的类
                        m_feats.append(feats[m_indices[0: num_tiles2, i]].mean(dim=0, keepdim=True))
                        g_feat = self.GCNNET(feats[m_indices[0: num_tiles2, i]])
                        g_feats.append(g_feat.mean(dim=0, keepdim=True))
                m_feats = torch.cat(m_feats, dim=0)[None,]
                g_feats = torch.cat(g_feats, dim=0)[None,]
            else:
                m_feats = feats[m_indices[0: num_tiles2]]
                g_feats = []
                for j in range(self.num_classes):
                    g_feat = self.GCNNET(feats[m_indices[0: num_tiles2, j]])
                    g_feats.append(g_feat.mean(dim=0, keepdim=True))
                # self.patch_selcet_dict['class_' + str(label)] = [path for path in image_paths[m_indices[0: num_tiles2, int(label)]]]
                m_feats = m_feats.mean(dim=0, keepdim=True)
                g_feats = torch.cat(g_feats, dim=0)[None,]
        else:
            if cf[0] > cf[1] and cf_index[0]==similarity_index[0].cuda():
                m_feats = []
                m_feats.append(feats[m_indices[0: num_tiles1, 0]].mean(dim=0, keepdim=True))  # 预测为0
                # self.patch_selcet_dict['class_0'] = [path for path in image_paths[m_indices[0: num_tiles1]]]
                m_feats.append(feats[m_indices[0: num_tiles2, 1]].mean(dim=0, keepdim=True))
                m_feats = torch.cat(m_feats, dim=0)[None,]
                g_feats = []
                g_feat1 = self.GCNNET(feats[m_indices[0: num_tiles1, 0]])
                g_feats.append(g_feat1.mean(dim=0, keepdim=True))
                g_feat2 = self.GCNNET(feats[m_indices[0: num_tiles2, 1]])
                g_feats.append(g_feat2.mean(dim=0, keepdim=True))
                g_feats = torch.cat(g_feats, dim=0)[None,]
            elif cf[1] > cf[0] and cf_index[1]==similarity_index[1]:
                m_feats = []
                m_feats.append(feats[m_indices[0: num_tiles2, 0]].mean(dim=0, keepdim=True))
                m_feats.append(feats[m_indices[0: num_tiles1, 1]].mean(dim=0, keepdim=True))
                # self.patch_selcet_dict['class_1'] = [path for path in image_paths[m_indices[0: num_tiles1], 1]]
                m_feats = torch.cat(m_feats, dim=0)[None,]
                g_feats = []
                g_feat1 = self.GCNNET(feats[m_indices[0: num_tiles2, 0]])
                g_feats.append(g_feat1.mean(dim=0, keepdim=True))
                g_feat2 = self.GCNNET(feats[m_indices[0: num_tiles1, 1]])
                g_feats.append(g_feat2.mean(dim=0, keepdim=True))
                g_feats = torch.cat(g_feats, dim=0)[None,]
            else:
                g_feats = []
                for j in range(self.num_classes):
                    g_feat = self.GCNNET(feats[m_indices[0: num_tiles2, j]])
                    g_feats.append(g_feat.mean(dim=0, keepdim=True))
                g_feats = torch.cat(g_feats, dim=0)[None,]
                m_feats = feats[m_indices[0: num_tiles2]]
                # self.patch_selcet_dict['class_' + str(label)] = [path for path in image_paths[m_indices[0: num_tiles2, int(label)]]]
                m_feats = m_feats.mean(dim=0, keepdim=True)

        feats = feats[None,] # 1,n_all,512
        feats = self.attention(self.LN_0_1(m_feats), self.LN_0_2(g_feats), self.LN_0_3(feats)) + m_feats
        feats = self.ffn(self.LN_1_1(feats)) + feats

        feats = self.attention2(self.LN_2_1(feats)) + feats
        feats = self.ffn2(self.LN_2_2(feats)) + feats

        # feats = self.LN_3_1(feats)
        # feats = self.dropout(feats)

        prediction_bag_conv = self.classification_layer_conv(feats)
        prediction_bag_fc = self.classification_layer_fc(feats.mean(dim=1))

        prediction_bag_conv = F.softmax(prediction_bag_conv.squeeze(dim=2), dim=1)
        prediction_bag_fc = F.softmax(prediction_bag_fc, dim=1)
        # feat是处理后的特征，用于画特征距离图
        # classes是实例级分类结果，prediction_bag_fc是bag_embedding的分类结果
        return classes, prediction_bag_conv, prediction_bag_fc, feats



