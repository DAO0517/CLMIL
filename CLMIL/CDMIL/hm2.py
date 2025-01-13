import torch
import torch.nn.functional as F
from torch import nn, autograd


class HM(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum, features_dict):
        ctx.features = features
        ctx.momentum = momentum
        ctx.feature_dict = features_dict
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, indexes):
            if y.item() not in ctx.feature_dict:
                ctx.feature_dict[y.item()] = []
            ctx.feature_dict[y.item()].append(x.unsqueeze(0))  # Add the feature to the corresponding class

            # Concatenate features for each class and update class features
            for class_index, feature_list in ctx.feature_dict.items():
                if len(feature_list) > 1:
                    if len(ctx.feature_dict[class_index]) > 100:
                        # 超出最大长度，删除最旧的元素（索引为 0 的元素）
                        del ctx.feature_dict[class_index][0]
                    # ctx.feature_dict[class_index] = torch.cat(feature_list, dim=0)
                    center_feature = sum(feature_list)/len(feature_list)# Concatenate features for the same class
                    # Compute mean feature as the new class center
                    # new_class_center =  ctx.feature_dict[class_index].mean(dim=0)
                    ctx.features[class_index] =  ctx.momentum * ctx.features[class_index] + (1. - ctx.momentum) * center_feature
                    ctx.features[class_index] /= ctx.features[class_index].norm()
                else:
                    ctx.features[class_index] = ctx.momentum * ctx.features[class_index] + (1. - ctx.momentum) * feature_list[0]
                    ctx.features[class_index] /= ctx.features[class_index].norm()
            # ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            # ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None, None


def hm(inputs, indexes, features, momentum=0.5, features_dict={}):  # indexed is label
    return HM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device), features_dict)


class HybridMemory(nn.Module):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super(HybridMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples  # n_classes

        self.momentum = momentum
        self.temp = temp
        self.feature_dict = {}

        self.register_buffer('features', F.normalize(torch.zeros(num_samples, num_features), dim=-1).cuda())
        self.register_buffer('labels', torch.zeros(num_samples).long().cuda())

    def forward(self, inputs, indexes, epoch=None):  # memory(output_feats[:, int(label_ori), :], label_ori.cuda(), epoch)
        # inputs: B*2048, features: L*2048
        if epoch is None:
            momentum = self.momentum
        else:
            momentum = self.momentum + 0.3 * (epoch/100)

        inputs = hm(inputs, indexes, self.features, momentum,self.feature_dict)
        inputs /= self.temp
        B = inputs.size(0)

        def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
            exps = torch.exp(vec)
            masked_exps = exps * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return (masked_exps/masked_sums)

        targets = self.labels[indexes].clone()
        labels = self.labels.clone()

        sim = torch.zeros(labels.max()+1, B).float().cuda()  # (n_classes,1)
        sim.index_add_(0, labels, inputs.t().contiguous())
        nums = torch.zeros(labels.max()+1, 1).float().cuda()
        nums.index_add_(0, labels, torch.ones(self.num_samples,1).float().cuda())
        mask = (nums > 0).float()
        sim /= (mask*nums+(1-mask)).clone().expand_as(sim)
        mask = mask.expand_as(sim)
        masked_sim = masked_softmax(sim.t().contiguous(), mask.t().contiguous())
        return F.nll_loss(torch.log(masked_sim+1e-8), targets)