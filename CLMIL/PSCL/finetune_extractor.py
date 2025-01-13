from utils.utils import *
from models.moco4me import MoCo4me
from tqdm import tqdm
from loss.losses import SupConLoss
import yaml, argparse
import os, copy, glob, datetime, sys
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, Generic_MOCO_Supcon_Dataset

from utils.supcon_util import *
from torch.utils.data import Dataset
from collections import OrderedDict
global memory_queue, label_queue

def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
def train_epoch(epoch, moco_model, train_loader, optimizer, criterion):
    # global memory_queue  # 全局变量：该变量可以从程序的不同函数或代码部分进行访问和修改，从而在程序的不同部分之间进行共享和访问。
    moco_model.train()

    running_loss, running_acc1, running_acc5, running_count = 0.0, 0.0, 0.0, 0

    # 循环遍历数据批次，并显示进度条
    for batch_idx, (datas, labels) in tqdm(enumerate(train_loader), desc='Processing batches of epoch ' + str(epoch),
                                           total=len(train_loader)):
        # for batch_idx, (data, target) in enumerate(train_loader):
        im_q = datas[0].cuda(non_blocking=True)
        im_k = datas[1].cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        query, key = moco_model(im_q, im_k)
        query = F.normalize(query, dim=-1)
        key  = F.normalize(key,  dim=-1)

        # # warm-up learning rate
        # warmup_learning_rate(args, epoch, batch_idx, len(train_loader), optimizer)
        global memory_queue, label_queue

        if epoch==0 and batch_idx==0:
            memory_queue = key.T
            label_queue = labels.T

        else:
            if len(label_queue) < 15000:
                memory_queue = torch.cat((memory_queue, key.T), dim=1)
                label_queue = torch.cat((label_queue, labels.T), dim=0)
            else:
                memory_queue = torch.cat((memory_queue, key.T), dim=1)[:, key.shape[0]:]  # key中的特征向量按列连接到 memory_queue
                label_queue = torch.cat((label_queue, labels.T), dim=0)[key.shape[0]:]

        # feature = torch.cat([query.unsqueeze(1), key.unsqueeze(1)], dim=1)  # batchsize, 1+memory_queue_size

        loss = criterion(query, labels, memory_queue, label_queue)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        running_count += im_q.shape[0]
        running_loss += loss.item() * im_q.shape[0]

    train_loss_value = running_loss / running_count

    return train_loss_value


def train():
    """The main training function."""
    moco_model = MoCo4me(arch=args.arch, feature_dim=args.feature_dim, moco_momentum=args.momentum,
                         num_class=args.n_classes, mlp=True)  # .to(device)
    '''============================加载image_net模型参数================================='''
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
            state_dict = OrderedDict()
            checkpoint = torch.load(args.resume, map_location="cpu")
            checkpoint_state_dict = checkpoint['state_dict']
            for k, v in checkpoint_state_dict.items():
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    state_dict[k[len('module.encoder_q.'):]] = v

            moco_model.load_state_dict(checkpoint_state_dict, strict=False)

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    '''=================================================================================='''

    moco_model = moco_model.to(device)

    # optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=eval(config['weight_decay']))
    optimizer_moco = torch.optim.SGD(moco_model.parameters(), args.lr_moco, momentum=args.momentum, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_moco, T_max=args.num_epoch4encoder, eta_min=0,
    #                                                        last_epoch=-1)
    criterion_weaksupcon = SupConLoss(temperature=args.temperature)

    model_checkpoints_folder = os.path.join(args.model_save_root, args.expname)

    if not os.path.exists(model_checkpoints_folder):
        os.mkdir(model_checkpoints_folder)

    saving_state = {
        'net': moco_model.state_dict(),
        'optimizer': optimizer_moco.state_dict(),
    }

    torch.save(saving_state, os.path.join(model_checkpoints_folder, 'epoch' + 'init' + 'model.pth'))

    args.model_dir = os.path.join(model_checkpoints_folder, 'epoch_init')

    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    best_loss = 0

    print("======== start training ========")


    for epoch_moco in range(args.num_epochs):
        contrastive_loss = train_epoch(epoch_moco, moco_model, moco_loader, optimizer_moco,
                                            criterion_weaksupcon)
        # print("Epoch {:d}, contrastive_loss_value: {:.4f}".format(epoch_moco, contrastive_loss))

        log_file_path = os.path.join(args.save_dir, 'log_train_round2' + '.txt')
        with open(log_file_path, "a") as file:
            # "a"参数表示以追加模式打开文件，如果文件不存在则创建文件
            file.write(
                "Epoch {:d}, contrastive_loss_value: {:.4f}\n".format(epoch_moco, contrastive_loss))
        print("=======================")

        if epoch_moco == 0:
            best_loss = contrastive_loss
        else:
            if contrastive_loss < best_loss:
                model_path = os.path.join(args.model_save_root, 'week_round2_model_best_smu_2.pth')
                torch.save(
                    {'epoch': epoch_moco, 'state_dict': moco_model.state_dict(),
                     'optimizer': optimizer_moco.state_dict()},
                    model_path)
                best_loss = contrastive_loss



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    ####################################################################################################################
    parser.add_argument('--dataset_name', default='smu_meningiomas', type=str, help='smu_meningiomas, TCGA_gioma')
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    ############################################## MOCO MODEL SETTING ##################################################
    parser.add_argument('--num_epoch4encoder', type=int, default=400, help='epoch to train the sup simCLR')
    parser.add_argument('--batch_size_4encoder', default=40, type=int, help='batch size for MIL training')
    parser.add_argument('--arch', default='resnet50', type=str, help='Specify model architecture (default: resnet50).')
    parser.add_argument('--feature_dim', default=128, type=float, help='Specify initial feature dim (default: 128).')
    parser.add_argument('--moco_momentum', default=0.999, type=float, help='Specify moco momentum of updating key encoder (default: 0.999).')
    parser.add_argument('--queue_size', default=65536, type=int, help='Specify initial queue size (default: 65536).')
    parser.add_argument('--temperature', default=0.07, type=float, help='Specify temperature for training (default: 0.07).')
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--resume', default=r"E:\2021022219\lkycode\model_checkpoint\week_round2_model_best_smu.pth", type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--lr_moco', default=0.001, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--expname', type=str, default='supmoco4me', help='Experiment name: used for comet and save model')
    parser.add_argument('--model_save_root', type=str, default='./model_checkpoint', help='model save root')
    parser.add_argument('--warm_epochs', type=int, default=50)
    parser.add_argument('--warm', type=bool, default=True)
    parser.add_argument('--init_MIL_training', type=bool, default=True, help='whether use init MIL training')
    parser.add_argument('--MIL_every_n_epochs', type=int, default=10, help='conduct MIL training every number of epoch')
    parser.add_argument('--update_pseudo_label', type=bool, default=True, help='whether to update pseudo label')
    parser.add_argument('--csv_path4pseudo_data', default=r"./select_patch/select_patch_smu_weak_round_p10.1,p20.05.csv", type=str, metavar='PATH',
                        help='path to latest pseudo data txt')
    parser.add_argument('--momentum', default=0.9, type=float, help='Specify initial lmomentum (default: 0.9).')
    ####################################################################################################################
    parser.add_argument('--save_dir', default='logs', type=str)
    # parser.add_argument('--feature_data_dir', type=str, default=r'G:\512_imagenet_norm_smu_meningoma_patch_feature',
    #                     help='data directory')

    args = parser.parse_args()
    # define gpu
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)

    # generate dataset and dataloader
    if args.dataset_name == 'smu_meningiomas':
        label_dict = {'pixibaoxing': 0, 'hunhexing': 1, 'xianweixing': 2, 'shalitixing': 3,'feidianxing': 4}
        csv_path = 'generate_split_data/all_data_csv/512_all_meningioma_tumor_subtyping_dummy_clean.csv'
        args.n_classes = 5
    elif args.dataset_name == 'TCGA_gioma':
        label_dict={'Astrocytoma':0, 'Oligodendroglioma':1, 'gbm':2}
        csv_path ='generate_split_data/all_data_csv/512_all_TCGA_GLIOMA_tumor_subtyping_dummy_clean.csv'
        args.n_classes = 4


    #################### dataset and dataloader for moco4me training #################################
    dataset_moco = Generic_MOCO_Supcon_Dataset(data_dir=args.csv_path4pseudo_data)
    moco_loader = torch.utils.data.DataLoader(dataset_moco, batch_size=args.batch_size_4encoder, shuffle=True,
        num_workers=4, pin_memory=True)
    train()
