import argparse, os, copy, glob, datetime, sys
from datasets_clam.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset
from datasets_clam.dataset_generic import save_splits
import random
import mdmil_lky as mil
from utils.mdmil_utils import Logger
from sklearn.metrics import roc_curve, roc_auc_score,precision_recall_fscore_support,auc
from hm2 import HybridMemory
from collections import defaultdict
from utils.utils import *
from sklearn.preprocessing import label_binarize


def get_bag_labels_ori(labels, num_classes):  
    label_one_hot_real = torch.zeros(labels.size(0), num_classes)
    for i in range(labels.size(0)):
        label_one_hot_real[i, labels[i]] = 1
    return label_one_hot_real

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss



def train(train_df, milnet, criterion_BCELL, criterion_CE,  optimizer, num_classes, args, epoch, memory):
    milnet.train()
    total_loss = 0

    for i, data in enumerate(train_df):
        A = i
        optimizer.zero_grad()
        feats, label_ori = data
        # label = get_bag_labels_ori(label_ori, num_classes)
        if random.random()>args.drop_probability:   # random.random()随机生成0-1
            feats = dropout_patches(feats, args.drop_p)
        length = feats.size(0)
        bag_label, bag_feats = (label_ori.cuda(), feats.cuda())
        
        # classes, prediction_bag_conv, prediction_bag_fc, feats
        ins_prediction, bag_prediction_conv, bag_prediction_fc, output_feats = milnet(bag_feats)
        
        # # for contrastive loss
        output_feats = F.normalize(output_feats, dim=-1)  # (bag,n_classes,feat)
        contrastive_loss = memory(output_feats[:, int(label_ori), :], label_ori.cuda(), epoch)

        # for instance loss and bag loss
        max_prediction, _ = torch.max(ins_prediction, 0)  # 按列，返回维度为（5，）返回的是每个类别的最大概率，索引对应的是样本索引。
        # bag_label = bag_label.long().squeeze()
        # criterion_CE(max_prediction[None,], bag_label)让对应类的预测值最高
        bag_loss =  criterion_CE(max_prediction[None,], bag_label) + criterion_CE(bag_prediction_fc, bag_label)

        # overall loss
        loss = bag_loss  + 0.6 *  contrastive_loss
        # loss = bag_loss
        # with autograd.detect_anomaly():
        #     loss.backward()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(milnet.parameters(), 2, norm_type=2)
        # print(list(milnet.children())[3].weight.grad)
        optimizer.step()
        total_loss = total_loss + loss.item()

    return total_loss / len(train_df)


def test(test_df, milnet, criterion_BCELL, criterion_CE, optimizer, num_classes, args):
    milnet.eval()

    total_loss = 0
    test_labels = []
    test_predictions = []
    test_prediction_label = []
    test_ori_labels = []
    right, count = 0, 0
    dic_num = defaultdict(int)
    dic_right_num = defaultdict(int)
    with torch.no_grad():
        for i, data in enumerate(test_df):
            feats, label_ori = data
            label = get_bag_labels_ori(label_ori, num_classes)

            bag_label, bag_feats = label_ori.cuda(), feats.cuda()
            ins_prediction, bag_prediction_conv, bag_prediction_fc, feats = milnet(bag_feats)

            # for loss calculation
            max_prediction, _ = torch.max(ins_prediction, 0)
            loss = criterion_CE(max_prediction[None,], bag_label) + criterion_CE(bag_prediction_fc, bag_label)
            total_loss = total_loss + loss.item()

            test_labels.extend([label])
            test_prediction = F.softmax(bag_prediction_fc, dim=1).cpu()
            # test_predictions.extend([torch.sigmoid(bag_prediction).cpu()])
            temp = torch.argmax(test_prediction).cpu()

            dic_num[int(label_ori)] += 1
            if temp == label_ori:
                right += 1
                dic_right_num[int(label_ori)] += 1

            test_predictions.extend([test_prediction])
            test_prediction_label.append(temp)
            label_cpu = int(label_ori)
            test_ori_labels.append(label_cpu)

        count = i + 1

    for key in sorted(dic_num.keys()):
        print('Accuracy of class {} is {}'.format(key, dic_right_num[key] / dic_num[key]))

    test_labels = torch.cat(test_labels, dim=0)
    test_predictions = torch.cat(test_predictions, dim=0)
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)
    test_prediction_label = np.array(test_prediction_label)
    test_ori_labels = np.array(test_ori_labels)

    # Compute Precision, Recall, and F1-score for each class
    precision, recall, f1, _ = precision_recall_fscore_support(test_ori_labels, test_prediction_label, average=None)
    # Calculate the average across all classes
    average_precision = precision.mean()
    average_recall = recall.mean()
    average_f1 = f1.mean()
    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, num_classes, pos_label=1)

    for i in range(num_classes):
        class_prediction_bag = copy.deepcopy(test_predictions[:, i])
        class_prediction_bag[test_predictions[:, i] >= thresholds_optimal[i]] = 1
        class_prediction_bag[test_predictions[:, i] < thresholds_optimal[i]] = 0
        test_predictions[:, i] = class_prediction_bag
    bag_score = 0
    for i in range(0, len(test_df)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i].squeeze()) + bag_score

    avg_score = right / count
    return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal, average_precision, average_recall, average_f1, test_labels, test_predictions


def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape) == 1:
        predictions = predictions[:, None]

    binary_labels = label_binarize(labels, classes=[i for i in range(num_classes)])
    for c in range(num_classes):
        label = binary_labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def dropout_patches(feats, p):
    idx = np.random.choice(np.arange(feats.shape[0]), int(feats.shape[0]*(1-p)), replace=False)
    sampled_feats = feats[idx, :]
    return sampled_feats


def main(args):
    # for reproduction
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # for log save
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end
    folds = np.arange(start, end)

    sys.stdout = Logger(os.path.join(args.save_dir, 'log_train' + str(len(os.listdir(args.save_dir)) + 1) + '.txt'))
    label_dict = {'Astrocytoma': 0, 'Oligodendroglioma': 1, 'Glioblastoma': 2}


    for i in folds:
        # 初始化模型
        i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=num_classes).cuda()
        milnet = mil.MILNet(i_classifier, num_classes, args.margin, args.p1, args.p2, CAMEYLON).cuda()
        optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
        if args.scheduler == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        criterion_BCELL = nn.BCEWithLogitsLoss()
        criterion_CE = nn.CrossEntropyLoss()

        # modify save path
        save_path = os.path.join('weights', datetime.date.today().strftime("%m%d%Y"))
        os.makedirs(save_path, exist_ok=True)

        # for memory bank
        memory = HybridMemory(512, num_classes, temp=args.temp, momentum=args.momentum)
        memory.labels = torch.arange(num_classes).cuda()

        # 定义数据define data
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False,
                                                                         csv_path='{}/splits_{}.csv'.format(
                                                                             args.split_dir, i))
        datasets = (train_dataset, val_dataset, test_dataset)
        print('\nInit train/val/test splits...', end=' ')
        train_split, val_split, test_split = datasets
        save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(i)))
        print('Done!')
        print("Training on {} samples".format(len(train_split)))
        print("Validating on {} samples".format(len(val_split)))
        print("Testing on {} samples".format(len(test_split)))

        train_dataloader = get_split_loader(train_split, training=True, testing=args.testing, weighted=args.weighted_sample)  # batch_size=1
        val_dataloader = get_split_loader(val_split, testing=args.testing)
        test_dataloader = get_split_loader(test_split, testing=args.testing)

        if args.early_stopping:
            early_stopping = EarlyStopping(patience=40, stop_epoch=70, verbose=True)

        else:
            early_stopping = None
        print('Done!')


    # start training and testing
        best_score_val = 0.0
        best_score_test = 0.0
        for epoch in range(1, args.num_epochs + 1):
            train_loss_bag = train(train_dataloader, milnet, criterion_BCELL, criterion_CE, optimizer, num_classes, args, epoch, memory)
            lr = scheduler.get_last_lr()
            print('Epoch {}/{}: average loss ->{:4f}  lr ->{}'.format(epoch, args.num_epochs, train_loss_bag, lr))
            scheduler.step()

            val_loss_bag, val_avg_score, val_aucs, _, val_precision, val_recall, val_f1, val_labels, val_predictions = test(val_dataloader, milnet, criterion_BCELL, criterion_CE, optimizer, num_classes, args)
            print('Val Set: Epoch [%d/%d] val loss: %.4f, average score: %.4f, val_precision: %.4f, val_recall: %.4f,val_f1: %.4f, AUC: ' %(epoch, args.num_epochs, val_loss_bag, val_avg_score, val_precision, val_recall, val_f1) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(val_aucs)))

            current_score = sum(val_aucs)/len(val_aucs) + val_avg_score
            if current_score > best_score_val:
                best_score_val = current_score
                save_name = os.path.join(save_path, 'best_model_val_s{}.pth'.format(i))
                torch.save(milnet.state_dict(), save_name)
                print('Get best val Model')
            if early_stopping:
                assert args.results_dir
                early_stopping(epoch, val_loss_bag, milnet,
                               ckpt_name=os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(i)))
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        if args.early_stopping:
            milnet.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(i))))
            # milnet.load_state_dict(torch.load(os.path.join(save_path, 'best_model_val_s{}.pth'.format(i))))
        else:
            torch.load(milnet.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(i)))

        test_loss_bag, test_avg_score, test_aucs, _, test_precision, test_recall, test_f1,test_labels, test_predictions = test(test_dataloader, milnet, criterion_BCELL, criterion_CE, optimizer, num_classes, args)
        print('Testing Set: Fold:%d test loss: %.4f, average score: %.4f, test_precision: %.4f, test_recall: %.4f,test_f1: %.4f, AUC: ' %(i, test_loss_bag, test_avg_score, test_precision, test_recall, test_f1) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(test_aucs)))
        print('\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    # must check before training
    parser.add_argument('--lr', default=0.0001, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of total training epochs [40|200]')

    parser.add_argument('--split_train', default=0.65, type=float, help='Training/Validation split [0.5]')
    parser.add_argument('--split_val', default=0.15, type=float, help='Training/Validation split [0.5]')
    parser.add_argument('--eva_epoch', default=1, type=int)
    ###################################
    parser.add_argument('--drop_p', default=0.01, type=float, help='drop portion during training')
    parser.add_argument('--drop_probability', default=0.6, type=float, help='drop portion during training')
    parser.add_argument('--scheduler', default='cos', type=str, help='type of schedular')
    ###################################
    parser.add_argument('--feats_size', default=1024, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
    parser.add_argument('--split_dir', type=str,
                        default='./splits/512_2021_tcga_glioma_6_2_2_k=5_task_2_tumor_subtyping_100',
                        help='manually specify指定 the set of splits to use, '
                             + 'instead of infering from the task and label_frac argument (default: None)')
    parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping'],
                        default='task_2_tumor_subtyping')
    parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
    parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
    parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--seed', type=int, default=2,
                        help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--early_stopping', action='store_true', default=True, help='enable early stopping')
    parser.add_argument('--exp_code', type=str, default='test_mdg_mil_tcga_moco_622_with_3gn_0.5iq+0.5gq_with_0.6constrast_loss_p10.20_p20.05',
                        help='experiment code for saving results')
    parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
    parser.add_argument('--weighted_sample', action='store_true', default=True, help='enable weighted sampling')
    ####################################
    parser.add_argument('--temp', default=0.05, type=float)
    parser.add_argument('--momentum', default=0.2, type=float, help='0.2')

    parser.add_argument('--margin', default=-0.1, type=float, )  # 0.2 for Cam
    parser.add_argument('--p1', default=0.20, type=float, help='for cam, 0.01')
    parser.add_argument('--p2', default=0.05, type=float, help='for cam 0.05')
    ####################################

    parser.add_argument('--save_dir', default='logs', type=str)
    parser.add_argument('--data_dir', type=str, default=r'G:\512_MOCO_norm_2021TCGA_patch_feature',
                        help='data directory')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)


    def seed_torch(seed=7):
        import random
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


    seed_torch(args.seed)

    if 'CAMELYON' in args.data_dir:
        CAMEYLON = True
    else:
        CAMEYLON = False
    print('CAMEYLON is {}'.format(CAMEYLON))



    # generate dataset and dataloader
    if args.task == 'task_1_tumor_vs_normal':
        args.n_classes = 2
        dataset = Generic_MIL_Dataset(csv_path='dataset_csv/camelyon16_tumor_vs_normal_dummy_clean.csv',
                                      data_dir=args.data_dir,
                                      shuffle=False,
                                      seed=args.seed,
                                      print_info=False,
                                      label_dict={'normal_tissue': 0, 'tumor_tissue': 1},
                                      patient_strat=False,
                                      ignore=[])

    elif args.task == 'task_2_tumor_subtyping':
        args.n_classes = 3
        dataset = Generic_MIL_Dataset(csv_path='dataset_csv/512_2021_TCGA_GLIOMA_tumor_subtyping_dummy_clean.csv',
                                      data_dir=args.data_dir,
                                      shuffle=False,
                                      seed=args.seed,
                                      print_info=False,
                                      label_dict={'Astrocytoma': 0, 'Oligodendroglioma': 1, 'glioblastoma': 2},
                                      patient_strat=False,
                                      ignore=[])

    num_classes = args.n_classes

    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    main(args)
