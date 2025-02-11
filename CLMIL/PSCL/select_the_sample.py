from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, Generic_MOCO_Supcon_Dataset
from generate_feature.datasets.dataset_generic import save_splits
import argparse, os, copy, glob, datetime, sys
from scipy import interp
import random
import models.cdmil_select_patch as mil
from sklearn.metrics import roc_curve, roc_auc_score,precision_recall_fscore_support,auc
from utils.utils import *
from sklearn.preprocessing import label_binarize


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

    def __call__(self, epoch, val_loss, model, ckpt_name='checkpoint.pt'):

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


def train(train_df, milnet, criterion_CE, num_classes, args):
    milnet.eval()
    total_loss = 0
    with torch.no_grad():
        for i, data in enumerate(train_df):

            feats, label_ori, coords, bag_name = data
            label_name = list(label_dict.keys())[label_ori]
            bag_path = os.path.join(args.image_dir, label_name, str(bag_name[0][0]))
            if random.random() > args.drop_probability:  # random.random()随机生成0-1
                feats = dropout_patches(feats, args.drop_p)
            bag_label, bag_feats, bag_coords = (label_ori.cuda(), feats.cuda(), torch.tensor(coords.astype(int)).cuda())

            combined_data = milnet(bag_feats, label_ori, bag_coords, bag_path)

    return combined_data




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
    idx = np.random.choice(np.arange(feats.shape[0]), int(feats.shape[0] * (1 - p)), replace=False)
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


    for i in folds:
        # 初始化模型
        i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
        milnet = mil.MILNet(i_classifier, args.num_classes, args.margin, args.p1, args.p2, CAMEYLON).cuda()
        milnet.load_state_dict(torch.load(r'..\weeksup\mdg_mil_smu_622_weeksup_3gn_0.5iq0.5gq_with_constrast_p10.2_p20.05_round2_s123\s_2_checkpoint.pt'))

        criterion_BCELL = nn.BCEWithLogitsLoss()
        criterion_CE = nn.CrossEntropyLoss()

        # modify save path
        save_path = os.path.join('weights', datetime.date.today().strftime("%m%d%Y"))
        os.makedirs(save_path, exist_ok=True)

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

        train_dataloader = get_split_loader_select(train_split, training=True, testing=args.testing, weighted=args.weighted_sample)  # batch_size=1

        print('Done!')

        combined_data = train(train_dataloader, milnet, criterion_CE, num_classes, args)
        # 指定保存整合后数据的文件路径
        output_file = r'..\select_patch\select_patch_smu_weak_round_p10.05,p20.01.csv'



        # 将整合后的数据保存到CSV文件中
        combined_data.to_csv(output_file, index=False)



        print('\n')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    ####################################################################################################################
    parser.add_argument('--dataset_name', default='smu_meningiomas', type=str, help='smu_meningiomas, TCGA_gioma')
    parser.add_argument('--feature_data_dir', type=str, default=r'G:\512_MOCO_norm_smu_meningoma_patch_feature',
                        help='data directory')
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
                        default='./splits/512_all_smu_6_2_2_k=5_task_2_tumor_subtyping_100',
                        help='manually specify指定 the set of splits to use, '
                             + 'instead of infering from the task and label_frac argument (default: None)')
    parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
    parser.add_argument('--k_start', type=int, default=2, help='start fold (default: -1, last fold)')
    parser.add_argument('--k_end', type=int, default=3, help='end fold (default: -1, first fold)')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--early_stopping', action='store_true', default=True, help='enable early stopping')
    parser.add_argument('--exp_code', type=str, default='mdg_mil_tcga_622_test',
                        help='experiment code for saving results')
    parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
    parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
    ####################################
    parser.add_argument('--temp', default=0.05, type=float)
    parser.add_argument('--momentum', default=0.2, type=float, help='0.2')

    parser.add_argument('--margin', default=-0.1, type=float, )  # 0.2 for Cam
    parser.add_argument('--p1', default=0.05, type=float, help='for cam, 0.01')
    parser.add_argument('--p2', default=0.01, type=float, help='for cam 0.05')
    ####################################

    parser.add_argument('--save_dir', default='logs', type=str)
    parser.add_argument('--data_dir', type=str, default=r'G:\512_weeksup_smu_meningoma_patch_feature',
                        help='data directory')
    parser.add_argument('--image_dir', type=str, default=r'G:\smu_meningoma_svs_cut',
                        help='data directory')


    args = parser.parse_args()
    # define gpu
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)

    # generate dataset and dataloader
    if args.dataset_name == 'smu_meningiomas':
        label_dict = {'pixibaoxing': 0, 'hunhexing': 1, 'xianweixing': 2, 'shalitixing': 3, 'feidianxing': 4}
        csv_path = 'generate_split_data/all_data_csv/512_all_meningioma_tumor_subtyping_dummy_clean.csv'
        args.num_classes = 5
    elif args.dataset_name == 'TCGA_gioma':
        label_dict = {'Astrocytoma': 0, 'Oligodendroglioma': 1, 'glioblastoma': 2}
        csv_path = 'generate_split_data/all_data_csv/512_2021_TCGA_GLIOMA_tumor_subtyping_dummy_clean.csv'
        args.num_classes = 3

    #################### dataset and dataloader for MIL training #################################
    dataset = Generic_MIL_Dataset(csv_path=csv_path,
                                      data_dir=args.feature_data_dir,
                                      shuffle=False,
                                      seed=args.seed,
                                      print_info=False,
                                      label_dict=label_dict,
                                      patient_strat=False,
                                      ignore=[])

    if 'CAMELYON' in args.data_dir:
        CAMEYLON = True
    else:
        CAMEYLON = False
    print('CAMEYLON is {}'.format(CAMEYLON))

    num_classes = args.num_classes

    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    main(args)