import argparse, os, copy, glob, datetime, sys
from datasets_clam.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset
from datasets_clam.dataset_generic import save_splits
import random
import model_mdmil as mil
from utils.mdmil_utils import Logger
from sklearn.metrics import roc_curve, roc_auc_score,precision_recall_fscore_support,auc, confusion_matrix
from hm2 import HybridMemory
from collections import defaultdict
from utils.utils import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize



def get_bag_labels_ori(labels, num_classes):  
    label_one_hot_real = torch.zeros(labels.size(0), num_classes)
    for i in range(labels.size(0)):
        label_one_hot_real[i, labels[i]] = 1
    return label_one_hot_real


def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=100):
    """

    @param label_true: [0,1,2,7,4,5,...]
    @param label_pred: [0,5,4,2,1,4,...]
    @param label_name: ['cat','dog','flower',...]
    @param title: Confusion Matrix
    @param pdf_save_path: pdf_save_path=xxx.png | xxx.pdf | ...等其他plt.savefig支持的保存格式
    @param dpi: 100
    @return:

    example：
            draw_confusion_matrix(label_true=y_gt,
                          label_pred=y_pred,
                          label_name=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
                          title="Confusion Matrix on Fer2013",
                          pdf_save_path="Confusion_Matrix_on_Fer2013.png",
                          dpi=300)

    """
    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='true')
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.imshow(cm, cmap='Blues',vmin=0,vmax=1)
    plt.title(title,fontsize=17,fontweight='bold')
    plt.xlabel("Predict label",fontsize=15,fontweight='bold')
    plt.ylabel("Truth label",fontsize=15,fontweight='bold')
    plt.yticks(range(label_name.__len__()), label_name,fontsize=15,fontweight='bold')
    plt.xticks(range(label_name.__len__()), label_name,fontsize=15,fontweight='bold')

    plt.tight_layout()

    plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0) 
            value = float(format('%.2f' % cm[j, i]))
            plt.text(i, j, value, fontsize=16,fontweight='bold',verticalalignment='center', horizontalalignment='center', color=color)


    plt.savefig(pdf_save_path)
    plt.show()


def test(test_df, milnet, criterion_BCELL, criterion_CE, optimizer, num_classes, args, fold=1):
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
    test_predictions1 = np.array(test_predictions)
    test_predictions = np.array(test_predictions)
    test_prediction_label = np.array(test_prediction_label)
    test_ori_labels = np.array(test_ori_labels)

    # Compute Precision, Recall, and F1-score for each class
    precision, recall, f1, _ = precision_recall_fscore_support(test_ori_labels, test_prediction_label, average=None)
    pdf_save_path=os.path.join(args.results_dir,'Confusion Matrix for Meningioma'+str(fold)+'.png')
    draw_confusion_matrix(test_ori_labels, test_prediction_label, args.label_name, title="Confusion Matrix for Meningioma", pdf_save_path=pdf_save_path,
                          dpi=300)

    # Calculate the average across all classes
    average_precision = precision.mean()
    average_recall = recall.mean()
    average_f1 = f1.mean()
    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions1, num_classes, pos_label=1)

    for i in range(num_classes):
        print('class %d precision %.4f, recall %.4f, f1: %.4f' %(i, precision[i], recall[i], f1[i]))
        class_prediction_bag = copy.deepcopy(test_predictions[:, i])
        class_prediction_bag[test_predictions[:, i] >= thresholds_optimal[i]] = 1
        class_prediction_bag[test_predictions[:, i] < thresholds_optimal[i]] = 0
        test_predictions[:, i] = class_prediction_bag
    bag_score = 0
    for i in range(0, len(test_df)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i].squeeze()) + bag_score

    avg_score = right / count
    return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal, average_precision, average_recall, average_f1, test_labels, test_predictions,test_predictions1
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

        # enter the model path   
        test_model_path = os.path.join(args.test_model_path, "best_model_val_s{}.pth".format(i))

        # start  testing    
        milnet.load_state_dict(torch.load(test_model_path))      
        test_loss_bag, test_avg_score, test_aucs, _, test_precision, test_recall, test_f1,test_labels, test_predictions, test_predictions1 = test(test_dataloader, milnet, criterion_BCELL, 
        criterion_CE, optimizer, num_classes, args,fold=i)
        print('Testing Set: Fold:%d test loss: %.4f, average score: %.4f, test_precision: %.4f, test_recall: %.4f,test_f1: %.4f, AUC: ' %(i, test_loss_bag, test_avg_score, test_precision, 
        test_recall, test_f1) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(test_aucs)))
 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')

    parser.add_argument('--lr', default=0.0001, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--eva_epoch', default=1, type=int)
    parser.add_argument('--drop_p', default=0.01, type=float, help='drop portion during training')
    parser.add_argument('--drop_probability', default=0.6, type=float, help='drop portion during training')
    parser.add_argument('--scheduler', default='cos', type=str, help='type of schedular')
    ###################################
    parser.add_argument('--feats_size', default=1024, type=int, help='Dimension of the feature size [512]')
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
    parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
    parser.add_argument('--weighted_sample', action='store_true', default=True, help='enable weighted sampling')
    ####################################
    parser.add_argument('--temp', default=0.05, type=float)
    parser.add_argument('--momentum', default=0.2, type=float, help='0.2')
    parser.add_argument('--margin', default=-0.1, type=float, )  # 0.2 for Cam
    parser.add_argument('--p1', default=0.20, type=float, help='for cam, 0.01')
    parser.add_argument('--p2', default=0.05, type=float, help='for cam 0.05')
    ###############  must check before testing #####################
    parser.add_argument('--task', type=str, choices=['Meningioma', 'TCGA_GLIOMA'],
                        default='Meningioma', help='select dataset task')
    parser.add_argument('--results_dir', default='CDMIL/results', help='results directory (default: ./results)')
    parser.add_argument('--exp_code', type=str, default=r'CDMIL\test_mdg_mil_tcga_moco_622_with_3gn_0.5iq+0.5gq_with_0.6constrast_loss_p10.20_p20.05',
                        help='experiment code for saving results')
    parser.add_argument('--save_dir', default='logs', type=str)
    parser.add_argument('--data_dir', type=str, default=r"I:\512_weeksup_smu_meningoma_patch_feature",
                        help='data directory')
    parser.add_argument('--split_dir', type=str,
                        default=r'CDMIL\splits\512_all_smu_6_2_2_k=5_task_2_tumor_subtyping_100',
                        help='manually specify the set of splits to use, '
                             + 'instead of infering from the task and label_frac argument (default: None)')
    parser.add_argument('--test_model_path', type=str, default=r"I:\save_model_test",
                help='data directory for load the testing model')                  
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
    if args.task == 'Meningioma':
        args.n_classes = 5
        args.label_name=['MM', 'TM', 'FM', 'PM', 'AM']
        dataset = Generic_MIL_Dataset(csv_path=r'CDMIL\dataset_csv\512_all_meningioma_tumor_subtyping_dummy_clean.csv',
                                      data_dir=args.data_dir,
                                      shuffle=False,
                                      seed=args.seed,
                                      print_info=False,
                                      label_dict={'pixibaoxing': 0, 'hunhexing': 1, 'xianweixing': 2, 'shalitixing': 3,
                                                  'feidianxing': 4},
                                      patient_strat=False,
                                      ignore=[])

    elif args.task == 'TCGA_GLIOMA':
        args.n_classes = 3
        args.label_name=['Astrocytoma', 'Oligodendroglioma', 'glioblastoma']
        dataset = Generic_MIL_Dataset(csv_path=r'CDMIL\dataset_csv\512_2021_TCGA_GLIOMA_tumor_subtyping_dummy_clean.csv',
                                      data_dir=args.data_dir,
                                      shuffle=False,
                                      seed=args.seed,
                                      print_info=False,
                                      label_dict={'Astrocytoma': 0, 'Oligodendroglioma': 1, 'glioblastoma': 2},
                                      patient_strat=False,
                                      ignore=[])

    num_classes = args.n_classes

    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)
    args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)
    main(args)
