
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import pandas as pd
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import random
import shutil
from models.gcn import *
from patch_predict import AverageMeter
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from utils import predict_binary
from tensorboardX import SummaryWriter
from time import gmtime, strftime, localtime
from itertools import cycle
import matplotlib.pyplot as plt


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='GCN', type=str,
                    help='baseline of the model')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--n_epoch', default=10, type=int,
                    help='number of epoch to change')
parser.add_argument('--epoch', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('--fold_index', default=0, type=int,
                    help='index of current fold')
parser.add_argument('--optimizer', default='Adam', type=str,
                    help='optimizer (Adam)')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--num_classes', default=1, type=int,
                    help='numbers of classes (default: 2)')
parser.add_argument('--tensorboard', default=True,
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--use_cuda', default=True,
                    help='whether to use_cuda(default: True)')
parser.add_argument('--batch_size', default=20, type=int,
                    help='mini-batch size (default: 20)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--weight_bg', default=0.5, type=float,
                    help='weight of background negative data in loss function')
parser.add_argument("--result_path", default=None, type=str,
                    help="result path for cnn experiment.")
parser.add_argument("--result_path_gcn", default=None, type=str,
                    help="result path for gcn experiment.")
parser.add_argument("--gpu", type=str, default="0", metavar="N",
                    help="input visible devices for training (default: 0)")
parser.add_argument('--seed', default=2, type=int, help='random seed(default: 1)')


def get_graph_data(fold, mode, path):
    result_path = path
    a = []
    for i in range(32):
        a.append('feat' + str(i))
    prob_feat_train = pd.read_csv(os.path.join(result_path, str(fold) + '_test_probs_' + mode + '.csv'))

    prob_feat_train.drop(['id_loc'], axis=1, inplace=True)
    if mode == 'train' or mode == 'valid':
        prob_feat_train.drop(['label'], axis=1, inplace=True)
    assert prob_feat_train.shape[1] == 40

    patient_id = prob_feat_train.iloc[:, 0]
    patient_id = patient_id.drop_duplicates()
    patient_id = patient_id.reset_index(drop=True)

    adjacency = []
    node_feat = []
    patient_y = []
    id = []
    adj_bg = []
    node_feat_bg = []
    patient_y_bg = []
    id_bg = []  #
    for i in range(len(patient_id)):
        id_i = patient_id.loc[i]
        patient_i = prob_feat_train[prob_feat_train['id'].isin([id_i])]
        if len(patient_i) < 50:
            print('delete:', id_i, ' len:', len(patient_i))
            continue

        patient_i_sort = patient_i.sort_values('prob', ascending=False)
        patient_i_choose = patient_i_sort.iloc[:50, :]
        graph_i = Graph(patient_i_choose)
        adjacency_i, node_feat_i = graph_i.ad_relationship()  # adj:(50*50), feat:(50*33)
        adjacency.append(adjacency_i)
        node_feat.append(node_feat_i)  # (n*50*33)

        if id_i[:2] == 'PC':
            label = 1
        else:
            label = 0
        patient_y.append(label)
        id.append(id_i)

        if mode=='train' and label == 1 and len(patient_i) >= 100:
            patient_i_bg = patient_i_sort.iloc[-50:, :]
            graph_i_bg = Graph(patient_i_bg)
            adjacency_i_bg, node_feat_i_bg = graph_i_bg.ad_relationship()
            id_i_bg = 'BG_' + id_i
            adj_bg.append(adjacency_i_bg)
            node_feat_bg.append(node_feat_i_bg)
            patient_y_bg.append(0)
            id_bg.append(id_i_bg)
            adj_bg.append(adjacency_i)
            node_feat_bg.append(node_feat_i)
            patient_y_bg.append(label)
            id_bg.append(id_i)

    if mode=='train':
        return np.array(id), adjacency, node_feat, patient_y, np.array(id_bg), adj_bg, node_feat_bg, patient_y_bg
    else:
        return np.array(id), adjacency, node_feat, patient_y


# 计算两个patch之间的空间欧氏距离
def cal_euclidean(a, b):
    a_2 = a ** 2
    b_2 = b ** 2
    sum_a_2 = torch.sum(a_2, dim=1).unsqueeze(1)  # [m, 1]
    sum_b_2 = torch.sum(b_2, dim=1).unsqueeze(0)  # [1, n]
    bt = b.t()
    return sum_a_2 + sum_b_2 - 2 * a.mm(bt)


class Graph(object):
    def __init__(self, patient_df):
        self.count = patient_df.shape[0]
        self.adj_mat = [[None for i in range(self.count)] for i in range(self.count)]
        self.df = patient_df
        self.node_feat = []
        for i in range(self.count):
            feat_idx = [2] + [i for i in range(8, 40)]  # [prob, 32 features]
            self.node = self.df.iloc[i, feat_idx]
            self.node_feat.append(self.node)

    def ad_relationship(self):
        for i in range(self.count):
            x = torch.tensor(self.df.iloc[:, 3].tolist()) * 0.75
            y = torch.tensor(self.df.iloc[:, 4].tolist()) * 0.75
            z = torch.tensor(self.df.iloc[:, 7].tolist()) * 5
            points = torch.stack([x, y, z], dim=1)
            self.adj_mat = torch.exp(- cal_euclidean(points, points) / 1800)
        adjacency = np.array(self.adj_mat)
        adjacency = adj_norm(adjacency) # 邻接矩阵归一化

        return adjacency, self.node_feat


# Graph dataset
class graph_dataset(Dataset):
    def __init__(self, id, adjacency, feature, y):
        self.id = id
        self.y = y
        self.adjacency = adjacency
        self.feature = feature

    def __getitem__(self, index):
        case_id = self.id[index]
        adjacency_i = self.adjacency[index]
        feature_i = self.feature[index]
        label = self.y[index]
        return adjacency_i, torch.Tensor(feature_i), label, case_id

    def __len__(self):
        return len(self.id)


def adj_norm(A):
    assert A.shape[0] == A.shape[1]
    # I = np.eye(A.shape[0])
    # A_hat = A + I
    D = np.sum(A, axis=0)
    d_inv_sqrt = np.power(D, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    A_norm = A.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

    return A_norm


def train_gcn(train_loader, train_loader_bg, weight_bg, model, criterion, optimizer, epoch, use_cuda=True):
    """Train for one epoch on the training set"""
    train_losses = AverageMeter()
    train_acc = AverageMeter()
    train_acc_bg = AverageMeter()
    chl2 = 8
    feat_pool = torch.zeros((0, chl2))
    feat_pool_bg = torch.zeros((0, chl2))
    target_roc = torch.zeros((0, args.num_classes))
    target_roc_bg = torch.zeros((0, args.num_classes))
    # switch to train mode
    model.train()

    with tqdm(zip(train_loader, cycle(train_loader_bg)), ncols=130) as t:
        for i, ([adjacency, feature, target, case_id], [adjacency_bg, feature_bg, target_bg, case_id_bg]) in enumerate(t):
            t.set_description("train epoch %s" % epoch)
            if use_cuda:
                # target = target.unsqueeze(1).type(torch.FloatTensor).cuda()
                target = target.type(torch.FloatTensor).cuda()
                adjacency = adjacency.type(torch.FloatTensor).cuda()
                feature = feature.type(torch.FloatTensor).cuda()
                # 新构建的负样本数据
                target_bg = target_bg.type(torch.FloatTensor).cuda()
                adjacency_bg = adjacency_bg.type(torch.FloatTensor).cuda()
                feature_bg = feature_bg.type(torch.FloatTensor).cuda()

            optimizer.zero_grad()
            output = model(feature, adjacency)
            output_bg = model(feature_bg, adjacency_bg)
            target_roc = torch.cat((target_roc, target.to(torch.float32).unsqueeze(1).data.cpu()), dim=0)
            target_roc_bg = torch.cat((target_roc_bg, target_bg.to(torch.float32).unsqueeze(1).data.cpu()), dim=0)

            train_loss = (criterion(output.view(output.shape[0]), target) + weight_bg * criterion(output_bg.view(output_bg.shape[0]), target_bg)) / (
                        weight_bg + 1)
            train_losses.update(train_loss.item(), adjacency.size(0))

            acc = accuracy(output.data, target)
            train_acc.update(acc, adjacency.size(0))

            acc_bg = accuracy(output_bg.data, target_bg)
            train_acc_bg.update(acc_bg, adjacency_bg.size(0))

            # compute gradient and do SGD step
            train_loss.backward()
            optimizer.step()

            t.set_postfix({
                'iter': '{i}'.format(i=i),
                'loss': '{loss.val:.4f}({loss.avg:.4f})'.format(loss=train_losses),
                'Acc': '{acc.val:.4f}({acc.avg:.4f})'.format(acc=train_acc)}
            )

    return train_losses, train_acc, train_acc_bg


def valid_gcn(val_loader, model, criterion, optimizer, epoch, use_cuda=True):
    """valid for one epoch on the validation set"""
    val_losses = AverageMeter()
    val_acc = AverageMeter()
    chl2 = 8
    # switch to train mode
    model.eval()
    target_roc = torch.zeros((0, args.num_classes))
    output_roc = torch.zeros((0, args.num_classes))

    with tqdm(val_loader, ncols=130) as t:
        for i, (adjacency, feature, target, case_id) in enumerate(t):
            t.set_description("train epoch %s" % epoch)
            if use_cuda:
                target = target.type(torch.FloatTensor).cuda()
                adjacency = adjacency.type(torch.FloatTensor).cuda()
                feature = feature.type(torch.FloatTensor).cuda()
                # print('feature：', feature)

            optimizer.zero_grad()
            output = model(feature, adjacency)
            val_loss = criterion(output.view(output.shape[0]), target)
            val_losses.update(val_loss.item(), adjacency.size(0))

            target_roc = torch.cat((target_roc, target.to(torch.float32).unsqueeze(1).data.cpu()), dim=0)
            output_roc = torch.cat((output_roc, output.data.cpu()), dim=0)

            acc = accuracy(output.data, target)
            val_acc.update(acc, adjacency.size(0))

            # compute gradient and do SGD step
            val_loss.backward()
            optimizer.step()

            t.set_postfix({
                'loss': '{loss.val:.4f}({loss.avg:.4f})'.format(loss=val_losses),
                'Acc': '{acc.val:.4f}({acc.avg:.4f})'.format(acc=val_acc)}
            )
        AUROC = aucrocs(output_roc, target_roc)
        print('The AUROC is %.4f' % AUROC)

        fpr, tpr, _ = roc_curve(target_roc, output_roc)
        plt.rc('font', family='Times New Roman')
        plt.figure()
        lw = 2
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=lw,
            label="ROC curve (area = %0.2f)" % AUROC,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic curve (fold "+ str(args.fold_index) +")")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(result_path_gcn,'ROC_val_fold'+str(args.fold_index)+'.png'), bbox_inches='tight')

    return val_losses, val_acc, val_acc.avg, AUROC


def test(test_loader, model, mode, use_cuda=True):
    probs = []
    model.eval()
    target_roc = torch.zeros((0, args.num_classes))
    output_roc = torch.zeros((0, args.num_classes))

    names = []
    y_patient = []
    with torch.no_grad():
        with tqdm(test_loader, ncols=130) as t:
            for i, (adjacency, feature, target, case_id) in enumerate(t):
                t.set_description("Calculate {} probs:".format(mode))
                if use_cuda:
                    adjacency = adjacency.type(torch.FloatTensor).cuda()
                    feature = feature.type(torch.FloatTensor).cuda()

                names.extend(case_id)
                # compute output
                output = model(feature, adjacency)
                output = output.detach().cpu()

                target_roc = torch.cat((target_roc, target.to(torch.float32).unsqueeze(1).data.cpu()), dim=0)
                output_roc = torch.cat((output_roc, output), dim=0)

                for bs in range(output.shape[0]):
                    probs.append(output.numpy()[bs])
                    y_patient.append(target.numpy()[bs])

        if mode != 'renmin' and mode != 'sixth':
            AUROC = aucrocs(output_roc, target_roc)
            print('The AUROC is %.4f' % AUROC)

    y_patient = list(map(int, y_patient))
    probs = np.array(probs)
    if mode == 'valid':
        return names, y_patient, probs, AUROC, target_roc, output_roc,
    elif mode != 'renmin' and mode != 'sixth':
        return names, y_patient, probs, AUROC
    else:
        return names, y_patient, probs


def aucrocs(output, target):  # 改准确度的计算方式

    """
    Returns:
    List of AUROCs of all classes.
    """
    output_np = output.cpu().numpy()
    target_np = target.cpu().numpy()
    AUROCs=roc_auc_score(target_np[:, 0], output_np[:, 0])
    return AUROCs


def adjust_learning_rate(optimizer, epoch, n_epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    print('epoch >= n_epoch', epoch >= n_epoch)
    print(' epoch % 10 == 0', epoch % 20 == 0)
    if epoch >= n_epoch and epoch % 20 == 0:
        print('Adjusting by 0.1')
        lr = lr * 0.1
    else:
        # lr = args.lr * (1 + np.cos((epoch - args.n_epoch) * math.pi / args.epoch)) / 2
        lr = lr
    if lr <= 0.0001:
        lr = 0.0001
    # log to TensorBoard
    writer.add_scalar('learning_rate', lr, epoch)
    print('adjusting lr:', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('lr:', lr)
    return lr


def save_checkpoint(state, is_best, epoch, fold):
    """Saves checkpoint to disk"""
    # filename = 'checkpoint' + str(fold) + '_' + str(epoch) + '.pth.tar'
    filename = 'checkpoint' + str(fold) + '.pth.tar'
    directory = result_path_gcn + "/checkpoint_gcn/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        torch.save(state, filename)
        shutil.copyfile(filename, directory + 'model_best' + str(fold) + '.pth.tar')


# 读取id，y，pred，prob，写入文件
def save_probs_pred(result_path_gcn, names, y, probs, pred, mode):
    patch_prob_filename = os.path.join(result_path_gcn, '{}_test_results_{}.csv'.format(args.fold_index, mode))
    info = []
    for i in range(len(names)):
        info.append([names[i], y[i], int(pred[i]), probs[i][0]])
    df = pd.DataFrame(info, columns=['id', 'label', 'pred', 'prob'])
    df.to_csv(patch_prob_filename, index=False)


def save_result_matrix(true,pred,auc,mode,time,save_result):
    tn,fp,fn,tp = confusion_matrix(true, pred, labels=[0, 1]).ravel()
    acc = (tn+tp)/(tn+fp+fn+tp)
    sen = tp/(tp+fn)
    spe = tn/(tn+fp)
    NPV = tn/(tn+fn)
    PPV = tp/(tp+fp)
    print('{}: {}  tn:{},fp:{},fn:{},tp:{}'.format(mode,time,tn,fp,fn,tp))
    print('{}: {} acc is {}'.format(mode,time,acc))
    print('{}: {} sen is {}, spe is {}'.format(mode,time,sen,spe))
    save_result.write(' '+','+mode+','+str(acc)+','+str(sen)+','+str(spe)
                      +','+str(tn)+','+str(tp)+','+str(fn)+','+str(fp)+','+str(NPV)+','+str(PPV)+','+str(auc)+'\n')


def accuracy(output, target):
    output_np = output.cpu().numpy()
    pred = predict_binary(output_np, 0.5)

    target_np = target.cpu().numpy()

    right = (pred.squeeze() == target_np)
    acc = np.sum(right) / output.shape[0]

    return acc


if __name__ == '__main__':
    global use_cuda
    args = parser.parse_args()
    use_cuda = args.use_cuda and torch.cuda.is_available()

    gpu = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    learning_rate = args.lr
    weight_decay = args.weight_decay
    total_epoch = args.epoch
    start_epoch = args.start_epoch
    batch_size = args.batch_size
    lr_factor = 0.1
    patience = 10
    seed_torch(args.seed)
    weight_bg = args.weight_bg

    result_basepath = args.result_path
    run_name = strftime("%Y%m%d_%H%M%S", localtime())
    result_path_gcn = os.path.join(result_basepath, args.result_path_gcn)

    fold = args.fold_index
    # get patches and create graphs
    print("fold: ", fold)
    model = GCN()
    model = model.cuda()
    criterion = nn.BCELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model_path = os.path.join(result_path_gcn, 'checkpoint_gcn', 'checkpoint' + str(fold) + '.pth.tar')
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        pretrained_dict = checkpoint['state_dict']
        model.load_state_dict(pretrained_dict)
        start_epoch = checkpoint['epoch']
        print("============ start epoch:{} ==============".format(start_epoch))
    else:
        print("=> no checkpoint found at '{}'".format(model_path))

    writer = SummaryWriter(result_path_gcn + '/tensorboard_gcn/')

    print("Creat Graphs...")
    patient_id_train, adjacency_train, node_feat_train, patient_y_train, id_bg, adjacency_bg, node_feat_bg, patient_y_bg = get_graph_data(fold, 'train', result_basepath)
    patient_id_val, adjacency_val, node_feat_val, patient_y_val = get_graph_data(fold, 'valid', result_basepath)
    patient_id_zhongs, adjacency_zhongs, node_feat_zhongs, patient_y_zhongs = get_graph_data(fold, 'zhongs', result_basepath)
    patient_id_renmin, adjacency_renmin, node_feat_renmin, patient_y_renmin = get_graph_data(fold, 'renmin', result_basepath)
    patient_id_sixth, adjacency_sixth, node_feat_sixth, patient_y_sixth = get_graph_data(fold, 'sixth', result_basepath)

    train_datasets = graph_dataset(patient_id_train, adjacency_train, node_feat_train, patient_y_train)
    train_datasets_bg = graph_dataset(id_bg, adjacency_bg, node_feat_bg, patient_y_bg)
    valid_datasets = graph_dataset(patient_id_val, adjacency_val, node_feat_val, patient_y_val)
    zhongs_datasets = graph_dataset(patient_id_zhongs, adjacency_zhongs, node_feat_zhongs, patient_y_zhongs)
    renmin_datasets = graph_dataset(patient_id_renmin, adjacency_renmin, node_feat_renmin, patient_y_renmin)
    sixth_datasets = graph_dataset(patient_id_sixth, adjacency_sixth, node_feat_sixth, patient_y_sixth)

    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(dataset=train_datasets, batch_size=batch_size, shuffle=True, **kwargs)
    train_loader_bg = torch.utils.data.DataLoader(dataset=train_datasets_bg, batch_size=batch_size, shuffle=True,
                                                  **kwargs)
    val_loader = torch.utils.data.DataLoader(dataset=valid_datasets, batch_size=batch_size, shuffle=True, **kwargs)

    print("Start Training...")
    best_prec = 0
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=lr_factor, patience=patience, min_lr=0.00001)
    for epoch in range(total_epoch):
        # train for one epoch
        train_losses, train_acc, train_acc_bg = train_gcn(train_loader, train_loader_bg, weight_bg, model, criterion, optimizer, epoch)
        val_losses, val_acc, prec1, val_AUC = valid_gcn(val_loader, model, criterion, optimizer, epoch)

        scheduler.step(val_losses.avg)

        writer.add_scalars('data' + str(fold) + '/loss',
                           {'train_loss': train_losses.avg, 'val_loss': val_losses.avg}, epoch)
        writer.add_scalars('data' + str(fold) + '/Accuracy',
                           {'train_acc': train_acc.avg, 'train_acc_bg': train_acc_bg.avg, 'val_acc': val_acc.avg}, epoch)

        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print('learining rate:', lr)
        writer.add_scalar('learning_rate', lr, epoch)

        is_best = prec1 > best_prec
        if is_best == 1:
            best_prec = max(prec1, best_prec)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec,
        }, is_best, epoch, fold)


    print("Start Testing...")
    save_filename = result_path_gcn + '/result_gcn_{}.csv'.format(fold)
    save_result = open(save_filename, 'a')
    save_result.write('gcn classification result, ,acc,sen,spe,tn,tp,fn,fp,NPV,PPV,AUC' + '\n')

    test_loader_zhongs = torch.utils.data.DataLoader(dataset=zhongs_datasets, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader_renmin = torch.utils.data.DataLoader(dataset=renmin_datasets, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader_sixth = torch.utils.data.DataLoader(dataset=sixth_datasets, batch_size=batch_size, shuffle=True, **kwargs)

    train_names, train_y, train_probs, train_AUC = test(train_loader, model, mode='train')
    train_pred = predict_binary(train_probs, 0.5)
    train_y = list(map(int, train_y))
    train_pred = list(map(int, train_pred))
    save_result_matrix(train_y, train_pred, train_AUC, 'train', 'gcn classify', save_result)
    save_probs_pred(result_path_gcn, train_names, train_y, train_probs, train_pred, mode='train')

    train_names_bg, train_y_bg, train_probs_bg, train_bg_AUC = test(train_loader_bg, model, mode='train_bg')
    train_pred_bg = predict_binary(train_probs_bg, 0.5)
    train_y_bg = list(map(int, train_y_bg))
    train_pred_bg = list(map(int, train_pred_bg))
    save_result_matrix(train_y_bg, train_pred_bg, train_bg_AUC, 'train_bg', 'gcn classify', save_result)
    save_probs_pred(result_path_gcn, train_names_bg, train_y_bg, train_probs_bg, train_pred_bg,
                    mode='train_bg')

    val_names, val_y, val_probs, val_AUC, val_target, val_output = test(val_loader, model, mode='valid')
    val_pred = predict_binary(val_probs, 0.5)
    val_y = list(map(int, val_y))
    val_pred = list(map(int, val_pred))
    save_result_matrix(val_y, val_pred, val_AUC, 'validate', 'gcn classify', save_result)
    save_probs_pred(result_path_gcn, val_names, val_y, val_probs, val_pred, mode='valid')
    np.save(os.path.join(result_path_gcn, 'target_val_fold{}.npy'.format(args.fold_index)), val_target)
    np.save(os.path.join(result_path_gcn, 'pred_val_fold{}.npy'.format(args.fold_index)), val_output)

    test_names_zhongs, test_y_zhongs, test_probs_zhongs, AUC_zhongs = test(test_loader_zhongs, model, mode='zhongs')
    test_pred_zhongs = predict_binary(test_probs_zhongs, 0.5)
    test_y_zhongs = list(map(int, test_y_zhongs))
    test_pred_zhongs = list(map(int, test_pred_zhongs))
    save_result_matrix(test_y_zhongs, test_pred_zhongs, AUC_zhongs, 'zhongs', 'gcn classify', save_result)
    save_probs_pred(result_path_gcn, test_names_zhongs, test_y_zhongs, test_probs_zhongs, test_pred_zhongs, mode='zhongs')

    test_names_renmin, test_y_renmin, test_probs_renmin = test(test_loader_renmin, model, mode='renmin')
    test_pred_renmin = predict_binary(test_probs_renmin, 0.5)
    test_y_renmin = list(map(int, test_y_renmin))
    test_pred_renmin = list(map(int, test_pred_renmin))
    save_result_matrix(test_y_renmin, test_pred_renmin, None, 'renmin', 'gcn classify', save_result)
    save_probs_pred(result_path_gcn, test_names_renmin, test_y_renmin, test_probs_renmin, test_pred_renmin, mode='renmin')

    test_names_sixth, test_y_sixth, test_probs_sixth = test(test_loader_sixth, model, mode='sixth')
    test_pred_sixth = predict_binary(test_probs_sixth, 0.5)
    test_y_sixth = list(map(int, test_y_sixth))
    test_pred_sixth = list(map(int, test_pred_sixth))
    save_result_matrix(test_y_sixth, test_pred_sixth, None, 'sixth', 'gcn classify', save_result)
    save_probs_pred(result_path_gcn, test_names_sixth, test_y_sixth, test_probs_sixth, test_pred_sixth, mode='sixth')

    save_result.close()
