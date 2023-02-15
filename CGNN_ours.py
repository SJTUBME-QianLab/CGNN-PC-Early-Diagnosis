
import numpy as np
from torch.utils.data import Dataset
import torch
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
from graph_construct import *


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



def train_gcn(train_loader, train_loader_bg, weight_bg, model, criterion, optimizer, epoch, use_cuda=True):
    """Train for one epoch on the training set"""
    train_losses = AverageMeter()
    train_acc = AverageMeter()
    train_acc_bg = AverageMeter()
    target_roc = torch.zeros((0, args.num_classes))
    target_roc_bg = torch.zeros((0, args.num_classes))
    # switch to train mode
    model.train()

    with tqdm(zip(train_loader, cycle(train_loader_bg)), ncols=130) as t:
        for i, ([adjacency, feature, target, _], [adjacency_bg, feature_bg, target_bg, _]) in enumerate(t):
            t.set_description("train epoch %s" % epoch)
            if use_cuda:
                target = target.type(torch.FloatTensor).cuda()
                adjacency = adjacency.type(torch.FloatTensor).cuda()
                feature = feature.type(torch.FloatTensor).cuda()
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


def valid_gcn(val_loader, model, criterion, epoch, use_cuda=True):
    """valid for one epoch on the validation set"""
    val_losses = AverageMeter()
    val_acc = AverageMeter()
    # switch to valid mode
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

            output = model(feature, adjacency)
            val_loss = criterion(output.view(output.shape[0]), target)
            val_losses.update(val_loss.item(), adjacency.size(0))

            target_roc = torch.cat((target_roc, target.to(torch.float32).unsqueeze(1).data.cpu()), dim=0)
            output_roc = torch.cat((output_roc, output.data.cpu()), dim=0)

            acc = accuracy(output.data, target)
            val_acc.update(acc, adjacency.size(0))

            t.set_postfix({
                'loss': '{loss.val:.4f}({loss.avg:.4f})'.format(loss=val_losses),
                'Acc': '{acc.val:.4f}({acc.avg:.4f})'.format(acc=val_acc)}
            )
        AUROC = aucrocs(output_roc, target_roc)
        print('The AUROC is %.4f' % AUROC)

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
    else:
        return names, y_patient, probs


def aucrocs(output, target):
    """
    Returns:
    List of AUROCs of all classes.
    """
    output_np = output.cpu().numpy()
    target_np = target.cpu().numpy()
    AUROCs=roc_auc_score(target_np[:, 0], output_np[:, 0])
    return AUROCs


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
    model = GCN_noGaussian()
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


    train_datasets = graph_dataset(patient_id_train, adjacency_train, node_feat_train, patient_y_train)
    train_datasets_bg = graph_dataset(id_bg, adjacency_bg, node_feat_bg, patient_y_bg)
    valid_datasets = graph_dataset(patient_id_val, adjacency_val, node_feat_val, patient_y_val)


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
        val_losses, val_acc, prec1, val_AUC = valid_gcn(val_loader, model, criterion, epoch)

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

    save_result.close()
