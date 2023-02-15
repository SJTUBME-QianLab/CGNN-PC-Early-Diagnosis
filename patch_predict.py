# coding=utf-8

import sys
import pandas as pd
sys.path.append("..")
import matplotlib as mpl
mpl.use('Agg')
from tqdm import tqdm
import argparse
from time import gmtime, strftime, localtime
from torch.utils.data import DataLoader
from utils import (load_config, predict_binary)
from data_loader.read_data import Patch_Dataset_orisample, read_image_name
import logging
from tensorboardX import SummaryWriter
from models.net_pytorch import *
from sklearn.metrics import roc_auc_score
import shutil
from sklearn.metrics import confusion_matrix


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(description='PyTorch pancreas early diagnosis')
parser.add_argument('--model', default='VGG_liuyedao_ori', type=str,
                    help='baseline of the model')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--num_classes', default=2, type=int,  # num_classes
                    help='numbers of classes (default: 1)')
parser.add_argument('--optimizer', default='Adam', type=str,  # Adam
                    help='optimizer (Adam)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--weight-decay-fc', '--wdfc', default=0, type=float,
                    help='weight decay fc (default: 1e-4)')
parser.add_argument('--tensorboard', default=True,
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--use_cuda', default=True,
                    help='whether to use_cuda(default: True)')
parser.add_argument('--batch_size_train', default=4000, type=int,
                    help='mini-batch size (default: 4000)')
parser.add_argument('--batch_size_val', default=2000, type=int,
                    help='mini-batch size (default: 2000)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,  # 0.01pyt
                    help='initial learning rate')
parser.add_argument("--config", default='./configs/3hos_renji.yml',
                    type=str, help="train configuration")
parser.add_argument("--run_name", default=None, type=str,
                    help="run name for this experiment. (Default: time)")
parser.add_argument('--fold', default=5, type=int, help='index of k-fold')
parser.add_argument('--fold_index', default=0, type=int, help='index of k-fold')
parser.add_argument('--n_epoch', default=10, type=int, help='number of epoch to change') #
parser.add_argument('--epoch', default=10, type=int, help='number of total epochs to run')
parser.add_argument('--seed', default=2, type=int,help='random seed(default: 1)')  #
parser.add_argument("--save_name", default='_renji_epoch30_50_lr0.001_seed2', type=str,
                    help="the fold of the experiment")
parser.add_argument("--gpu", type=str, default="0", metavar="N", help="input visible devices for training (default: 0)")
parser.add_argument("--image_path_renji", default="../data_renji/image_npy/", type=str)


def main():
    global use_cuda, writer
    # args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.tensorboard:
        writer = SummaryWriter(result_basepath+'/tensorboard/')
    use_cuda = args.use_cuda and torch.cuda.is_available()

    if args.seed >= 0:
        seed_torch(args.seed)

    model = VGG_liuyedao_ori_cnn_CEL()
    if use_cuda:
        model = model.cuda()

    model_path = os.path.join(result_basepath, 'checkpoint', 'checkpoint' + str(args.fold_index) + '.pth.tar')
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        pretrained_dict = checkpoint['state_dict']
        model.load_state_dict(pretrained_dict)
        args.start_epoch = checkpoint['epoch']
        print("============ start epoch:{} ==============".format(args.start_epoch))
    else:
        print("=> no checkpoint found at '{}'".format(model_path))

    weights = torch.FloatTensor([1.0, 2.0]).cuda()
    criterion = nn.CrossEntropyLoss(weight=weights)

    # define optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    nesterov=True, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.99))
    else:
        print('Please choose true optimizer.')
        return 0

    # Get train patches
    partition, num_part_h_renji, num_part_t_renji = read_image_name(config, args, logging)
    train_datasets = Patch_Dataset_orisample(config, args, partition, mode='train')
    val_datasets = Patch_Dataset_orisample(config, args, partition, mode='valid')

    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(dataset=train_datasets, batch_size=args.batch_size_train, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(dataset=val_datasets, batch_size=args.batch_size_val, shuffle=True, **kwargs)

    best_prec = 0
    for epoch in range(args.start_epoch, args.epoch):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_losses, train_acc = train(train_loader, model, criterion, optimizer, epoch, args.fold_index)
        image_names, val_losses, val_acc, prec1 = validate(val_loader, model, criterion, epoch, args.fold_index)

        if args.tensorboard:
            writer.add_scalars('data' + str(args.fold_index) + '/loss',
                               {'train_loss': train_losses.avg, 'val_loss': val_losses.avg}, epoch)
            writer.add_scalars('data' + str(args.fold_index) + '/Accuracy',
                               {'train_acc': train_acc.avg, 'val_acc': val_acc.avg}, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec
        print('prec:', prec1)
        print('best_prec:', best_prec)
        print('is best:', is_best)
        if is_best == 1:
            best_prec = max(prec1, best_prec)  #
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec,
        }, is_best, epoch, args.fold_index)

    # -------------
    #     test
    # -------------
    # load best model
    # model_path = os.path.join(result_basepath, 'checkpoint', 'model_best' + str(args.fold_index) + '.pth.tar')
    # checkpoint = torch.load(model_path)
    # pretrained_dict = checkpoint['state_dict']
    # model.load_state_dict(pretrained_dict)

    save_filename = result_basepath + '/result_{}.csv'.format(args.fold_index)
    save_result = open(save_filename, 'a')
    save_result.write(args.save_name + '\n')
    save_result.write('healthy people renji' + ',' + str(num_part_h_renji) + '\n')
    save_result.write('patient renji' + ',' + str(num_part_t_renji) + '\n')
    save_result.write("fold_index" + ',' + str(args.fold_index) + '\n')
    save_result.write("seed" + ',' + str(args.seed) + '\n')
    save_result.write("epoch" + ',' + str(args.epoch) + '\n')
    save_result.write('optimizer' + ',' + str(args.optimizer) + '\n')
    save_result.write('lr' + ',' + str(args.lr) + '\n')
    save_result.write('ori sample before threshold patch, ,acc,sen,spe,tn,tp,fn,fp,NPV,PPV'+'\n')

    train_datasets.trainmode = False
    val_datasets.trainmode = False
    test_datasets_zhongs.trainmode = False
    test_datasets_renmin.trainmode = False
    test_datasets_sixth.trainmode = False

    train_loader = torch.utils.data.DataLoader(dataset=train_datasets, batch_size=args.batch_size_train, shuffle=False, **kwargs)
    val_loader = torch.utils.data.DataLoader(dataset=val_datasets, batch_size=args.batch_size_val, shuffle=False, **kwargs)
    test_loader_zhongs = torch.utils.data.DataLoader(dataset=test_datasets_zhongs, batch_size=args.batch_size_val, shuffle=False, **kwargs)
    test_loader_renmin = torch.utils.data.DataLoader(dataset=test_datasets_renmin, batch_size=args.batch_size_val, shuffle=False, **kwargs)
    test_loader_sixth = torch.utils.data.DataLoader(dataset=test_datasets_sixth, batch_size=args.batch_size_val, shuffle=False, **kwargs)

    train_names, train_y_patch, train_probs_patch, train_feat_patch = test_renji(train_loader, model, mode='train')
    train_CNN_pred_patch = predict_binary(train_probs_patch)
    save_result_matrix(train_y_patch, train_CNN_pred_patch, 'train', 'ori sample before threshold', save_result)

    vaild_names, valid_y_patch, valid_probs_patch, valid_feat_patch = test_renji(val_loader, model, mode='valid')
    valid_CNN_pred_patch = predict_binary(valid_probs_patch)
    save_result_matrix(valid_y_patch, valid_CNN_pred_patch, 'valid', 'ori sample before threshold', save_result)

    patch_loc_train = []
    patch_loc_val = []
    patch_loc_test_zhongs = []
    patch_loc_test_renmin = []
    patch_loc_test_sixth = []
    for i in range(len(train_datasets)):
        patch_loc_train_i = train_datasets.get_locs(i)  # case_id, x_min, y_min, x_max, y_max, z
        patch_loc_train.append(patch_loc_train_i)
    for i in range(len(val_datasets)):
        patch_loc_val_i = val_datasets.get_locs(i)  # case_id, x_min, y_min, x_max, y_max, z
        patch_loc_val.append(patch_loc_val_i)
    for i in range(len(test_datasets_zhongs)):
        patch_loc_test_i = test_datasets_zhongs.get_locs(i)  # case_id, x_min, y_min, x_max, y_max, z
        patch_loc_test_zhongs.append(patch_loc_test_i)
    for i in range(len(test_datasets_renmin)):
        patch_loc_test_i = test_datasets_renmin.get_locs(i)  # case_id, x_min, y_min, x_max, y_max, z
        patch_loc_test_renmin.append(patch_loc_test_i)
    for i in range(len(test_datasets_sixth)):
        patch_loc_test_i = test_datasets_sixth.get_locs(i)  # case_id, x_min, y_min, x_max, y_max, z
        patch_loc_test_sixth.append(patch_loc_test_i)

    # save patch probs pred
    save_patch_probs_pred(train_names, train_y_patch, train_CNN_pred_patch, train_probs_patch, patch_loc_train, train_feat_patch, mode='train' )
    save_patch_probs_pred(vaild_names, valid_y_patch, valid_CNN_pred_patch, valid_probs_patch, patch_loc_val, valid_feat_patch, mode='valid')

    save_result.close()


def train(train_loader, model, criterion, optimizer, epoch, fold):
    """Train for one epoch on the training set"""
    train_losses = AverageMeter()
    train_acc = AverageMeter()
    # switch to train mode
    model.train()

    with tqdm(train_loader, ncols=130) as t:
        for i, (input, target, case_id) in enumerate(t):
            t.set_description("train epoch %s" % epoch)
            if use_cuda:
                target = target.type(torch.FloatTensor).cuda()
                input = input.type(torch.FloatTensor).cuda()
            optimizer.zero_grad()
            output = model(input.to(torch.float32))

            train_loss = criterion(output.squeeze(), target.to(torch.float32).long())
            train_losses.update(train_loss.item(), input.size(0))

            acc = accuracy(output.data, target)
            train_acc.update(acc, input.size(0))

            # compute gradient
            train_loss.backward()
            optimizer.step()

            t.set_postfix({
                'loss': '{loss.val:.4f}({loss.avg:.4f})'.format(loss=train_losses),
                'Acc': '{acc.val:.4f}({acc.avg:.4f})'.format(acc=train_acc)}
            )

    # log to TensorBoard
    if args.tensorboard:
        writer.add_scalar('data' + str(fold) + '/train_loss', train_losses.avg, epoch)
        writer.add_scalar('data' + str(fold) + '/train_acc', train_acc.avg, epoch)
    return train_losses, train_acc


def validate(val_loader, model, criterion, epoch, fold):
    """Perform validation on the validation set"""
    val_losses = AverageMeter()
    val_acc = AverageMeter()
    # switch to evaluate mode
    model.eval()
    image_names = []

    with torch.no_grad():
        with tqdm(val_loader, ncols=130) as t:
            for i, (input, target, case_id) in enumerate(t):
                t.set_description("valid epoch %s" % epoch)

                if use_cuda:
                    target = target.type(torch.FloatTensor).cuda()
                    input = input.type(torch.FloatTensor).cuda()

                # compute output
                output = model(input.to(torch.float32))
                val_loss = criterion(output.squeeze(), target.to(torch.float32).long())
                val_losses.update(val_loss.item(), input.size(0))

                # -------------------------------------Accuracy--------------------------------- #
                acc = accuracy(output.data, target)
                val_acc.update(acc, input.size(0))

                t.set_postfix({
                    'loss': '{loss.val:.4f}({loss.avg:.4f})'.format(loss=val_losses),
                    'Acc': '{acc.val:.4f}({acc.avg:.4f})'.format(acc=val_acc)}
                )

    if args.tensorboard:
        writer.add_scalar('data' + str(fold) + '/val_loss', val_losses.avg, epoch)
        writer.add_scalar('data' + str(fold) + '/val_acc', val_acc.avg, epoch)
    return image_names, val_losses, val_acc, val_acc.avg


def test(test_loader, model, mode):
    probs = []
    model.eval()
    names = []
    features = []
    with torch.no_grad():
        with tqdm(test_loader, ncols=130) as t:
            for i, (input, case_id) in enumerate(t):
                t.set_description("Calculate {} probs:".format(mode))
                if use_cuda:
                    input = input.type(torch.FloatTensor).cuda()

                names.extend(case_id)
                # compute output
                output = model(input)
                output = output.detach().cpu().numpy()
                feature = model.feature(input)
                feature = feature.detach().cpu().numpy()
                for i in range(feature.shape[0]):
                    if feature[i].shape[0] != 32:
                        print('feat shape:', feature[i].shape)
                        record_path = os.path.join(result_basepath, 'record.txt')
                        record = open(record_path, mode='a')
                        record.write(str(feature[i].shape[0]) + ',' + ' '.join(str(feature[i][j]) for j in range(feature[i].shape[0])) + '\n')
                        record.close()
                    features.append(list(feature[i]))  # 返回特征
                for bs in range(output.shape[0]):
                    probs.append(output[bs])

    probs = np.array(probs)
    return names, probs, features


def test_renji(test_loader, model, mode):
    probs = []
    model.eval()
    names = []
    y_patient = []
    features = []
    with torch.no_grad():
        with tqdm(test_loader, ncols=130) as t:
            for i, (input, target, case_id) in enumerate(t):
                t.set_description("Calculate {} probs:".format(mode))
                if use_cuda:
                    input = input.type(torch.FloatTensor).cuda()

                names.extend(case_id)
                # compute output
                output = model(input)
                output = output.detach().cpu().numpy()
                feature = model.feature(input)
                feature = feature.detach().cpu().numpy()
                for i in range(feature.shape[0]):
                    if feature[i].shape[0] != 32:
                        print('feat shape:', feature[i].shape)
                        record_path = os.path.join(result_basepath, 'record.txt')
                        record = open(record_path, mode='a')
                        record.write(str(feature[i].shape[0]) + ',' + ' '.join(str(feature[i][j]) for j in range(feature[i].shape[0])) + '\n')
                        record.close()
                    features.append(list(feature[i]))
                for bs in range(output.shape[0]):
                    probs.append(output[bs])
                    y_patient.append(target.numpy()[bs])

    probs = np.array(probs)
    return names, y_patient, probs, features


def save_result_matrix(true,pred,mode,time,save_result):
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
                      +','+str(tn)+','+str(tp)+','+str(fn)+','+str(fp)+','+str(NPV)+','+str(PPV)+'\n')


def save_patch_probs_pred(names, y, pred, probs, patch_loc, patch_feat, mode):
    patch_prob_filename = os.path.join(result_basepath, '{}_test_probs_{}.csv'.format(args.fold_index, mode))
    feat_head = ['feat'+str(x) for x in range(32)]
    info = []
    for i in range(len(names)):
        info.append([names[i], y[i], int(pred[i]), probs[i][1]] + list(patch_loc[i]) + patch_feat[i])

    df = pd.DataFrame(info, columns=['id', 'label', 'pred', 'prob', 'id_loc', 'x_min', 'y_min', 'x_max', 'y_max', 'z']+feat_head)
    df.to_csv(patch_prob_filename, index=False)


def save_patch_probs_pred_test(names, pred, probs, patch_loc, patch_feat, mode):
    patch_prob_filename = os.path.join(result_basepath, '{}_test_probs_{}.csv'.format(args.fold_index, mode))
    feat_head = ['feat'+str(x) for x in range(32)]
    info = []
    for i in range(len(names)):
        info.append([names[i], int(pred[i]), probs[i][1]] + list(patch_loc[i]) + patch_feat[i])

    df = pd.DataFrame(info, columns=['id', 'pred', 'prob', 'id_loc', 'x_min', 'y_min', 'x_max', 'y_max', 'z']+feat_head)
    df.to_csv(patch_prob_filename, index=False)


def save_checkpoint(state, is_best, epoch, fold):
    """Saves checkpoint to disk"""
    # filename = 'checkpoint' + str(fold) + '_' + str(epoch) + '.pth.tar'
    filename = 'checkpoint' + str(fold) + '.pth.tar'
    directory = result_basepath + "/checkpoint/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        torch.save(state, filename)
        shutil.copyfile(filename, result_basepath + '/checkpoint/' + 'model_best' + str(fold) + '.pth.tar')


def aucrocs(output, target):

    """
    Returns:
    List of AUROCs of all classes.
    """
    output_np = output.cpu().numpy()
    # print('output_np:',output_np)
    target_np = target.cpu().numpy()
    # print('target_np:',target_np)
    AUROCs = roc_auc_score(target_np, output_np, average='macro', multi_class='ovo')
    return AUROCs


def predict_binary(prob):
    pred = np.argmax(prob, axis=1)

    return pred

def accuracy(output, target):
    output_np = output.cpu().numpy()
    # output_np[output_np > 0.5] = 1
    # output_np[output_np <= 0.5] = 0
    pred = np.zeros(output_np.shape[0])
    pred[output_np[:,0]<output_np[:,1]] = 1

    target_np = target.cpu().numpy()

    right = (pred == target_np)
    acc = np.sum(right) / output.shape[0]

    return acc


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    if epoch < args.n_epoch:
        lr = args.lr
    else:
        lr = args.lr * 0.1
    # log to TensorBoard
    if args.tensorboard:
        writer.add_scalar('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    args = parser.parse_args()
    config = load_config(args.config)
    if args.run_name is None:
        config['run_name'] = strftime("%Y%m%d_%H%M%S", localtime())
    else:
        config['run_name'] = args.run_name

    print('epoch is {}'.format(args.epoch))

    result_basepath = os.path.join(config['log']['result_dir'], config['run_name']+args.save_name)
    if not os.path.isdir(result_basepath):
        os.mkdir(result_basepath)


    logging_filename = os.path.join(result_basepath, 'log_file.log')
    logging.basicConfig(filename=logging_filename,
                        level=logging.INFO,
                        datefmt="%Y-%m-%d %H:%M:%S",
                        format='%(levelname)-8s: %(asctime)-12s: %(message)s')
    logging.info(config)
    logging.info(args)

    main()