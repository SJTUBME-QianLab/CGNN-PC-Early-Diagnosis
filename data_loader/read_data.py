from torch.utils.data import Dataset
import pandas as pd
import random
import pickle
from tqdm import tqdm
import numpy as np
import nibabel as nib
import os
import json
import torch
import cv2


class Patch_Dataset_orisample(Dataset):
    def __init__(self, config, args, partition, mode):  # image_names=partition['train']
        self.image_names = []
        self.mode = mode
        # save image_name coordinates
        patch_size = config['dataset']['input_dim'][0]
        stride = config['dataset']['stride']
        list_path_renji = config['dataset']['csv_renji']
        df_renji = pd.read_csv(list_path_renji, sep='\t')

        print("{} GET_PATCHES:\tLoading image coordinates from origin dataset".format(mode))
        if mode == 'train' or mode == 'valid':
            self.image_path = "../data_renji/image_npy/"
        else:
            self.image_path = "../data_" + mode + "/image_npy/"

        self.idx = []
        self.image_mat = {}
        image_location = generate_patch_location(df_renji, partition[mode], self.image_path, mode, patch_size, stride, max_amount=1000)
        for case_id, coords in image_location.items():
            for coord in coords:  # coord:[label, x_min, y_min, x_max, y_max, z]
                image_name = [case_id] + coord
                self.image_names.append(image_name)
            self.idx.append([partition['data_type'], case_id, len(coords)])
            self.image_mat[case_id] = np.load(self.image_path + 'box_image/' + case_id + '.npy')


    def __getitem__(self, index):
        image_name = self.image_names[index] #
        case_id = image_name[0]
        patch_label = image_name[1]
        x_min = image_name[2]
        y_min = image_name[3]
        x_max = image_name[4]
        y_max = image_name[5]
        z = image_name[6]
        image = self.image_mat[case_id][x_min:x_max, y_min:y_max, z]
        image = torch.tensor(image).unsqueeze(0)
        if self.mode == 'train' or self.mode == 'valid':
            label = torch.tensor(patch_label)
            return image, label, case_id
        else:
            return image, case_id

    def __len__(self):
        return len(self.image_names)

    def get_locs(self, index):
        image_name = self.image_names[index]
        case_id = image_name[0]
        x_min = image_name[2]
        y_min = image_name[3]
        x_max = image_name[4]
        y_max = image_name[5]
        z = image_name[6]
        return case_id, x_min, y_min, x_max, y_max, z


def read_image_name(config, args, logging):
    # load csv
    list_path_renji = config['dataset']['csv_renji'] # train, validate
    list_path_renmin = config['dataset']['csv_renmin'] # test
    list_path_zhongs = config['dataset']['csv_zhongs']
    list_path_sixth = config['dataset']['csv_sixth']

    df_renji = pd.read_csv(list_path_renji, sep='\t')
    healthy_list_renji = list(df_renji[(df_renji['label1'] == 0)]['filename'])
    tumor_list_renji = list(df_renji[(df_renji['label1'] == 1)]['filename'])
    healthy_partition, num_parts_h_renji = split_fold_partition_self(healthy_list_renji, args.fold_index, args.fold,
                                                               random_seed=args.seed)
    tumor_partition, num_parts_t_renji = split_fold_partition_self(tumor_list_renji, args.fold_index, args.fold,
                                                             random_seed=args.seed)

    df_zhongs = pd.read_csv(list_path_zhongs)
    all_list_zhongs = list(df_zhongs['filename'])
    num_parts_h_zhongs = len(df_zhongs[(df_zhongs['label1'] == 0)]['filename'])
    num_parts_t_zhongs = len(df_zhongs[(df_zhongs['label1'] == 1)]['filename'])

    df_renmin = pd.read_csv(list_path_renmin)
    all_list_renmin = list(df_renmin['filename'])
    num_parts_t_renmin = len(df_renmin[(df_renmin['label1'] == 1)]['filename'])

    df_sixth = pd.read_csv(list_path_sixth)
    all_list_sixth = list(df_sixth['filename'])
    num_parts_t_sixth = len(df_sixth[(df_sixth['label1'] == 1)]['filename'])

    partition = {}
    partition['fold'] = args.fold_index
    partition['data_type'] = config['dataset']['data_type']  # ruijin_renji
    partition['all'] = healthy_partition['all'] + tumor_partition['all']
    partition['train'] = healthy_partition['train'] + tumor_partition['train']
    partition['valid'] = healthy_partition['valid'] + tumor_partition['valid']
    partition['zhongs'] = all_list_zhongs
    partition['renmin'] = all_list_renmin
    partition['sixth'] = all_list_sixth
    # save image_name
    logging.info(partition)

    return partition, num_parts_h_renji, num_parts_t_renji, num_parts_h_zhongs, num_parts_t_zhongs, num_parts_t_renmin, num_parts_t_sixth # [[xx …… xx],……] [xx,xx,xx] [xx,xx,xx]


def split_fold_partition_self(case_list, fold_index, fold=5, path=None, random_seed=None):
    print('SPLIT_CASE_PARTITION:\tStart spliting cases...')

    split_number = round(len(case_list) / fold)
    random.Random(random_seed).shuffle(case_list)

    folds = []
    for i in range(fold):
        folds_fold = []
        if i != fold - 1:
            folds_fold.extend(case_list[i * split_number: (i + 1) * split_number])
            folds.append(folds_fold)
        else:
            folds_fold.extend(case_list[i * split_number:])
            folds.append(folds_fold)

    partition = {}
    partition['all'] = case_list
    partition['train'] = []
    for idx in range(fold):
        if idx == fold_index:
            partition['valid'] = folds[idx]
        else:
            partition['train'].extend(folds[idx])

    # report actual partition ratio
    num_parts = list(map(len, [partition[part] for part in ['train', 'valid']]))
    print('fold {}\tSPLIT_CASE_PARTITION:\tActual Partition Number: (train, val)={}'.format(fold_index, (num_parts)))

    # saving partition dict to disk
    if path is not None:
        print('SPLIT_SAVE_CASE_PARTITION:\tStart saving partition dict to {}'.format(path))
        with open(path, 'wb') as f:
            pickle.dump(partition, f, pickle.HIGHEST_PROTOCOL)
        print('SPLIT_SAVE_CASE_PARTITION:\tDone saving')

    return partition, num_parts


# data_preprocess
def save_box_to_csv(list_path, data_path, save_path, data_type):

    save_filename = data_type + '_all.csv'
    save_result = open(save_filename, 'w')
    save_result.write('case_id,type, xmin, ymin, zmin, xmax, ymax, zmax' + '\n')

    if data_type=='ruijin':
        df = pd.read_excel(list_path, converters={'add_date': str}, engine='openpyxl')
        healthy_list = list(df[(df['type'] == 'healthy')]['case_id'])
        tumor_list = list(df[(df['type'] == 'tumor')]['case_id'])
    elif data_type=='renji':
        df = pd.read_csv(list_path, sep='\t')
        healthy_list = list(df[(df['label1'] == 0)]['filename'])
        tumor_list = list(df[(df['label1'] == 1)]['filename'])
    all_list = healthy_list + tumor_list

    for file in all_list:
        filename, box_location = save_nii_to_npy(file, data_path, save_path, data_type)
        box_location = [str(i) for i in box_location]
        save_result.write(filename+','+ ','.join(box_location) + '\n')
        print('finish process:', filename)


def generate_patch_location(df_renji, image_list, save_path, mode, patch_size=50, stride=25, max_amount=1000):

    threshold = 1 / (patch_size ** 2)
    image_location = {}

    bar = tqdm(image_list, ncols=130)
    for file in bar:
        bar.set_description("Generating {} patch location: ".format(mode))
        pancreas = np.load(os.path.join(save_path, 'box_pancreas', file + '.npy'))
        lesion = np.zeros(pancreas.shape)

        coords = []
        for z in range(pancreas.shape[2]):
            for row in range((pancreas.shape[0] - patch_size) // stride + 1):
                for col in range((pancreas.shape[1] - patch_size) // stride + 1):
                    x_min = row * stride
                    y_min = col * stride

                    x_max = x_min + patch_size
                    if x_max > pancreas.shape[0]:
                        x_max = pancreas.shape[0]
                        x_min = x_max - patch_size
                    y_max = y_min + patch_size
                    if y_max > pancreas.shape[1]:
                        y_max = pancreas.shape[1]
                        y_min = y_max - patch_size

                    patch_lesion = lesion[x_min:x_max, y_min:y_max, z]
                    patch_pancreas = pancreas[x_min:x_max, y_min:y_max, z]
                    if np.sum(patch_lesion) / (patch_size ** 2) > threshold:
                        value = 1
                        coords.append([value, x_min, y_min, x_max, y_max, z])
                    elif np.sum(patch_pancreas) / (patch_size ** 2) > threshold:
                        value = 0
                        coords.append([value, x_min, y_min, x_max, y_max, z])

        while len(coords) > max_amount:
            coords = coords[::2]

        image_location[file] = coords
        # print('finish process:', file)

    # save image_location
    json_file = "3hos_renji_image_patch_location.json"
    with open(json_file, 'w') as f:
        json.dump(image_location, f)

    return image_location


def save_nii_to_npy(filename, data_path, save_path, data_type, border=(10, 10, 3)):
    imagepath = os.path.join(data_path, 'image', filename + '.nii.gz')
    labelpath_p = os.path.join(data_path, 'label', 'pancreas', filename + '.nii.gz')
    img = nib.load(imagepath)
    image_array = img.get_fdata()
    image = window_transform(image_array, windowWidth=340, windowCenter=70, normal=True)
    label_p = nib.load(labelpath_p).get_fdata()
    image = np.transpose(np.flip(image, axis=2), (1, 0, 2))
    label_p = np.transpose(np.flip(label_p, axis=2), (1, 0, 2))
    label_t = np.zeros(label_p.shape)
    if data_type == 'renji' and filename[:2] == 'PC':
        labelpath_t = os.path.join(data_path, 'label', 'tumor', filename + '.nii.gz')
        label_t = nib.load(labelpath_t).get_fdata()
        label_t = np.transpose(np.flip(label_t, axis=2), (1, 0, 2))

    pancreas = np.zeros(label_p.shape)
    pancreas[np.where(label_p != 0)] = 1
    lesion = np.zeros(label_p.shape)
    if data_type == 'renji':
        lesion[np.where(label_t != 0)] = 1

    # Generate box index
    xmin, ymin, zmin = np.max(
        [np.min(np.where(pancreas != 0), 1) - border, (0, 0, 0)], 0)
    xmax, ymax, zmax = np.min(
        [np.max(np.where(pancreas != 0), 1) + border, label_p.shape], 0)

    # Generate box data
    box_image = image[xmin:xmax, ymin:ymax, zmin:zmax]
    box_pancreas = pancreas[xmin:xmax, ymin:ymax, zmin:zmax]
    box_lesion = lesion[xmin:xmax, ymin:ymax, zmin:zmax]
    box_location = [xmin, ymin, zmin, xmax, ymax, zmax]
    print(xmin, xmax, ymin, ymax, zmin, zmax)
    np.save(os.path.join(save_path, 'box_image/', filename + '.npy'), box_image)
    np.save(os.path.join(save_path, 'box_pancreas/', filename + '.npy'), box_pancreas)
    if box_lesion.sum() > 0:
        np.save(os.path.join(save_path, 'box_lesion/', filename + '.npy'), box_lesion)

    return filename, box_location


def window_transform(ct_array, windowWidth=340, windowCenter=70, normal=False):
    """
    return: trucated image according to window center and window width
    and normalized to [0,1]
    """
    minWindow = float(windowCenter) - 0.5 * float(windowWidth)
    newimg = (ct_array - minWindow) / float(windowWidth)
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    if not normal:
        newimg = (newimg * 255).astype('uint8')
    # else:
    #     newimg = newimg.astype('uint8')
    return newimg


if __name__ == '__main__':
    for hospital in ['renji', 'zhongs', 'renmin', 'sixth']:
        data_path = '../../data_' + hospital + '/image_nii/'
        save_path = '../../data_' + hospital + '/image_npy/'

        list_path = '../configs/' + hospital + '_test.xlsx'

        if not os.path.isdir(save_path):
            os.mkdir(save_path)
            os.mkdir(os.path.join(save_path, 'box_image/'))
            os.mkdir(os.path.join(save_path, 'box_pancreas/'))
            os.mkdir(os.path.join(save_path, 'box_lesion/'))
        save_box_to_csv(list_path, data_path, save_path, data_type=hospital)
