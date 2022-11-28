import collections
import hashlib
import ruamel.yaml
# from keras import backend as K
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix
import random
import os
# import tensorflow as tf


def get_config_sha1(config, digit=5):
    """Get the sha1 of configuration for Experiment ID

    config will be converted to str and sha.

    Args:
        config (dict): The dictionary contains configuration information.
        digit (int, optional): The number of starting digit. Defaults to 5.

    Returns:
        str: First "digit" of config's sha1

    """
    s = hashlib.sha1()
    s.update(str(config).encode('utf-8'))
    return s.hexdigest()[:digit]


def count_parameters(model):
    """Get the number of trainable params

    Parameters is trainable iff it requires gradient.

    Args:
        model (pytorch model): The pytorch model.

    Returns:
        int: number of trainable parameters

    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_config(path):
    """load YAML config

    Args:
        path: path to config.

    Returns:
        config: dict

    """

    with open(path, 'r', encoding='utf-8') as f:
        config = ruamel.yaml.safe_load(f)
    config['config_sha1'] = get_config_sha1(config, 5)

    return config


def predict_binary(prob, threshold):
    prob = np.array(prob)
    binary = np.zeros(prob.shape)
    binary[prob < threshold] = 0
    binary[prob >= threshold] = 1
    return binary


def predict_binary_list(prob, threshold):
    binary = np.zeros(len(prob))
    binary[np.array(prob) < threshold] = 0
    binary[np.array(prob) >= threshold] = 1
    return binary


def Youden(tp,tn,fn,fp):
    sen = tp/(tp+fn)
    spe = tn/(tn+fp)
    return (sen + spe - 1)


def find_threshold(predict_probs, groundtrue):

    fpr, tpr, thresholds = roc_curve(groundtrue, predict_probs)
    return thresholds[np.argmax(1 - fpr + tpr)]


def save_result_matrix(true,pred,mode,time,save_result,fold_index):
    tn,fp,fn,tp = confusion_matrix(true,pred).ravel()
    acc = (tn+tp)/(tn+fp+fn+tp)
    sen = tp/(tp+fn)
    spe = tn/(tn+fp)
    NPV = tn/(tn+fn)
    PPV = tp/(tp+fp)
    print('{}: {} tn:{},fp:{},fn:{},tp:{}'.format(mode,time,tn,fp,fn,tp))
    print('{}: {} acc is {}'.format(mode,time,acc))
    print('{}: {} sen is {}, spe is {}'.format(mode,time,sen,spe))
    save_result.write(str(fold_index)+','+mode+','+str(acc)+','+str(sen)+','+str(spe)
                      +','+str(tn)+','+str(tp)+','+str(fn)+','+str(fp)+','+str(NPV)+','+str(PPV)+'\n')


def patient_matrix_get(data_idx, data_pred):
    pre_idx = 0
    cur_idx = 0
    patient_y, patient_probs = [], []
    for patient in data_idx:
        cur_idx = cur_idx + patient[2]
        pred_y = data_pred[pre_idx:cur_idx]
        pre_idx = cur_idx

        if patient[0] == 'msd' or patient[1][:1] == 'C' or patient[1][:2] == 'PC':
            patient_y.append(1)
        else:
            patient_y.append(0)
        patient_probs.append(sum(pred_y) / patient[2])

    return patient_y, patient_probs
