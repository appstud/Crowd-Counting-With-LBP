
import glob
import os
import random
import numpy as np


def get_file_id(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]
    
def get_data_list(data_root, mode='train'):

    """
    Returns a list of images that are to be used during training, validation and testing.
    It looks into various folders depending on the mode and prepares the list.
    :param mode: selection of appropriate mode from train, validation and test.
    :return: a list of filenames of images and corresponding ground truths after random shuffling.
    """

    if mode == 'train':
        imagepath = os.path.join(data_root, 'train_data', 'images')
        gtpath = os.path.join(data_root, 'train_data', 'ground-truth')

    elif mode == 'valid':
        imagepath = os.path.join(data_root, 'valid_data', 'images')
        gtpath = os.path.join(data_root, 'valid_data', 'ground-truth')

    else:
        imagepath = os.path.join(data_root, 'test_data', 'images')
        gtpath = os.path.join(data_root, 'test_data', 'ground-truth')
    
    
    image_list = [file for file in glob.glob(os.path.join(imagepath,'*.jpg'))]
    gt_list = []

    for filepath in image_list:
        file_id = get_file_id(filepath)
        gt_file_path = os.path.join(gtpath, 'GT_'+ file_id + '.mat')
        gt_list.append(gt_file_path)
    xy = list(zip(image_list, gt_list))
    random.shuffle(xy)
    s_image_list, s_gt_list = zip(*xy)
   
    return s_image_list, s_gt_list

