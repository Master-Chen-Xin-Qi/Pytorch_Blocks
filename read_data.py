#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : utils.py
@Date         : 2022/04/15 16:39:20
@Author       : Xinqi Chen 
@Software     : VScode
@Description  : Some utils functions for the project
'''

import os
import torch
import torch.nn as nn
import numpy as np
import config
import json
import math

# set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    
# 将名字转为label
def label_process(different_name):
    label_dict = dict()
    for i in range(len(different_name)):
        label_dict[different_name[i]] = i
    item = json.dumps(label_dict)
    path = './name_to_label.txt'
    if os.path.exists(path):
        return label_dict
    else:
        with open(path, "a", encoding='utf-8') as f:
            f.write(item + ",\n")
            print("^_^ write success")
    return label_dict

# 整个流程汇总
def prepare_data(folder_name, save_path):
    label_dict = label_process(config.app_name)
    df = divide_files_by_name(folder_name, config.app_name)
    all_data, all_label = process_file_data(df, label_dict, save_path)
    np.save('./data_np/mag_np/all_data.npy', all_data)
    np.save('./label_np/label_np/all_label.npy', all_label)
    print('ALL data have been saved!')
    
# Read files of different categories and divide into several parts
def divide_files_by_name(folder_name, different_category):
    dict_file = dict(zip(different_category, [[]] * len(different_category)))  # initial
    for category in different_category:
        dict_file[category] = []  # Essential here
        for (root, _, files) in os.walk(folder_name):  # List all file names
            for filename in files:  # Attention here
                file_ = os.path.join(root, filename)
                if category in filename:  
                    dict_file[category].append(file_)
    return dict_file


def process_file_data(df, label_dict, save_path):
    all_data = np.zeros((1, config.WINDOW_LEN))
    all_label = np.zeros((1, 1))
    for app in df.keys():
        label = label_dict[app]
        mag_data_total = np.zeros((1, config.WINDOW_LEN))
        label_total = np.zeros((1, 1))
        for single_file in df[app]:
            mag_data, total_num = read_single_file_data(single_file)
            mag_data_total = np.vstack((mag_data_total, mag_data))
            mag_label = np.array([label]*total_num).reshape(-1, 1)
            label_total = np.vstack((label_total, mag_label))
        mag_data_total = mag_data_total[1:]
        mag_data_total = mag_data_total.astype('float32')
        label_total = label_total[1:]
        np.save(save_path+'/mag_np/'+f'{app}_'+str(config.WINDOW_LEN), mag_data_total)
        np.save(save_path + '/label_np/'+ f'{app}_'+str(config.WINDOW_LEN), label_total)
        all_data = np.vstack((all_data, mag_data_total))
        all_label = np.vstack((all_label, label_total))
    all_data = all_data[1:]
    all_label = all_label[1:]
    return all_data, all_label
    
    
    
# 读取每一个文件中的数据
def read_single_file_data(single_file_name):
    mag_data = []
    fid = open(single_file_name, 'r')
    for line in fid:
        if 'loggingTime' in line or ',,,,,' in line:  # Wrong data type
            continue
        line = line.strip('\n')
        data = line.split(',')
        data_x, data_y, data_z = float(data[-4]), float(data[-3]), float(data[-2])
        data_tmp = math.sqrt(data_x**2 + data_y**2 + data_z**2)
        mag_data.append(data_tmp)
    slide_data = slide_window(mag_data)
    total_num = slide_data.shape[0]
    return slide_data, total_num

# 用滑动窗口来处理数据
def slide_window(mag_data):
    mag_l = len(mag_data)
    start = 0
    emp = np.zeros((1, config.WINDOW_LEN))
    while start < mag_l - config.WINDOW_LEN:
        tmp_data = np.array(mag_data[start:start + config.WINDOW_LEN]).reshape(1, config.WINDOW_LEN)
        emp = np.vstack((emp, tmp_data))
        start += int(config.SLIDE_RATE * config.WINDOW_LEN)
    data = emp[1:].reshape(-1, config.WINDOW_LEN)
    return data

# FFT
def fft_transform(data):
    transformed = np.fft.fft(data)
    transformed = np.abs(transformed)
    return transformed

# IFFT
def ifft_transform(data):
    transformed = np.fft.ifft(data)
    transformed = np.abs(transformed)
    return transformed

# 高斯滤波
def gaussian_filter(data, sigma):
    import scipy.ndimage
    gaussian_data = scipy.ndimage.filters.gaussian_filter1d(data, sigma)
    return gaussian_data


if __name__ == '__main__':
    folder_name = './raw_mag_data'
    save_path = './data_np'
    prepare_data(folder_name, save_path)