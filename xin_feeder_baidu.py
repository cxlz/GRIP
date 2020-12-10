import os
import sys
import numpy as np
import random
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# from data_process import generate_data

from layers.graph import Graph
import config.configure as config

import time


class Feeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
    """

    def __init__(self, data_path, graph_args={}, train_val_test='train'):
        '''
        train_val_test: (train, val, test)
        '''
        self.data_path = data_path
        self.load_data()

        total_num = len(self.all_feature)
        # equally choose validation set
        train_id_list = list(np.linspace(0, total_num-1, int(total_num*0.8)).astype(int))
        val_id_list = list(set(list(range(total_num))) - set(train_id_list))

        # # last 20% data as validation set
        self.train_val_test = train_val_test

        if train_val_test.lower() == 'train':
            self.all_feature = self.all_feature[train_id_list]
            self.all_adjacency = self.all_adjacency[train_id_list]
            self.all_mean_xy = self.all_mean_xy[train_id_list]
            self.all_map_feature = self.all_map_feature[train_id_list]
            self.all_lane_label = self.all_lane_label[train_id_list]
            self.all_trajectory = self.all_trajectory[train_id_list]
            self.all_seq_id_city = self.all_seq_id_city[train_id_list]

        elif train_val_test.lower() == 'val':
            self.all_feature = self.all_feature[val_id_list]
            self.all_adjacency = self.all_adjacency[val_id_list]
            self.all_mean_xy = self.all_mean_xy[val_id_list]
            self.all_map_feature = self.all_map_feature[val_id_list]
            self.all_lane_label = self.all_lane_label[val_id_list]
            self.all_trajectory = self.all_trajectory[val_id_list]
            self.all_seq_id_city = self.all_seq_id_city[val_id_list]

        self.graph = Graph(**graph_args) #num_node = 120,max_hop = 1

    def load_data(self):
        with open(self.data_path, 'rb') as reader:
            # Training (N, C, T, V)=(5010, 11, 12, 120), (5010, 120, 120), (5010, 2)
            [
                self.all_feature, 
                self.all_adjacency, 
                self.all_mean_xy, 
                self.all_map_feature, 
                self.all_lane_label, 
                self.all_trajectory, 
                self.all_seq_id_city,
            ] = pickle.load(reader)
            

    def __len__(self):
        return len(self.all_feature)

    def __getitem__(self, idx):
        # C = 11: [frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading] + [mask]
        now_feature = self.all_feature[idx].copy() # (C, T, V) = (11, 12, 120)
        now_mean_xy = self.all_mean_xy[idx].copy() # (2,) = (x, y) 
        now_map_feature = self.all_map_feature[idx].copy() # (C, T, V) = (11, 10, 100)
        now_lane_label = self.all_lane_label[idx].copy()
        now_trajectory = self.all_trajectory[idx].copy()
        now_seq_id_city = self.all_seq_id_city[idx].copy()

        if self.train_val_test.lower() == 'train' and np.random.random()>0.5:
            angle = 2 * np.pi * np.random.random()
            sin_angle = np.sin(angle)
            cos_angle = np.cos(angle)

            angle_mat = np.array(
                [[cos_angle, -sin_angle],
                [sin_angle, cos_angle]])

            xy = now_feature[3:5, :, :]
            map_xy = now_map_feature[3:5, :, :]
            num_xy = np.sum(xy.sum(axis=0).sum(axis=0) != 0) # get the number of valid data
            num_map_xy = np.sum(map_xy.sum(axis=0).sum(axis=0) != 0) # get the number of valid data

            # angle_mat: (2, 2), xy: (2, 12, 120)
            out_xy = np.einsum('ab,btv->atv', angle_mat, xy)
            out_map_xy = np.einsum('ab,btv->atv', angle_mat, map_xy)
            xy[:,:,:num_xy] = out_xy[:,:,:num_xy]
            map_xy[:,:,:num_map_xy] = out_map_xy[:,:,:num_map_xy]
            now_mean_xy = np.matmul(angle_mat, now_mean_xy)

            now_feature[3:5, :, :] = xy
            now_map_feature[3:5, :, :] = map_xy

        now_adjacency = self.graph.get_adjacency(self.all_adjacency[idx])
        now_A = self.graph.normalize_adjacency(now_adjacency)
        if now_seq_id_city[1].lower() == "PIT":
            now_seq_id_city[1] = 0
        else:
            now_seq_id_city[1] = 1
        now_seq_id_city = now_seq_id_city.astype("float")
        
        return now_feature[:,::config.frame_steps], now_A, now_mean_xy, now_map_feature, now_lane_label, now_trajectory[:,::config.frame_steps], now_seq_id_city



# def my_collate_fn(batch_data):
#     train_val_test = batch_data[0][1]
#     graph:Graph = batch_data[0][2]
#     nameList = []
#     for name, _, _ in batch_data:
#         nameList.append(name)
#     train = False
#     if train_val_test == "train" or train_val_test == "val":
#         train = True
#     now_feature, now_adjacency, now_mean_xy, now_map_feature, now_lane_label = generate_data(nameList, train, save_data=False)
#     now_A = np.transpose(np.zeros((3, *now_adjacency.shape)), (1, 0, 2, 3))
#     for idx in range(now_feature.shape[0]):
#         if train_val_test.lower() == 'train' and np.random.random()>0.5:
#             angle = 2 * np.pi * np.random.random()
#             sin_angle = np.sin(angle)
#             cos_angle = np.cos(angle)

#             angle_mat = np.array(
#                 [[cos_angle, -sin_angle],
#                 [sin_angle, cos_angle]])

#             xy = now_feature[idx, 3:5, :, :]
#             map_xy = now_map_feature[idx, 3:5, :, :]
#             num_xy = np.sum(xy.sum(axis=0).sum(axis=0) != 0) # get the number of valid data
#             num_map_xy = np.sum(map_xy.sum(axis=0).sum(axis=0) != 0) # get the number of valid data

#             # angle_mat: (2, 2), xy: (2, 12, 120)
#             out_xy = np.einsum('ab,btv->atv', angle_mat, xy)
#             out_map_xy = np.einsum('ab,btv->atv', angle_mat, map_xy)
#             xy[:,:,:num_xy] = out_xy[:,:,:num_xy]
#             map_xy[:,:,:num_map_xy] = out_map_xy[:,:,:num_map_xy]
#             now_mean_xy[idx] = np.matmul(angle_mat, now_mean_xy[idx])

#             now_feature[idx, 3:5, :, :] = xy
#             now_map_feature[idx, 3:5, :, :] = map_xy

#         now_adjacency[idx] = graph.get_adjacency(now_adjacency[idx])
#         now_A[idx] = graph.normalize_adjacency(now_adjacency[idx])
#     now_feature = torch.from_numpy(now_feature)
#     now_A = torch.from_numpy(now_A)
#     now_adjacency = torch.from_numpy(now_adjacency)
#     now_map_feature = torch.from_numpy(now_map_feature)
#     now_lane_label = torch.from_numpy(now_lane_label)
#     return now_feature, now_A, now_mean_xy, now_map_feature, now_lane_label


# class new_Feeder(torch.utils.data.Dataset):

#     def __init__(self, DATA_PATH, graph_args={}, train_val_test='train'):
#         self.DATA_PATH = DATA_PATH
#         self.nameList = [os.path.join(DATA_PATH, name)  for name in sorted(os.listdir(DATA_PATH)) if name.endswith(".txt")]
#         train_size = round(len(self.nameList) * 0.8)
#         if train_val_test == "train":
#             self.nameList = self.nameList[:train_size]
#         elif train_val_test == "val":
#             self.nameList = self.nameList[train_size:]
#         self.train_val_test = train_val_test            
#         self.graph = Graph(**graph_args)
#         # self.X, self.Y = load_data(self.DATA_PATH, self.nameList)

#     def __getitem__(self, index):
#         return self.nameList[index], self.train_val_test, self.graph
#         # return self.X[index], self.Y[index]

#     def __len__(self):
#         return len(self.nameList)
        
        

