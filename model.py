import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from layers.graph import Graph
from layers.graph_conv_block import Graph_Conv_Block
from layers.seq2seq import Seq2Seq, EncoderRNN
import numpy as np 
import config.configure as config


class Model(nn.Module):
    def __init__(self, in_channels, graph_args, edge_importance_weighting, use_cuda, dropout):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = np.ones((graph_args['max_hop']+1, graph_args['num_node'], graph_args['num_node'])) # (3, 120, 120)

        # build networks
        # spatial_kernel_size = np.shape(A)[0]
        spatial_kernel_size = config.spatial_kernel_size
        temporal_kernel_size = config.temporal_kernel_size #9 #5 # 3
        kernel_size = (temporal_kernel_size, spatial_kernel_size) # (5, 3)
        gcn_hidden_size = config.gcn_hidden_size
        encoder_input_size = config.encoder_input_size # 64
        seq2seq_dropout = config.seq2seq_dropout
        # best
        # self.conv_networks = nn.ModuleList((
        #     nn.BatchNorm2d(in_channels),
        #     nn.Conv2d(in_channels, gcn_hidden_size, kernel_size=1, stride=(1,1)),
        #     nn.BatchNorm2d(gcn_hidden_size)
        # ))
        # self.st_gcn_networks = nn.ModuleList((            
        #     Graph_Conv_Block(gcn_hidden_size, gcn_hidden_size, kernel_size, 1, dropout),
        #     Graph_Conv_Block(gcn_hidden_size, gcn_hidden_size, kernel_size, 1, dropout),
        #     Graph_Conv_Block(gcn_hidden_size, gcn_hidden_size, kernel_size, 1, dropout),
        # ))
        self.conv_network = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, gcn_hidden_size, kernel_size=1, stride=(1,1)),
            nn.BatchNorm2d(gcn_hidden_size)
        )
        self.gcn_networks_1 = Graph_Conv_Block(gcn_hidden_size, gcn_hidden_size, kernel_size, 1, dropout)
        self.gcn_networks_2 = Graph_Conv_Block(gcn_hidden_size, gcn_hidden_size, kernel_size, 1, dropout)
        self.gcn_networks_3 = Graph_Conv_Block(gcn_hidden_size, gcn_hidden_size, kernel_size, 1, dropout)

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones(np.shape(A))) for i in range(3)]
                )
        else:
            self.edge_importance = [1] * 3

        self.num_node = self.graph.num_node
        self.out_dim_per_node = out_dim_per_node = config.out_dim_per_node # 2 (x, y) coordinate
        self.seq2seq_car = Seq2Seq(input_size=(encoder_input_size), hidden_size=out_dim_per_node, num_layers=2, dropout=seq2seq_dropout, isCuda=use_cuda)
        self.seq2seq_human = Seq2Seq(input_size=(encoder_input_size), hidden_size=out_dim_per_node, num_layers=2, dropout=seq2seq_dropout, isCuda=use_cuda)
        self.seq2seq_bike = Seq2Seq(input_size=(encoder_input_size), hidden_size=out_dim_per_node, num_layers=2, dropout=seq2seq_dropout, isCuda=use_cuda)


    def reshape_for_lstm(self, feature):
        # prepare for skeleton prediction model
        '''
        N: batch_size
        C: channel
        T: time_step
        V: nodes
        '''
        N, C, T, V = feature.size() 
        now_feat = feature.permute(0, 3, 2, 1).contiguous() # to (N, V, T, C)
        now_feat = now_feat.view(N*V, T, C) 
        return now_feat

    def reshape_from_lstm(self, predicted):
        # predicted (N*V, T, C)
        NV, T, C = predicted.size()
        now_feat = predicted.view(-1, self.num_node, T, self.out_dim_per_node) # (N, T, V, C) -> (N, C, T, V) [(N, V, T, C)]
        now_feat = now_feat.permute(0, 3, 2, 1).contiguous() # (N, C, T, V)
        return now_feat

    def forward(self, pra_x:torch.Tensor, pra_A:torch.Tensor, pra_pred_length:int=6): #, pra_teacher_forcing_ratio:int=0, pra_teacher_location=torch.zeros(1)
        x = pra_x # (N, 4, 6, 120)
        
        # forwad
        # for network in self.conv_networks:
        #     x = network(x)

        # for gcn in self.st_gcn_networks:
        #     # importance = self.edge_importance[ii]
        #     # if type(gcn) is nn.BatchNorm2d or type(gcn) is nn.Conv2d:
        #     #     x = gcn(x)
        #     # else:
        #     # x, _ = gcn(x, pra_A + importance)
        #     x, _ = gcn(x, pra_A)
        x = self.conv_network(x)
        x, _ = self.gcn_networks_1(x, pra_A)
        x, _ = self.gcn_networks_2(x, pra_A)
        x, _ = self.gcn_networks_3(x, pra_A)
                
        # prepare for seq2seq lstm model
        graph_conv_feature = self.reshape_for_lstm(x)
        last_position = self.reshape_for_lstm(pra_x[:,:2]) #(N, C, T, V)[:, :2] -> (N, T, V*2) [(N*V, T, C)]
        # if pra_teacher_forcing_ratio>0 and pra_teacher_location.dim() > 1: #
        #     pra_teacher_location = self.reshape_for_lstm(pra_teacher_location)

        # now_predict.shape = (N, T, V*C)
        self.num_node = x.shape[-1]
        now_predict_car = self.seq2seq_car(in_data=graph_conv_feature, last_location=last_position[:,-1:,:], pred_length=pra_pred_length) #, teacher_forcing_ratio=pra_teacher_forcing_ratio, teacher_location=pra_teacher_location
        now_predict_car = self.reshape_from_lstm(now_predict_car) # (N, C, T, V)

        now_predict_human = self.seq2seq_human(in_data=graph_conv_feature, last_location=last_position[:,-1:,:], pred_length=pra_pred_length) #, teacher_forcing_ratio=pra_teacher_forcing_ratio, teacher_location=pra_teacher_location
        now_predict_human = self.reshape_from_lstm(now_predict_human) # (N, C, T, V)

        now_predict_bike = self.seq2seq_bike(in_data=graph_conv_feature, last_location=last_position[:,-1:,:], pred_length=pra_pred_length) #, teacher_forcing_ratio=pra_teacher_forcing_ratio, teacher_location=pra_teacher_location
        now_predict_bike = self.reshape_from_lstm(now_predict_bike) # (N, C, T, V)

        now_predict = (now_predict_car + now_predict_human + now_predict_bike)/3.

        return now_predict 

if __name__ == '__main__':
    model = Model(in_channels=3, pred_length=6, graph_args={}, edge_importance_weighting=True)
    print(model)
