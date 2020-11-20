import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np 
import time

import config.configure as config
from layers.attention import Attention

####################################################
# Seq2Seq LSTM AutoEncoder Model
#     - predict locations
####################################################
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, isCuda=False):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.isCuda = isCuda
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm = nn.GRU(input_size, hidden_size*30, num_layers, batch_first=True)
        
    def forward(self, input):
        output, hidden = self.lstm(input)
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout=0.5, isCuda=False):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.isCuda = isCuda
        # self.lstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=True)
        self.lstm = nn.GRU(hidden_size, output_size*30, num_layers, batch_first=True)

        #self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(output_size*30, output_size)
        self.tanh = nn.Tanh()
    
    def forward(self, encoded_input, hidden):
        decoded_output, hidden = self.lstm(encoded_input, hidden)
        # decoded_output = self.tanh(decoded_output)
        # decoded_output = self.sigmoid(decoded_output)
        decoded_output = self.dropout(decoded_output)
        # decoded_output = self.tanh(self.linear(decoded_output))
        decoded_output = self.linear(decoded_output)
        # decoded_output = self.sigmoid(self.linear(decoded_output))
        return decoded_output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.5, isCuda=False):
        super(Seq2Seq, self).__init__()
        self.isCuda = isCuda
        self.pred_length = 6
        self.encoder = EncoderRNN(input_size, hidden_size, num_layers, isCuda)
        self.decoder = DecoderRNN(hidden_size, hidden_size, num_layers, dropout, isCuda)
        self.encoder_map = EncoderRNN(input_size, hidden_size, num_layers, isCuda)
        self.attention = Attention(hidden_size*30)

    def forward(self, in_data, last_location, map_data, pred_length:int=6, map_mask=torch.Tensor()): #, teacher_forcing_ratio:int=0, teacher_location=torch.zeros(1)
        
        
        batch_size = in_data.shape[0]
        out_dim = self.decoder.output_size
        self.pred_length = pred_length

        outputs = torch.zeros(batch_size, self.pred_length, out_dim)
        # hidden = torch.zeros(self.encoder.lstm.num_layers, batch_size, self.encoder.lstm.hidden_size)
        if self.isCuda:
            outputs = outputs.cuda()    
            # hidden = hidden.cuda()
            #    
        encoded_output, hidden = self.encoder(in_data) #in_data (N, T, C) --> hidden (L, N, H)



        # N, V, mV = att.shape
        # argmax_att = torch.argmax(att, dim=-1)
        # for i in range(N):
        #     iv = i * V
        #     imv = i * mV
        #     for j in range(V):
        #         hidden[:, iv + j] = hidden[:, iv + j] + map_hidden[:, imv + argmax_att[i, j]]
        if config.use_map:
            map_encoded_output, map_hidden = self.encoder(map_data) #map_data (NmV, mT, C) --> map_hidden (L, NmV, H)
            map_hidden, att = self.attention(hidden, map_hidden, map_mask) #map_hidden (L, N, H) , att (N, mV)
            hidden = hidden + map_hidden
            # hidden = torch.cat((hidden, map_hidden), dim=-1)
            hidden = torch.tanh(hidden)
        decoder_input = last_location
        for t in range(self.pred_length):
            # encoded_input = torch.cat((now_label, encoded_input), dim=-1) # merge class label into input feature
            now_out, hidden = self.decoder(decoder_input, hidden)
            now_out += decoder_input
            outputs[:,t:t+1] = now_out 
            # teacher_force = np.random.random() < teacher_forcing_ratio
            # decoder_input = (teacher_location[:,t:t+1] if (teacher_location.dim() > 1) and teacher_force else now_out)
            decoder_input = now_out

        # att = None
        return outputs.permute(0,2,1).unsqueeze(-1), att

####################################################
####################################################