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
        self.lstm = nn.GRU(hidden_size, output_size*60, num_layers, batch_first=True)

        #self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(output_size*30, output_size)
        self.tanh = nn.Tanh()
    
    def forward(self, encoded_input, hidden):
        decoded_output, hidden = self.lstm(encoded_input, hidden)
        # decoded_output = self.tanh(decoded_output)
        # decoded_output = self.sigmoid(decoded_output)
        # output = self.dropout(decoded_output)
        # decoded_output = self.tanh(self.linear(decoded_output))
        # output = self.linear(decoded_output)
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
        self.maxpool = nn.MaxPool1d(hidden_size*30)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_size*60, hidden_size)
        

    def forward(self, in_data, last_location, map_data, pred_length:int=6, mask=torch.Tensor(), map_mask=torch.Tensor()): #, teacher_forcing_ratio:int=0, teacher_location=torch.zeros(1)
        
        
        batch_size = in_data.shape[0] // config.max_num_map
        out_dim = self.decoder.output_size
        history_frames = in_data.shape[1]

        outputs = torch.zeros(in_data.shape[0], pred_length + history_frames, out_dim)
        # hidden = torch.zeros(self.encoder.lstm.num_layers, batch_size, self.encoder.lstm.hidden_size)
        if self.isCuda:
            outputs = outputs.cuda()    
            # hidden = hidden.cuda()
            #    
        encoded_output, hidden = self.encoder(in_data) #in_data (NV, T, C) -->encoded_output (NV, T, H) hidden (L, NV, H)
        if self.training:
            # encoded_output = encoded_output.reshape((batch_size, -1, encoded_output.shape[-2], encoded_output.shape[-1]))[:,0]
            history_out = self.dropout(encoded_output)
            history_out = self.linear(history_out)
            outputs[:, :history_frames] = history_out

        if config.use_map:
            map_encoded_output, map_hidden = self.encoder(map_data) #map_data (NmV, mT, C) --> map_hidden (L, NmV, H)
            map_hidden = map_hidden.view((map_hidden.shape[0], batch_size, -1, hidden.shape[-1])).contiguous()
            map_hidden, att = self.attention(map_hidden, map_hidden, map_mask) #map_hidden (L, N, H) , att (N, mV)
            # hidden = hidden + map_hidden
            hidden = torch.cat((hidden, map_hidden), dim=-1)
            # hidden = torch.tanh(hidden)
        else:
            curr_hidden = hidden.view((hidden.shape[0], batch_size, -1, hidden.shape[-1])).contiguous() #(L, N, V, H)
            # last_location = last_location.view(batch_size, -1, last_location.shape[-2], last_location.shape[-1]).contiguous()
            if config.multi_lane:
                _, att = self.attention(curr_hidden, curr_hidden, mask)
                # curr_hidden = self.maxpool(hidden[-1]).squeeze(-1) #(N, V)
                # zero_tensor = torch.ones_like(curr_hidden) * -1e1
                # curr_hidden = torch.where(mask, curr_hidden, zero_tensor)
                # att = torch.softmax(curr_hidden, dim=1) #(N, V)
                # argmax_att = torch.argmax(att, dim=-1)
                # hidden = hidden.gather(dim=2, index=argmax_att.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat((hidden.shape[0], 1, 1, hidden.shape[-1]))).squeeze(-2) #(L, N, H)
                # last_location = last_location.gather(dim=1, index=argmax_att.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat((1, 1, last_location.shape[-2], last_location.shape[-1]))).squeeze(1)
            else:
                hidden = hidden[:,:,0].contiguous()
                last_location = last_location[:,0].contiguous()
                att = torch.zeros((batch_size, config.max_num_map)).to(config.dev)
                att[:,0] = 1
        decoder_input = last_location
        for t in range(history_frames, outputs.shape[1]):
            decoder_out, hidden = self.decoder(decoder_input, hidden)
            now_out = self.dropout(decoder_out)
            now_out = self.linear(now_out)
            now_out += decoder_input
            outputs[:,t:t+1] = now_out 
            # teacher_force = np.random.random() < teacher_forcing_ratio
            # decoder_input = (teacher_location[:,t:t+1] if (teacher_location.dim() > 1) and teacher_force else now_out)
            decoder_input = now_out
        outputs = outputs.reshape(batch_size, -1, outputs.shape[-2], outputs.shape[-1])
        # att = None
        if self.training:
            return outputs.permute(0,3,2,1), att
        else:
            return outputs[:,:,history_frames:].permute(0,3,2,1), att

####################################################
####################################################