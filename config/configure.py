import torch
import os

# data process
total_feature_dimension = 10 + 1
data_time_resolution = 0.1
frame_steps = 2
history_frames = 15 # 3 second * 2 frame/second
future_frames = 10 # 3 second * 2 frame/second
# xy_range = 120 # max_x_range=121, max_y_range=118
max_num_object = 150 # maximum number of observed objects is 70
neighbor_distance = 10 # meter

data_root = 'data/our_data/0916_type'
data_root = 'data/argo'
# data_root = "data/xincoder/ApolloScape"
work_dir = 'trained_models/argo'
train_data_path = "prediction_train/"
test_data_path = "prediction_test/"
train_data_file = 'train_data_%d_%d_%d.pkl'%(frame_steps, history_frames, future_frames)
test_data_file = 'test_data_%d_%d_%d.pkl'%(frame_steps, history_frames, future_frames)



# train param
batch_size_train = 64 
batch_size_val = 32
batch_size_test = 1
total_epoch = 500
base_lr = 0.001
lr_decay_epoch = 5
lr_decay = 0.5
in_channels=4
dropout = 0.5

dev = torch.device("cpu")
use_cuda = False
if torch.cuda.is_available():
    dev = torch.device("cuda")
    use_cuda = True
log_file = os.path.join(work_dir,'log_test.txt')
test_result_file = 'prediction_result.txt'

max_hop = 2
num_node = max_num_object
graph_args={'max_hop':max_hop, 'num_node':num_node}

train = False
load_model = False
convert_model = False
pretrained_model_path = 'trained_models/argo/model_epoch_0026_1016_16:16:57.pt'
save_view = True
save_view_path = os.path.join(work_dir, "view")
if not os.path.exists(save_view_path):
    os.makedirs(save_view_path)

# model param
spatial_kernel_size = max_hop + 1
temporal_kernel_size = 5
gcn_hidden_size = 64
seq2seq_dropout = 0.5
encoder_input_size = 64
encoder_hidden_size = 30
out_dim_per_node = 2