import torch
import os

# data process
total_feature_dimension = 10 + 1
data_time_resolution = 0.1
frame_steps = 1
history_frames = 20 # 3 second * 2 frame/second
future_frames = 30 # 3 second * 2 frame/second
# xy_range = 120 # max_x_range=121, max_y_range=118
max_num_object = 1 # maximum number of observed objects is 70
max_num_map = 10 # nearby_lanes 32
neighbor_distance = 10 # meter

# map param
only_nearby_lanes = True
lane_search_radius = 4
segment_point_num = 50
map_path = "util/map/base_map.bin"
map_type = "argo"


# data_root = 'data/our_data/0908'
data_root = '/datastore/data/cxl/new_model/data/argo/all_data'
# data_root = '/datastore/data/cxl/GRIP/data/argo/no_slow_obj_grip'
# data_root = "data/xincoder/ApolloScape"
work_dir = '/datastore/data/cxl/GRIP/trained_models/argo/all_data'
train_data_path = "prediction_train/"
val_data_path = "prediction_val/"
test_data_path = "prediction_test/"
# train_data_file = 'train_data_%d_%d_%d.pkl'%(frame_steps, history_frames, future_frames)
# test_data_file = 'test_data_%d_%d_%d.pkl'%(frame_steps, history_frames, future_frames)
train_data_file = 'train_data_%d_%d_%d_%d.pkl'%(frame_steps, history_frames, future_frames, lane_search_radius)
test_data_file = 'test_data_%d_%d_%d_%d.pkl'%(frame_steps, history_frames, future_frames, lane_search_radius)
val_data_file = 'val_data_%d_%d_%d_%d.pkl'%(frame_steps, history_frames, future_frames, lane_search_radius)
save_model_prefix = "model_argo_all_sl_vel_encode_"



# train param
batch_size_train = 64 
batch_size_val = 32
batch_size_test = 1
total_epoch = 25
base_lr = 0.0001
lr_decay_epoch = 5
lr_decay = 0.5
dropout = 0.5
sample_times = 6
random_threshold = 1

log_file = os.path.join(work_dir,'log_test.txt')
test_result_file = 'prediction_result.txt'

max_hop = 2
num_node = max_num_object
graph_args={'max_hop':max_hop, 'num_node':num_node}
loss_weight = [1, 1, 1]

train = True
load_model = False
vel_mode = True
use_map = True
use_celoss = True
use_history = False
multi_lane = True
convert_model = False

pretrained_model_path = '/datastore/data/cxl/GRIP/trained_models/argo/all_data/model_argo_all_sl_vel_0107_19:16:41_epoch_0024.pt'
view = True 
save_view = True
save_view_path = os.path.join(work_dir, "view", pretrained_model_path.split(".")[0].split("/")[-1])
if not train and not os.path.exists(save_view_path):
    os.makedirs(save_view_path)
    print("save view path: [%s]"%save_view_path)

dev = torch.device("cpu")
use_cuda = False
if torch.cuda.is_available():#not convert_model and 
    dev = torch.device("cuda")
    use_cuda = True
    
# model param
in_channels = 4
spatial_kernel_size = max_hop + 1
temporal_kernel_size = 5
gcn_hidden_size = 64
seq2seq_dropout = 0.5
num_lstm_layers = 1
encoder_input_size = gcn_hidden_size
encoder_hidden_size = 30
out_dim_per_node = 2