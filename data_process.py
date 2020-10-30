import numpy as np 
import glob
import os 
from scipy import spatial 
import pickle
from tqdm import tqdm

from argoverse.map_representation.map_api import ArgoverseMap

import config.configure as config
from util.map_util import HDMap, LaneSegment

if config.map_type == "argo":
    print("loading ArgoverseMap")
    avm = ArgoverseMap()
else:
    print("loading map form %s"%config.map_path)
    hdmap = HDMap(config.map_path)
# Please change this to your location
data_root = config.data_root


history_frames = config.history_frames
future_frames = config.future_frames
frame_steps = config.frame_steps
total_frames = history_frames + future_frames
# xy_range = 120 # max_x_range=121, max_y_range=118
max_num_object = config.max_num_object # maximum number of observed objects is 70
max_num_map = config.max_num_map
neighbor_distance = config.neighbor_distance # meter

map_num = 0
# Baidu ApolloScape data format:
# frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading
total_feature_dimension = config.total_feature_dimension # we add mark "1" to the end of each row to indicate that this row exists

# after zero centralize data max(x)=127.1, max(y)=106.1, thus choose 130

def get_frame_instance_dict(pra_file_path):
    '''
    Read raw data from files and return a dictionary: 
        {frame_id: 
            {object_id: 
                # 10 features
                [frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading]
            }
        }
    '''
    with open(pra_file_path, 'r') as reader:
        # print(train_data_path)\
        city = ""
        if config.map_type == "argo":
            city = reader.readline().strip().split(' ')[-1]
        content = np.array([x.strip().split(' ')[:config.total_feature_dimension - 1] for x in reader.readlines()]).astype(float)
        now_dict = {}
        for row in content:
            # instance = {row[1]:row[2:]}
            n_dict = now_dict.get(row[0], {})
            n_dict[row[1]] = row#[2:]
            # n_dict.append(instance)
            # now_dict[]
            now_dict[row[0]] = n_dict
    return now_dict, city


def process_map_data(center, search_radius, point_num, city=""):
    global map_num
    map_feature_list = []
    if config.map_type == "argo":
        idList = avm.get_lane_ids_in_xy_bbox(center[0], center[1], city, search_radius)
        for l, id in enumerate(idList):
            lane = avm.city_lane_centerlines_dict[city][id]
            t = 0
            if lane.turn_direction == 'LEFT':
                t = 1
            elif lane.turn_direction == 'RIGHT':
                t = 2
            segment = lane.centerline
            now_map_feature = np.zeros((point_num, total_feature_dimension))
            for i in range(segment.shape[0]):
                now_map_feature[i] = np.array([l, 0, 0, segment[i][0] - center[0], segment[i][1] - center[1], 0, 0, 0, 0, t, 1], dtype="float")
            map_feature_list.append(now_map_feature)

    else:
        lane_list = hdmap.get_lanes(center, search_radius, point_num)
        for l, lane in enumerate(lane_list):
            lane_id = lane.id
            for segment in lane.segment:
                now_map_feature = np.zeros((point_num, total_feature_dimension))
                segment = np.array(segment).astype("float")
                for i in range(segment.shape[0]):
                    now_map_feature[i] = np.array([l, 0, 0, segment[i][0] - center[0], segment[i][1] - center[1], 0, 0, 0, 0, lane.turn, 1], dtype="float")
                map_feature_list.append(now_map_feature)

    map_feature_list = np.array(map_feature_list, dtype="float")
    if map_feature_list.shape[0] > map_num:
        map_num = map_feature_list.shape[0]
        print("\nmap_num: %d"%map_num)
    return map_feature_list



def process_data(pra_now_dict, pra_start_ind, pra_end_ind, pra_observed_last, city=""):
    visible_object_id_list = list(pra_now_dict[pra_observed_last].keys()) # object_id appears at the last observed frame 
    num_visible_object = len(visible_object_id_list) # number of current observed objects

    # compute the mean values of x and y for zero-centralization. 
    visible_object_value = np.array(list(pra_now_dict[pra_observed_last].values()))
    xy = visible_object_value[:, 3:5].astype(float)
    mean_xy = np.zeros_like(visible_object_value[0], dtype=float)
    m_xy = np.mean(xy, axis=0)
    mean_xy[3:5] = m_xy

    # compute distance between any pair of two objects
    dist_xy = spatial.distance.cdist(xy, xy)
    # if their distance is less than $neighbor_distance, we regard them are neighbors.
    neighbor_matrix = np.zeros((max_num_object, max_num_object))
    neighbor_matrix[:num_visible_object, :num_visible_object] = (dist_xy<neighbor_distance).astype(int)

    now_all_object_id = set([val for x in range(pra_start_ind, pra_end_ind, frame_steps) for val in pra_now_dict[x].keys()])
    non_visible_object_id_list = list(now_all_object_id - set(visible_object_id_list))
    num_non_visible_object = len(non_visible_object_id_list)
    
    # for all history frames(6) or future frames(6), we only choose the objects listed in visible_object_id_list
    object_feature_list = []
    # non_visible_object_feature_list = []
    for frame_ind in range(pra_start_ind, pra_end_ind, frame_steps):    
        # we add mark "1" to the end of each row to indicate that this row exists, using list(pra_now_dict[frame_ind][obj_id])+[1] 
        # -mean_xy is used to zero_centralize data
        # now_frame_feature_dict = {obj_id : list(pra_now_dict[frame_ind][obj_id]-mean_xy)+[1] for obj_id in pra_now_dict[frame_ind] if obj_id in visible_object_id_list}
        now_frame_feature_dict = {obj_id : (list(pra_now_dict[frame_ind][obj_id]-mean_xy)+[1] if obj_id in visible_object_id_list else list(pra_now_dict[frame_ind][obj_id]-mean_xy)+[0]) for obj_id in pra_now_dict[frame_ind] }
        # if the current object is not at this frame, we return all 0s by using dict.get(_, np.zeros(11))
        now_frame_feature = np.array([now_frame_feature_dict.get(vis_id, np.zeros(total_feature_dimension)) for vis_id in visible_object_id_list+non_visible_object_id_list])
        object_feature_list.append(now_frame_feature)

    # object_feature_list has shape of (frame#, object#, 11) 11 = 10features + 1mark
    object_feature_list = np.array(object_feature_list)
    map_feature_list = process_map_data(m_xy, config.lane_search_radius, config.segment_point_num, city)
    # object feature with a shape of (frame#, object#, 11) -> (object#, frame#, 11)
    object_frame_feature = np.zeros((max_num_object, round((pra_end_ind-pra_start_ind) / frame_steps), total_feature_dimension))

    map_frame_feature = np.zeros((max_num_map, config.segment_point_num, total_feature_dimension))
    try:
        map_frame_feature[:map_feature_list.shape[0]] = map_feature_list
    except:
        print(map_feature_list)
    
    # np.transpose(object_feature_list, (1,0,2))
    object_frame_feature[:num_visible_object+num_non_visible_object] = np.transpose(object_feature_list, (1,0,2))
    
    return object_frame_feature, neighbor_matrix, m_xy, map_frame_feature
    

def generate_train_data(pra_file_path):
    '''
    Read data from $pra_file_path, and split data into clips with $total_frames length. 
    Return: feature and adjacency_matrix
        feture: (N, C, T, V) 
            N is the number of training data 
            C is the dimension of features, 10raw_feature + 1mark(valid data or not)
            T is the temporal length of the data. history_frames + future_frames
            V is the maximum number of objects. zero-padding for less objects. 
    '''
    now_dict, city = get_frame_instance_dict(pra_file_path)
    frame_id_set = sorted(set(now_dict.keys()))

    all_feature_list = []
    all_adjacency_list = []
    all_mean_list = []
    all_map_list = []
    for start_ind in frame_id_set[:-total_frames+1:frame_steps]:
        start_ind = int(start_ind)
        end_ind = int(start_ind + total_frames * frame_steps)
        observed_last = start_ind + (history_frames - 1)*frame_steps
        for ind in range(start_ind, end_ind, frame_steps):
            if ind not in frame_id_set:
                break
        else:
            object_frame_feature, neighbor_matrix, mean_xy, map_frame_feature= process_data(now_dict, start_ind, end_ind, observed_last, city)

            all_feature_list.append(object_frame_feature)
            all_adjacency_list.append(neighbor_matrix)    
            all_mean_list.append(mean_xy)  
            all_map_list.append(map_frame_feature)  

    # (N, V, T, C) --> (N, C, T, V)
    all_feature_list = np.array(all_feature_list)
    all_adjacency_list = np.array(all_adjacency_list)
    all_mean_list = np.array(all_mean_list)
    all_map_list = np.array(all_map_list)
    if all_feature_list.shape[0] > 0:
        all_feature_list = np.transpose(all_feature_list, (0, 3, 2, 1))
        all_map_list = np.transpose(all_map_list, (0, 3, 2, 1))
    # print(all_feature_list.shape, all_adjacency_list.shape)
    return all_feature_list, all_adjacency_list, all_mean_list, all_map_list


def generate_test_data(pra_file_path):
    now_dict, city = get_frame_instance_dict(pra_file_path)
    frame_id_set = sorted(set(now_dict.keys()))
    
    all_feature_list = []
    all_adjacency_list = []
    all_mean_list = []
    all_map_list = []

    # get all start frame id
    start_frame_id_list = frame_id_set[::history_frames]
    for start_ind in start_frame_id_list:
        start_ind = int(start_ind)
        end_ind = int(start_ind + history_frames)
        observed_last = start_ind + history_frames - 1
        # print(start_ind, end_ind)
        for ind in range(start_ind, end_ind):
            if ind not in frame_id_set:
                break
        else:
            object_frame_feature, neighbor_matrix, mean_xy, map_frame_feature = process_data(now_dict, start_ind, end_ind, observed_last, city)

            all_feature_list.append(object_frame_feature)
            all_adjacency_list.append(neighbor_matrix)    
            all_mean_list.append(mean_xy)
            all_map_list.append(map_frame_feature)  

    # (N, V, T, C) --> (N, C, T, V)
    all_feature_list = np.array(all_feature_list)
    all_adjacency_list = np.array(all_adjacency_list)
    all_mean_list = np.array(all_mean_list)
    all_map_list = np.array(all_map_list)
    if all_feature_list.shape[0] > 0:
        all_feature_list = np.transpose(all_feature_list, (0, 3, 2, 1))
        all_map_list = np.transpose(all_map_list, (0, 3, 2, 1))
    # print(all_feature_list.shape, all_adjacency_list.shape)
    return all_feature_list, all_adjacency_list, all_mean_list, all_map_list


def generate_data(pra_file_path_list, pra_is_train=True):
    all_data = []
    all_adjacency = []
    all_mean_xy = []
    all_map_data = []
    for file_path in tqdm(pra_file_path_list):
        # print(file_path)
        if pra_is_train:
            now_data, now_adjacency, now_mean_xy, now_map_data = generate_train_data(file_path)
        else:
            now_data, now_adjacency, now_mean_xy, now_map_data = generate_train_data(file_path)
        if now_data.shape[0] > 0:
            all_data.extend(now_data)
            all_adjacency.extend(now_adjacency)
            all_mean_xy.extend(now_mean_xy)
            all_map_data.extend(now_map_data)

    all_data = np.array(all_data) #(N, C, T, V)=(5010, 11, 12, 70) Train
    all_adjacency = np.array(all_adjacency) #(5010, 70, 70) Train
    all_mean_xy = np.array(all_mean_xy) #(5010, 2) Train
    all_map_data = np.array(all_map_data)

    # Train (N, C, T, V)=(5010, 11, 12, 70), (5010, 70, 70), (5010, 2)
    # Test (N, C, T, V)=(415, 11, 6, 70), (415, 70, 70), (415, 2)
    print(np.shape(all_data), np.shape(all_adjacency), np.shape(all_mean_xy), np.shape(all_map_data))

    # save training_data and trainjing_adjacency into a file.
    if pra_is_train:
        save_path = os.path.join(data_root, train_data_path, config.train_data_file)
    else:
        save_path = os.path.join(data_root, test_data_path, config.test_data_file)
    with open(save_path, 'wb') as writer:
        pickle.dump([all_data, all_adjacency, all_mean_xy, all_map_data], writer)


if __name__ == '__main__':
    train_data_path = config.train_data_path
    test_data_path = config.test_data_path
    # train_data_path = "prediction_train/"
    train_data_path_list = sorted(glob.glob(os.path.join(data_root, train_data_path, '*.txt')), key=lambda x: int(x.split(".")[-2].split("_")[-1]))
    test_data_path_list  = sorted(glob.glob(os.path.join(data_root, test_data_path,  '*.txt')), key=lambda x: int(x.split(".")[-2].split("_")[-1]))

    print('Generating Training Data.')
    generate_data(train_data_path_list, pra_is_train=True)
    
    print('Generating Testing Data.')
    generate_data(test_data_path_list, pra_is_train=False)
    


