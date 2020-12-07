import numpy as np 
import glob
import os 
from scipy import spatial 
import pickle
from tqdm import tqdm
import cv2

from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils import centerline_utils
from shapely.geometry import LineString

import config.configure as config
# from util.map_util import HDMap, LaneSegment

if config.map_type == "argo":
    print("loading ArgoverseMap")
    avm = ArgoverseMap()
# else:
#     print("loading map form %s"%config.map_path)
#     hdmap = HDMap(config.map_path)
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


def visulize(tar, lane, a):
    max_xy = np.max(np.max(lane, axis=-1), axis=-1)
    min_xy = np.min(np.min(lane, axis=-1), axis=-1)
    # max_xy[max_xy>20] = 20
    # min_xy[min_xy<-20] = -20
    mean_xy = (max_xy + min_xy) / 2
    scale_val =  max(np.max(np.abs(max_xy)), np.max(np.abs(min_xy))) / 200
    offset = 255 - ((mean_xy - min_xy).reshape((2,1)) / scale_val)
    min_xy = min_xy.reshape((2,1))
    zero_pos = (- min_xy / scale_val + offset).astype("int")
    # a = att[idx, i::history_frame_num].copy() # T, T'*V'
    # a = np.mean(a, axis=0)
    # a = a.reshape((lane.shape[1], lane.shape[2])) # T', V'
    # sum_a = np.sum(a, axis=-1)
    argmax_a = np.argmax(a, axis=-1)
    img = np.ones((512,512,3)) * 255
    for ii in range(lane.shape[-1]):
        segment = (lane[:,:,ii] - min_xy) / scale_val + offset
        segment = segment.astype("int")
        for jj in range(segment.shape[-1] - 1):
            color = (0,0,0) 
            thickness = 1
            if abs(a[ii] - a[argmax_a]) < 1e-6:
                color = (0,255,0)
                thickness = 2   
            p1 = segment[:,jj]
            p2 = segment[:, jj + 1]
            if (p1[0] != zero_pos[0] or p1[1] != zero_pos[1]) and (p2[0] != zero_pos[0] or p2[1] != zero_pos[1]):
                img = cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), color, thickness)

    tar = (tar - min_xy) / scale_val + offset
    tar = tar.astype("int")
    zero_pos = zero_pos.astype("int")
    # out *= 5 
    # tar *= 5
    # out = out.astype("int") + 255 
    # tar = tar.astype("int") + 255
    for j in range(tar.shape[-1]):
        if (tar[0,j] != zero_pos[0] or tar[1,j] != zero_pos[1]):
            img = cv2.circle(img, (tar[0,j], tar[1,j]), 3, (255,0,0), thickness=2)
    cv2.imshow("img_grip", img)
    cv2.waitKey(1)


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
        lines = reader.readlines()
        if config.map_type == "argo":
            city = lines[0].strip().split(' ')[-1]
        content = np.array([line.strip().split(' ')[:total_feature_dimension - 1] for line in lines]).astype(float)
        now_dict = {}
        for row in content:
            # instance = {row[1]:row[2:]}
            n_dict = now_dict.get(row[0], {})
            n_dict[row[1]] = row#[2:]
            # n_dict.append(instance)
            # now_dict[]
            now_dict[row[0]] = n_dict
    return now_dict, city


def get_current_lane(trajectory, centerlines):
    centerlines = np.array(centerlines[:, :, 3:5])
    min_dist = float("inf")
    for i in range(centerlines.shape[0]):
        curr_dist = 0
        segment = centerlines[i]
        for j in range(trajectory.shape[0]):
            traj = trajectory[j:j+1]
            curr_dist += np.min(np.sqrt(np.sum((segment - traj) ** 2, axis=1)))
        if curr_dist < min_dist:
            curr_lane_id = i
            min_dist = curr_dist

    return curr_lane_id


def process_trajectory_data(id, city, now_traj_sl:list, trajectory_sl_list:list, trajectory:np.ndarray, accumulated_s=0, deepth=0):
    lane = avm.city_lane_centerlines_dict[city][id]
    segment = lane.centerline
    lane_string = LineString(segment)
    lane_length = lane_string.length
    lane_id = lane.id
    turn = lane.turn_direction.lower()
    if turn == 'right':
        turn = 1
    elif turn == 'left':
        turn = 2
    else:
        turn = 0
    ori_idx = len(now_traj_sl)
    s = 0
    l = 0
    idx = ori_idx
    while idx < trajectory.shape[0]:
        if trajectory[idx][-1] == 0:
            now_traj_sl.append([0,0,0,0,0])
            idx += 1 
            continue
        s, l = centerline_utils.get_normal_and_tangential_distance_point(trajectory[idx][0], trajectory[idx][1], segment) 
        p = lane_string.interpolate(s);
        if s < lane_length:# and s > 0
            now_traj_sl.append([lane_id, s + accumulated_s, l, turn, 1])
            idx += 1
        else:
            break
    if s == 0 and (idx > 1 and now_traj_sl[-1][1] <= now_traj_sl[-2][1]):
        now_traj_sl = now_traj_sl[:ori_idx]
        return
    accumulated_s += lane_length
    if idx >= trajectory.shape[0]:
        tmp = np.array(now_traj_sl, dtype="float")
        if len(trajectory_sl_list) == 0 or not np.any([np.array_equal(tmp[:,0], traj[:,0]) for traj in trajectory_sl_list]):
            trajectory_sl_list.append(tmp)
    else:
        if not lane.successors is None and deepth < 100:
            deepth += 1
            # cur_idx = len(now_traj_sl)
            for s_id in lane.successors:
                # now_traj_sl = now_traj_sl[0:cur_idx]
                process_trajectory_data(s_id, city, now_traj_sl, trajectory_sl_list, trajectory, accumulated_s, deepth)
        else:
            while len(now_traj_sl) < trajectory.shape[0]:
                now_traj_sl.append([0,0,0,0,0])
            tmp = np.array(now_traj_sl, dtype="float")
            if len(trajectory_sl_list) == 0 or not np.any([np.array_equal(tmp[:,0], traj[:,0]) for traj in trajectory_sl_list]):
                trajectory_sl_list.append(tmp)
    accumulated_s -= lane_length
    # tmp = now_traj_sl[0:ori_idx]
    for l in range(ori_idx, len(now_traj_sl)):
        now_traj_sl.pop()
    # now_traj_sl = now_traj_sl[0:ori_idx]
    return


def process_lane_data(traj_sl:np.ndarray, mean_xy, city, lane_num):
    lane_ids = set([t[0] for t in traj_sl if t[0] != 0])
    lane_ids = []
    for t in traj_sl:
        if t[0] != 0 and t[0] not in lane_ids:
            lane_ids.append(t[0])
    now_map_feature = []
    last_frame = np.zeros((total_feature_dimension))
    for lane_id in lane_ids:
        lane = avm.city_lane_centerlines_dict[city][lane_id]
        turn = lane.turn_direction.lower()
        if turn == 'right':
            turn = 1
        elif turn == 'left':
            turn = 2
        else:
            turn = 0
        segment = lane.centerline
        for i in range(segment.shape[0]):
            curr_frame = np.array([lane_num, lane_id, 0, segment[i][0] - mean_xy[0], segment[i][1] - mean_xy[1], 0, 0, 0, 0, turn, 1], dtype="float")
            last_frame = curr_frame
            now_map_feature.append(curr_frame)  
            if len(now_map_feature) >= config.segment_point_num:
                return np.array(now_map_feature, dtype="float")
    for i in range(len(now_map_feature), config.segment_point_num):
        now_map_feature.append(last_frame)
    return np.array(now_map_feature, dtype="float")



def process_map_data(center, trajectory, search_radius, point_num, mean_xy, city=""):
    global map_num
    map_feature_list = []
    trajectory_sl_list = []
    idList = []
    if config.map_type == "argo":
        if config.only_nearby_lanes:
            idList = []
            while not idList:
                idList = avm.get_lane_ids_in_xy_bbox(center[0], center[1], city, 100)
                search_radius *= 2
            # _ = avm.get_nearest_centerline(center, city, visualize=True)
            idList = np.array(idList, dtype="int")
            nearby_lane_objs = [avm.city_lane_centerlines_dict[city][lane_id] for lane_id in idList]
            per_lane_dists, min_dist_nn_indices, _ = centerline_utils.lane_waypt_to_query_dist(center, nearby_lane_objs)

            curr_lane_ids = idList[per_lane_dists < 2]
            nearest_lane = idList[min_dist_nn_indices[0]]
            if nearest_lane not in curr_lane_ids:
                curr_lane_ids = np.append(curr_lane_ids, nearest_lane)
            idList = []
            for lane_id in curr_lane_ids:
                if lane_id not in idList:
                    idList.append(lane_id)
                curr_lane = avm.city_lane_centerlines_dict[city][lane_id]
                if not curr_lane.l_neighbor_id is None and curr_lane.l_neighbor_id not in idList:
                    idList.append(curr_lane.l_neighbor_id)
                if not curr_lane.r_neighbor_id is None and curr_lane.r_neighbor_id not in idList:
                    idList.append(curr_lane.r_neighbor_id)
                # if not curr_lane.predecessors is None:
                #     for p_id in curr_lane.predecessors:
                #         if p_id not in idList:
                #             idList.append(p_id)
                # if not curr_lane.successors is None:
                #     for s_id in curr_lane.successors:
                #         if s_id not in idList:
                #             idList.append(s_id)
        else:
            idList = np.array(avm.get_lane_ids_in_xy_bbox(mean_xy[0], mean_xy[1], city, search_radius), dtype="int")
        # print(idList)
        for ln, id in enumerate(idList):
            # print(id)
            now_traj_sl = []
            process_trajectory_data(id, city, now_traj_sl, trajectory_sl_list, trajectory)

        trajectory_sl_list = np.array(trajectory_sl_list, dtype="float")

        for lane_num, traj_sl in enumerate(trajectory_sl_list):
            now_map_feature = process_lane_data(traj_sl, mean_xy, city, lane_num)
            map_feature_list.append(now_map_feature)

        map_feature_list = np.array(map_feature_list, dtype="float")

    # else:
    #     lane_list = hdmap.get_lanes(center, search_radius, point_num)
    #     for ln, lane in enumerate(lane_list):
    #         lane_id = lane.id
    #         for segment in lane.segment:
    #             now_map_feature = np.zeros((point_num, total_feature_dimension))
    #             segment = np.array(segment).astype("float")
    #             for i in range(segment.shape[0]):
    #                 now_map_feature[i] = np.array([ln, 0, 0, segment[i][0] - mean_xy[0], segment[i][1] - mean_xy[1], 0, 0, 0, 0, lane.turn, 1], dtype="float")
    #             map_feature_list.append(now_map_feature)

    # map_feature_list = np.array(map_feature_list, dtype="float")
    # trajectory_sl_list = np.array(trajectory_sl_list, dtype="float")
    if map_feature_list.shape[0] > map_num:
        map_num = map_feature_list.shape[0]
        # print("\nmap_num: %d"%map_num)
    return map_feature_list, trajectory_sl_list



def process_data(pra_now_dict, pra_start_ind, pra_end_ind, pra_observed_last, frame_id_set, city=""):
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

    now_all_object_id = set([val for x in range(pra_start_ind, pra_end_ind, frame_steps) if x in frame_id_set for val in pra_now_dict[x].keys() ])
    non_visible_object_id_list = list(now_all_object_id - set(visible_object_id_list))
    num_non_visible_object = len(non_visible_object_id_list)
    
    # for all history frames(6) or future frames(6), we only choose the objects listed in visible_object_id_list
    object_feature_list = []
    # non_visible_object_feature_list = []
    # empty_frame_feature = np.array([np.zeros(total_feature_dimension) for obj_id in visible_object_id_list+non_visible_object_id_list])
    for frame_ind in range(pra_start_ind, pra_end_ind, frame_steps):
        # if frame_ind not in frame_id_set:    
        #     object_feature_list.append(empty_frame_feature)
        #     continue
        # we add mark "1" to the end of each row to indicate that this row exists, using list(pra_now_dict[frame_ind][obj_id])+[1] 
        # -mean_xy is used to zero_centralize data
        # now_frame_feature_dict = {obj_id : list(pra_now_dict[frame_ind][obj_id]-mean_xy)+[1] for obj_id in pra_now_dict[frame_ind] if obj_id in visible_object_id_list}
        now_frame_feature_dict = {obj_id : (list(pra_now_dict[frame_ind][obj_id]-mean_xy)+[1] if obj_id in visible_object_id_list else list(pra_now_dict[frame_ind][obj_id]-mean_xy)+[0]) for obj_id in pra_now_dict[frame_ind] }
        # if the current object is not at this frame, we return all 0s by using dict.get(_, np.zeros(11))
        now_frame_feature = np.array([now_frame_feature_dict.get(vis_id, np.zeros(total_feature_dimension)) for vis_id in visible_object_id_list+non_visible_object_id_list])
        object_feature_list.append(now_frame_feature)

    # object_feature_list has shape of (frame#, object#, 11) 11 = 10features + 1mark
    object_feature_list = np.array(object_feature_list)

    ego_position = object_feature_list[0, 0, 3:5] + m_xy
    ego_trajectory = object_feature_list[:, 0, 3:].copy()
    ego_trajectory[:, :2] += m_xy

    map_feature_list, trajectory_sl_list = process_map_data(ego_position, ego_trajectory, config.lane_search_radius, config.segment_point_num, m_xy, city)
    for map_feat in map_feature_list:
        if np.array_equal(map_feat[0,3:5], map_feat[-1,3:5]):
            print(map_feat)
    # if curr_lane_id >= 10:
    #     print(curr_lane_id)
    # object feature with a shape of (frame#, object#, 11) -> (object#, frame#, 11)

    object_frame_feature = np.zeros((max_num_object, round((pra_end_ind-pra_start_ind) / frame_steps), total_feature_dimension))
    map_frame_feature = np.zeros((max_num_map, config.segment_point_num, total_feature_dimension))
    trajectory_frame_feature = np.zeros((max_num_map, total_frames, 5))
    curr_lane_label = np.zeros(max_num_map)
    try:
        mean_lane_l = np.mean(np.abs(trajectory_sl_list[:, :, 2]), axis=1)
        sorted_lane_idx = np.argsort(mean_lane_l, axis=0)
        map_feature_list = map_feature_list[sorted_lane_idx]
        trajectory_sl_list = trajectory_sl_list[sorted_lane_idx]
        curr_lane_label[0] = 1
        ego_trajectory[:, :2] -= m_xy
        # visulize(np.transpose(ego_trajectory[:,:2], (1,0)), np.transpose(map_feature_list[:,:,3:5], (2,1,0)), curr_lane_label)    
        map_frame_feature[:map_feature_list.shape[0]] = map_feature_list
        trajectory_frame_feature[:trajectory_sl_list.shape[0]] = trajectory_sl_list    
    except:
        aaaa = 1
    
    # np.transpose(object_feature_list, (1,0,2))
    object_frame_feature[:num_visible_object+num_non_visible_object] = np.transpose(object_feature_list, (1,0,2))
    
    return object_frame_feature, neighbor_matrix, m_xy, map_frame_feature, curr_lane_label, trajectory_frame_feature
    

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
    seq_id = int(pra_file_path.split("/")[-1].split(".")[-2].split("_")[-1])
    frame_id_set = sorted(set(now_dict.keys()))
    # print(pra_file_path)
    all_feature_list = []
    all_adjacency_list = []
    all_mean_list = []
    all_map_list = []
    all_lane_list = []
    all_trajectory_list = []
    for start_ind in frame_id_set[:-total_frames+1:frame_steps]:
        start_ind = int(start_ind)
        end_ind = int(start_ind + total_frames * frame_steps)
        observed_last = start_ind + (history_frames - 1)*frame_steps
        for ind in range(start_ind, end_ind, frame_steps):
            if ind not in frame_id_set:
                break
        else:
            if observed_last not in frame_id_set:
                continue;
            object_frame_feature, neighbor_matrix, mean_xy, map_frame_feature, curr_lane_label, trajectory_frame_feature \
                 = process_data(now_dict, start_ind, end_ind, observed_last, frame_id_set, city)

            all_feature_list.append(object_frame_feature)
            all_adjacency_list.append(neighbor_matrix)    
            all_mean_list.append(mean_xy)  
            all_map_list.append(map_frame_feature) 
            all_lane_list.append(curr_lane_label) 
            all_trajectory_list.append(trajectory_frame_feature)

    # (N, V, T, C) --> (N, C, T, V)
    all_feature_list = np.array(all_feature_list)
    all_adjacency_list = np.array(all_adjacency_list)
    all_mean_list = np.array(all_mean_list)
    all_map_list = np.array(all_map_list)
    all_lane_list = np.array(all_lane_list)
    all_trajectory_list = np.array(all_trajectory_list)
    try:
        all_feature_list = np.transpose(all_feature_list, (0, 3, 2, 1))
        all_map_list = np.transpose(all_map_list, (0, 3, 2, 1))
        all_trajectory_list = np.transpose(all_trajectory_list, (0, 3, 2, 1))
    except:
        print(all_feature_list.shape, all_map_list.shape, all_trajectory_list.shape)
    # print(all_feature_list.shape, all_adjacency_list.shape)
    return all_feature_list, all_adjacency_list, all_mean_list, all_map_list, all_lane_list, all_trajectory_list, seq_id, city


def generate_test_data(pra_file_path):
    now_dict, city = get_frame_instance_dict(pra_file_path)
    frame_id_set = sorted(set(now_dict.keys()))
    seq_id = int(pra_file_path.split("/")[-1].split(".")[-2].split("_")[-1])
    all_feature_list = []
    all_adjacency_list = []
    all_mean_list = []
    all_map_list = []
    all_lane_list = []
    all_trajectory_list = []

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
            object_frame_feature, neighbor_matrix, mean_xy, map_frame_feature, curr_lane_label, trajectory_frame_feature \
                 = process_data(now_dict, start_ind, end_ind, observed_last, frame_id_set, city)

            all_feature_list.append(object_frame_feature)
            all_adjacency_list.append(neighbor_matrix)    
            all_mean_list.append(mean_xy)  
            all_map_list.append(map_frame_feature) 
            all_lane_list.append(curr_lane_label) 
            all_trajectory_list.append(trajectory_frame_feature)

    # (N, V, T, C) --> (N, C, T, V)
    all_feature_list = np.array(all_feature_list)
    all_adjacency_list = np.array(all_adjacency_list)
    all_mean_list = np.array(all_mean_list)
    all_map_list = np.array(all_map_list)
    all_lane_list = np.array(all_lane_list)
    all_trajectory_list = np.array(all_trajectory_list)
    if all_trajectory_list.shape[0] > 0:
        all_feature_list = np.transpose(all_feature_list, (0, 3, 2, 1))
        all_map_list = np.transpose(all_map_list, (0, 3, 2, 1))
        all_trajectory_list = np.transpose(all_trajectory_list, (0, 3, 2, 1))
    # print(all_feature_list.shape, all_adjacency_list.shape)
    return all_feature_list, all_adjacency_list, all_mean_list, all_map_list, all_lane_list, all_trajectory_list, seq_id, city


def generate_data(pra_file_path_list, pra_is_train=True, save_data=True, idx=-1):
    all_data = []
    all_adjacency = []
    all_mean_xy = []
    all_map_data = []
    all_lane_label = []
    all_trajetory = []
    all_seq_id_city = []
    # for file_path in pra_file_path_list:
    for file_path in tqdm(pra_file_path_list):
        # print(file_path)
        if pra_is_train == "train" or pra_is_train == "val":
            now_data, now_adjacency, now_mean_xy, now_map_data, now_lane_label, now_trajectory, seq_id, city = generate_train_data(file_path)
        else:
            now_data, now_adjacency, now_mean_xy, now_map_data, now_lane_label, now_trajectory, seq_id, city = generate_test_data(file_path)
        if now_trajectory.shape[0] > 0:
            all_data.extend(now_data)
            all_adjacency.extend(now_adjacency)
            all_mean_xy.extend(now_mean_xy)
            all_map_data.extend(now_map_data)
            all_lane_label.extend(now_lane_label)
            all_trajetory.extend(now_trajectory)
            all_seq_id_city.append([seq_id, city])

    all_data = np.array(all_data) #(N, C, T, V)=(5010, 11, 12, 70) Train
    all_adjacency = np.array(all_adjacency) #(5010, 70, 70) Train
    all_mean_xy = np.array(all_mean_xy) #(5010, 2) Train
    all_map_data = np.array(all_map_data)
    all_lane_label = np.array(all_lane_label)
    all_trajetory = np.array(all_trajetory)
    all_seq_id_city = np.array(all_seq_id_city)

    # Train (N, C, T, V)=(5010, 11, 12, 70), (5010, 70, 70), (5010, 2)
    # Test (N, C, T, V)=(415, 11, 6, 70), (415, 70, 70), (415, 2)

    # save training_data and trainjing_adjacency into a file.
    if save_data:
        print(np.shape(all_data), np.shape(all_adjacency), np.shape(all_mean_xy), np.shape(all_map_data), np.shape(all_lane_label))
        save_path = ""
        if pra_is_train == "train":
            save_path = os.path.join(data_root, train_data_path, "%02d_sl_"%idx + config.train_data_file)
        elif pra_is_train == "val":
            save_path = os.path.join(data_root, val_data_path, "%02d_sl_"%idx + config.val_data_file)
        elif pra_is_train == "test":
            save_path = os.path.join(data_root, test_data_path, "%02d_sl_"%idx + config.test_data_file)

        print("saving %s data: [%s]"%(pra_is_train, save_path))


        with open(save_path, 'wb') as writer:
            pickle.dump([all_data, all_adjacency, all_mean_xy, all_map_data, all_lane_label, all_trajetory, all_seq_id_city], writer)

    # return all_data, all_adjacency, all_mean_xy, all_map_data, all_lane_label, all_trajetory


if __name__ == '__main__':
    train_data_path = config.train_data_path
    test_data_path  = config.test_data_path
    val_data_path   = config.val_data_path
    # train_data_path = "prediction_train/"
    train_data_path_list = sorted(glob.glob(os.path.join(data_root, train_data_path + "/raw_data", '*.txt')), key=lambda x: int(x.split(".")[-2].split("_")[-1]))
    val_data_path_list   = sorted(glob.glob(os.path.join(data_root, val_data_path   + "/raw_data", '*.txt')), key=lambda x: int(x.split(".")[-2].split("_")[-1]))
    # test_data_path_list  = sorted(glob.glob(os.path.join(data_root, test_data_path  + "/raw_data", '*.txt')), key=lambda x: int(x.split(".")[-2].split("_")[-1]))
    train_data_size = 10000
    val_data_size   = 1000
    test_data_size  = 5000

    train_data_length = len(train_data_path_list) // train_data_size
    if len(train_data_path_list) % train_data_size != 0:
        train_data_length += 1 
    for idx in range(0,train_data_length):#
        print('Generating Training Data_%02d/%02d'%(idx+1, train_data_length))
        start_index = idx * train_data_size
        end_index = min((idx + 1) * train_data_size, len(train_data_path_list))
        generate_data(train_data_path_list[start_index:end_index], pra_is_train="train", idx=idx)

    val_data_length = len(val_data_path_list) // val_data_size
    if len(val_data_path_list) % val_data_size != 0:
        val_data_length += 1 
    for idx in range(val_data_length):#
        print('Generating Validate Data_%02d/%02d'%(idx+1, val_data_length))
        start_index = idx * val_data_size
        end_index = min((idx + 1) * val_data_size, len(val_data_path_list))
        generate_data(val_data_path_list[start_index:end_index], pra_is_train="val", idx=idx)

    # test_data_length = len(test_data_path_list) // test_data_size
    # if len(test_data_path_list) % test_data_size != 0:
    #     test_data_length += 1 
    # for idx in range(test_data_length):#
    #     print('Generating Testing Data_%02d/%02d'%(idx+1, test_data_length))
    #     start_index = idx * test_data_size
    #     end_index = min((idx + 1) * test_data_size, len(test_data_path_list))
    #     generate_data(test_data_path_list[start_index:end_index], pra_is_train="test", idx=idx)
  


