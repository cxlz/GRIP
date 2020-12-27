import argparse
import os 
import sys
import numpy as np 
import cv2
import torch
import torch.optim as optim
from torchnet import meter
from model import Model
from xin_feeder_baidu import Feeder
import config.configure as config
from datetime import datetime
import random
import itertools
from tqdm import tqdm
import time
from types import ModuleType
# from util.map_util import HDMap

from argoverse.evaluation.competition_util import generate_forecasting_h5
from argoverse.evaluation import eval_forecasting
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils import centerline_utils
from shapely.geometry import LineString
from argoverse.map_representation.lane_segment import LaneSegment

import config.configure as config
# from util.map_util import HDMap, LaneSegment

if config.map_type == "argo":
    print("loading ArgoverseMap")
    avm = ArgoverseMap()

CUDA_VISIBLE_DEVICES='0'
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

# def seed_torch(seed=0):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
# seed_torch()

def my_print(pra_content):
    with open(log_file, 'a') as writer:
        print(pra_content)
        writer.write(pra_content+'\n')
max_x = 1. 
max_y = 1. 
history_frames = config.history_frames # 3 second * 2 frame/second
future_frames = config.future_frames # 3 second * 2 frame/second

batch_size_train = config.batch_size_train
batch_size_val = config.batch_size_val
batch_size_test = config.batch_size_test
total_epoch = config.total_epoch
base_lr = config.base_lr
lr_decay_epoch = config.lr_decay_epoch
lr_decay = config.lr_decay


dev = config.dev 
use_cuda = config.use_cuda 
print(use_cuda, dev)
# dev = 'cuda:0' 
work_dir = config.work_dir
log_file = config.log_file
test_result_file = config.test_result_file

criterion = torch.nn.SmoothL1Loss()

if not os.path.exists(work_dir):
    os.makedirs(work_dir)


def display_result(pra_results, pra_pref='Train_epoch', pra_error_order=2):
    all_overall_sum_list, all_overall_num_list = pra_results
    overall_sum_time = np.sum(all_overall_sum_list**(1/pra_error_order), axis=0)
    overall_num_time = np.sum(all_overall_num_list, axis=0)
    overall_loss_time = (overall_sum_time / overall_num_time) 
    # idx = all_overall_num_list != 0
    # overall_loss_time = np.mean(all_overall_sum_list[idx] ** (1/ pra_error_order) / all_overall_num_list[idx], axis=0)
    overall_log = '|{}|[{}] All_All: {}'.format(datetime.now(), pra_pref, ' '.join(['{:.3f}'.format(x) for x in list(overall_loss_time) + [np.mean(overall_loss_time)]]))
    my_print(overall_log)
    return overall_loss_time

count = 0
def visulize(outputs, targets, output_mask, lanes=None, att=None, label=None):
    global count
    # targets: N C T V
    # lanes: N' C' T' V'
    # att: N T*V, T'*V
    history_frame_num = targets.shape[2] - outputs.shape[2]
    idx = 0
    for outs, tars, mask in zip(outputs, targets, output_mask):
        for i in range(outs.shape[-1]):
            img = np.ones((512,512,3)) * 255
            out = outs[:,:,i].copy()
            tar = tars[-2:,:,0].copy()
            if mask[0, 0, i] == 1:
                if not lanes is None:
                    lane = lanes[idx].copy() # C' T' V'
                    max_xy = np.max(np.max(lane, axis=-1), axis=-1)
                    min_xy = np.min(np.min(lane, axis=-1), axis=-1)
                    # max_xy[max_xy>20] = 20
                    # min_xy[min_xy<-20] = -20
                    mean_xy = (max_xy + min_xy) / 2
                    scale_val =  max(np.max(np.abs(max_xy)), np.max(np.abs(min_xy))) / 200
                    if scale_val == 0:
                        continue 
                    offset = 255 - ((mean_xy - min_xy).reshape((2,1)) / scale_val)
                    min_xy = min_xy.reshape((2,1))
                    zero_pos = (- min_xy / scale_val + offset).astype("int")
                    # a = att[idx, i::history_frame_num].copy() # T, T'*V'
                    # a = np.mean(a, axis=0)
                    # a = a.reshape((lane.shape[1], lane.shape[2])) # T', V'
                    # sum_a = np.sum(a, axis=-1)
                    a = att[idx]
                    argmax_a = np.argmax(a, axis=-1)
                    for ii in range(lane.shape[-1]):
                        # if a[ii] == 0:
                        #     continue
                        segment = (lane[:,:,ii] - min_xy) / scale_val + offset
                        segment = segment.astype("int")
                        for jj in range(segment.shape[-1] - 1):
                            color = (0,0,0) 
                            thickness = 1
                            if abs(a[ii] - a[argmax_a]) < 1e-6: #config.use_map and 
                                color = (0,255,0)
                            if label[0] == ii:
                                thickness = 2   
                            p1 = segment[:,jj]
                            p2 = segment[:, jj + 1]
                            if (p1[0] != zero_pos[0] or p1[1] != zero_pos[1]) and (p2[0] != zero_pos[0] or p2[1] != zero_pos[1]):
                                img = cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), color, thickness)
                            # if (a[ii] > 0.2 or abs(a[ii] - a[argmax_a]) < 1e-6) and jj == segment.shape[-1] * ii // lane.shape[-1]:
                            if ii == i and jj == 0:
                                img = cv2.putText(img, "%.2f"%a[ii], (p1[0], p1[1]), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0,0,0), 1)
                        # cv2.imshow("img_grip", img)
                        # cv2.waitKey(1)
                else:
                    max_xy = tar[:, 0:1]
                    min_xy = tar[:, history_frame_num-1:history_frame_num]
                    maxmin_x = abs(float((max_xy[0] - min_xy[0])))
                    maxmin_y = abs(float((max_xy[1] - min_xy[1])))
                    scale_val = max(maxmin_x, maxmin_y) / 100
                    offset = 255
                    if scale_val == 0:
                        continue 
                x2y2= np.sum((out - tar[:, history_frame_num:]) ** 2, axis=0)**0.5
                last_speed = np.sum((tar[:, history_frame_num] - tar[:, history_frame_num - 1]) ** 2) ** 0.5 / (config.frame_steps * config.data_time_resolution)
                out = (out - min_xy) / scale_val + offset 
                tar = (tar - min_xy) / scale_val + offset
                zero_pos = - min_xy / scale_val + offset
                out = out.astype("int")
                tar = tar.astype("int")
                zero_pos = zero_pos.astype("int")
                # out *= 5 
                # tar *= 5
                # out = out.astype("int") + 255 
                # tar = tar.astype("int") + 255
                for j in range(history_frame_num - 1):
                    if (tar[0,j] != zero_pos[0] or tar[1,j] != zero_pos[1]) and (tar[0,j + 1] != zero_pos[0] or tar[1,j + 1] != zero_pos[1]):
                        img = cv2.line(img, (tar[0,j], tar[1,j]), (tar[0,j + 1], tar[1,j + 1]), (0,0,255), thickness=2)
                for j in range(history_frame_num, tar.shape[1]):
                    try:
                        img = cv2.circle(img, (out[0,j - history_frame_num], out[1,j - history_frame_num]), 3, (255,0,0), thickness=2)
                        img = cv2.circle(img, (tar[0,j], tar[1,j]), 3, (0,0,255), thickness=-1)
                    except:
                        continue


                obs_info = "obs[{:0>5d}]_frame[{:0>5d}]_type[{:d}]_[{:0>4d}].jpg".format(*(tars[(1,0,2),history_frame_num,0].astype("int")), count)
                img = cv2.putText(img, obs_info, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), thickness=1)
                img = cv2.putText(img, "speed[{:.3f}]".format(last_speed),    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), thickness=1)
                img = cv2.putText(img, "FDE  [{:.3f}]".format(x2y2[-1]),      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), thickness=1)
                img = cv2.putText(img, "ADE  [{:.3f}]".format(np.mean(x2y2)), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), thickness=1)
                img = cv2.putText(img, "range[{:.3f}]".format(abs(scale_val * 512)), (360,480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), thickness=1)
                cv2.imshow("img_grip", img)
                cv2.waitKey(1)
                if not config.train and config.save_view:
                    save_path = os.path.join(config.save_view_path, obs_info)
                    cv2.imwrite(save_path, img)
            else:
                continue

        idx +=1
    count += 1

def my_save_model(pra_model, pra_epoch):
    model_time = time.strftime('%m%d_%H:%M:%S')
    path = os.path.join(work_dir, config.save_model_prefix + '{}_epoch_{:04}.pt'.format(model_time, pra_epoch))
    torch.save(
        {
            'xin_graph_seq2seq_model': pra_model.state_dict(),
        }, 
        path)
    my_print('Successfull saved to {}'.format(path))


def my_load_model(pra_model, pra_path):
    checkpoint = torch.load(pra_path, map_location=config.dev)
    pra_model.load_state_dict(checkpoint['xin_graph_seq2seq_model'])
    my_print('Successfull loaded from {}'.format(pra_path))
    return pra_model


def data_loader(pra_path, pra_batch_size=128, pra_shuffle=False, pra_drop_last=False, train_val_test='train'):
    feeder = Feeder(data_path=pra_path, graph_args=graph_args, train_val_test=train_val_test)
    # feeder = new_Feeder(pra_path, graph_args=graph_args, train_val_test=train_val_test)
    loader = torch.utils.data.DataLoader(
        dataset=feeder,
        batch_size=pra_batch_size,
        shuffle=pra_shuffle,
        drop_last=pra_drop_last, 
        num_workers=0,
        # collate_fn=my_collate_fn,
        )
    return loader
    
def preprocess_data(pra_data, pra_rescale_xy, feature_id=[3, 4, -2, -1]):
    # pra_data: (N, C, T, V)
    # C = 11: [frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading] + [mask]    
    
    ori_data = pra_data[:,feature_id].detach()
    data = ori_data.detach().clone()

    if config.vel_mode:
        new_mask = (data[:, :2, 1:]!=0) * (data[:, :2, :-1]!=0) 
        data[:, :2, 1:] = (data[:, :2, 1:] - data[:, :2, :-1]).float() * new_mask.float()
        data[:, :2, 0] = 0    

    # # small vehicle: 1, big vehicles: 2, pedestrian 3, bicycle: 4, others: 5
    object_type = pra_data[:,2:3]

    data = data.float().to(dev)
    ori_data = ori_data.float().to(dev)
    object_type = object_type.to(dev) #type
    data[:,:2] = data[:,:2] / pra_rescale_xy

    return data, ori_data, object_type
    

def compute_RMSE(pra_pred, pra_GT, pra_mask, pra_error_order=2):
    pred = pra_pred * pra_mask # (N, C, T, V)=(N, 2, 6, 120)
    GT = pra_GT * pra_mask # (N, C, T, V)=(N, 2, 6, 120)
    
    x2y2 = torch.sum(torch.abs(pred - GT)**pra_error_order, dim=1) # x^2+y^2, (N, C, T, V)->(N, T, V)=(N, 6, 120)
    overall_sum_time = x2y2.sum(dim=-1) # (N, T, V) -> (N, T)=(N, 6)
    overall_mask = pra_mask.sum(dim=1).sum(dim=-1) # (N, C, T, V) -> (N, T)=(N, 6)
    overall_num = overall_mask 

    return overall_sum_time, overall_num, x2y2

def compute_HuberLoss(pra_pred, pra_GT, pra_mask, delta=3):
    pred = pra_pred * pra_mask # (N, C, T, V)=(N, 2, 6, 120)
    GT = pra_GT * pra_mask # (N, C, T, V)=(N, 2, 6, 120)
    
    xy = torch.abs(pred - GT) # 

    large_x2y2 = xy ** 2 * 0.5 # x^2+y^2, (N, C, T, V)->(N, T, V)=(N, 6, 120)
    small_x2y2 = xy * delta - 0.5 * delta ** 2
    cond = xy < delta
    x2y2 = torch.sum(torch.where(cond, large_x2y2, small_x2y2), dim=1)
    overall_sum_time = x2y2.sum(dim=-1) # (N, T, V) -> (N, T)=(N, 6)
    overall_mask = pra_mask.sum(dim=1).sum(dim=-1) # (N, C, T, V) -> (N, T)=(N, 6)
    overall_num = overall_mask 

    return overall_sum_time, overall_num, x2y2


def point_from_lane(lane_id, s, l, city):
    try:
        lane = avm.city_lane_centerlines_dict[city][lane_id]
    except:
        print(city, lane_id)
        return 0, 0
    segment = lane.centerline
    lane_string = LineString(segment)
    curr_point = lane_string.interpolate(s).bounds
    n_point = curr_point
    if (s + 0.3) < lane_string.length:
        n_point = lane_string.interpolate(s + 0.3).bounds
        heading = np.arctan2((n_point[1] - curr_point[1]), (n_point[0] - curr_point[0]))
    else:
        n_point = lane_string.interpolate(max(s - 0.3, 0)).bounds
        heading = np.arctan2((curr_point[1] - n_point[1]), (curr_point[0] - n_point[0]))
    x = curr_point[0] - l * np.sin(heading)
    y = curr_point[1] + l * np.cos(heading)
    return x, y

def get_lane_length(lane_id, city) -> (LaneSegment, float):
    try:
        lane = avm.city_lane_centerlines_dict[city][lane_id]
    except:
        print(city, lane_id)
        return 0
    segment = lane.centerline
    lane_string = LineString(segment)
    lane_length = lane_string.length
    return lane, lane_length

def get_xy_trajectory(predicted, now_mean_xy, seq_id_city, now_lane_id, now_history_frames, future_lane_id=[]):
    """ 
    predicted:   [C=2, T, V]
    now_mean_xy: [2]
    seq_id_city: [2]
    now_lane_id: [T, V]
    """
    now_pred = torch.zeros(predicted.shape).to(config.dev)
    predicted = predicted[:,:,0].float().detach().cpu().numpy()
    mean_xy = now_mean_xy.float().detach().cpu().numpy() 
    curr_seq_id_city = seq_id_city
    city = curr_seq_id_city[1].long().item()

    lane_id = now_lane_id[:,0].long().detach().cpu().numpy()
    curr_lane_id = lane_id[now_history_frames - 1]
    if curr_lane_id == 0:
        return now_pred
    _, uni_idx = np.unique(lane_id, return_index=True)
    lane_id = lane_id[sorted(uni_idx)]
    if future_lane_id:
        lane_id = np.concatenate((lane_id, future_lane_id), axis=0)
    city = "PIT" if city == 0 else "MIA"
    l_idx = 0
    lane_length = 0 
    accumulated_s = 0
    for l_idx in range(len(lane_id)):
        if lane_id[l_idx] != 0:
            lane, lane_length = get_lane_length(lane_id[l_idx], city)
            if lane_id[l_idx] == curr_lane_id:
                break
            accumulated_s += lane_length
    
    for t in range(predicted.shape[1]):
        s = predicted[0, t] - accumulated_s
        l = predicted[1, t]
        if s > lane_length:
            if l_idx + 1 < len(lane_id) and lane_id[l_idx + 1] != 0:
                l_idx += 1
                curr_lane_id = lane_id[l_idx]
                accumulated_s += lane_length
                s -= lane_length
                lane, lane_length = get_lane_length(curr_lane_id, city)
            elif not lane.successors is None:
                curr_lane_id = lane.successors[0]
                accumulated_s += lane_length
                s -= lane_length
                lane, lane_length = get_lane_length(curr_lane_id, city)

        x, y = point_from_lane(curr_lane_id, s, l, city)
        if x != 0 or y != 0:
            now_pred[0,t,0] = x - mean_xy[0]
            now_pred[1,t,0] = y - mean_xy[1]    
    return now_pred

def get_future_lane_id(now_lane_ids, curr_lane_id, city, future_s, future_lane_ids):
    curr_lane = avm.city_lane_centerlines_dict[city][curr_lane_id]
    if not curr_lane.successors is None:
        for s_id in curr_lane.successors:
            s_lane, s_lane_length = get_lane_length(s_id, city)
            now_lane_ids.append(s_id)
            if s_lane_length >= future_s:
                future_lane_ids.append(now_lane_ids.copy())
            else:
                get_future_lane_id(now_lane_ids, s_id, city, future_s - s_lane_length, future_lane_ids)
            now_lane_ids.pop()


def get_lane_id(predicted, ori_lane_id, seq_id_city):
    overall_lane_ids = []
    for n in range(predicted.shape[0]):
        now_pred_s = predicted[n,0,-1,0].item()
        city = seq_id_city[n, 1].item()
        lane_id = ori_lane_id[n,:,0].long().detach().cpu().numpy()
        _, uni_idx = np.unique(lane_id, return_index=True)
        lane_id = lane_id[sorted(uni_idx)]
        city = "PIT" if city == 0 else "MIA"
        accumulated_s = 0
        curr_lane_id = 0
        for l_idx in range(len(lane_id)):
            if lane_id[l_idx] != 0:
                lane, lane_length = get_lane_length(lane_id[l_idx], city)
                accumulated_s += lane_length
                curr_lane_id = lane_id[l_idx]
            else:
                break
        now_lane_ids = []
        future_lane_ids = []
        if accumulated_s < now_pred_s and curr_lane_id != 0:
            get_future_lane_id(now_lane_ids, curr_lane_id, city, now_pred_s - accumulated_s, future_lane_ids)
        overall_lane_ids.append(future_lane_ids)
    return overall_lane_ids
        

def train_model(pra_model, pra_data_loader, pra_optimizer, pra_epoch_log):
    # pra_model.to(dev)
    pra_model.train()
    rescale_xy = torch.ones((1,2,1,1)).to(dev)
    rescale_xy[:,0] = max_x
    rescale_xy[:,1] = max_y
    celossfunc = torch.nn.NLLLoss()
    loss_meter = meter.AverageValueMeter()
    # train model using training data

    loss_meter.reset()
    torch.autograd.set_detect_anomaly(True)
    for iteration, (ori_data, A, _, ori_map_data, lane_label, ori_trajectory, _) in enumerate(pra_data_loader):
        # print(iteration, ori_data.shape, A.shape)
        # ori_data: (N, C, T, V)
        # C = 11: [frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading] + [mask]
        # max_xy = torch.max(torch.max(torch.max(torch.abs(ori_data[:,3:5]), dim=2)[0], dim=-1)[0], dim=0)[0]
        # rescale_xy[:,:,0,0] = max_xy
        data, no_norm_loc_data, object_type = preprocess_data(ori_data, rescale_xy)
        map_data, _, _, = preprocess_data(ori_map_data, rescale_xy)
        trajectory, no_norm_trajectory, _, = preprocess_data(ori_trajectory, rescale_xy, [1,2,3,4])
        lane_label = torch.argmax(lane_label, dim=1).long().to(dev)
        for now_history_frames in range(history_frames, history_frames + 1):
            # for now_history_frames in range(1, data.shape[-2]):
            input_data = data[:,:,:now_history_frames,:] # (N, C, T, V)=(N, 4, 6, 120)
            output_loc_GT = data[:,:2,now_history_frames:,:] # (N, C, T, V)=(N, 2, 6, 120)
            input_trajectory = trajectory[:,:,:now_history_frames,:]
            input_no_norm_trajectory = no_norm_trajectory[:,:,:now_history_frames,:]
            output_trajectory_GT = trajectory[:,:2,now_history_frames:,:] # (N, C, T, V)=(N, 2, 6, 120)
            
            A = A.float().to(dev)

            predicted, att = pra_model(pra_x=input_trajectory, pra_map=input_no_norm_trajectory, pra_A=A, pra_pred_length=output_trajectory_GT.shape[-2]) #, pra_teacher_forcing_ratio=0, pra_teacher_location=output_loc_GT (N, C, T, V)=(N, 2, 6, 120)
            
            argmax_att = torch.argmax(att, dim=-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            predicted = predicted.gather(dim=-1, index=argmax_att.repeat((1, predicted.shape[1], predicted.shape[2], 1)))
            trajectory = trajectory.gather(dim=-1, index=argmax_att.repeat((1, trajectory.shape[1], trajectory.shape[2], 1)))
            trajectory = trajectory[:,:,:,0:1]

            output_trajectory_GT = trajectory[:,:2,now_history_frames:,:1]
            history_output_trajectory_GT = trajectory[:,:2,:now_history_frames,:1]
            output_mask = trajectory[:,-1:,now_history_frames:,:1] # (N, C, T, V)=(N, 1, 6, 120)
            history_output_mask = trajectory[:,-1:,:now_history_frames,:1] # (N, C, T, V)=(N, 1, 6, 120)

            history_predicted = predicted[:,:,:now_history_frames] * rescale_xy
            predicted = predicted[:,:,now_history_frames:] * rescale_xy
            output_trajectory_GT = output_trajectory_GT * rescale_xy
            history_output_trajectory_GT = history_output_trajectory_GT * rescale_xy
          
            # overall_loss
            overall_sum_time, overall_num, _ = compute_RMSE(predicted[:,:,:,0:1], output_trajectory_GT[:,:,:,0:1], output_mask[:,:,:,0:1], pra_error_order=2)
            history_overall_sum_time, history_overall_num, _ = compute_RMSE(history_predicted[:,:,:,0:1], history_output_trajectory_GT[:,:,:,0:1], history_output_mask[:,:,:,0:1], pra_error_order=2)
            total_loss = config.loss_weight[0] * torch.sum(overall_sum_time) / torch.max(torch.sum(overall_num), torch.ones(1,).to(dev)) #(1,)
            if config.use_map and config.use_celoss:
                celoss = config.loss_weight[2] * celossfunc(att, lane_label)
                total_loss += celoss
            elif config.use_history:
                history_loss = config.loss_weight[1] * torch.sum(history_overall_sum_time) / torch.max(torch.sum(history_overall_num), torch.ones(1,).to(dev)) #(1,)
                total_loss += history_loss
            loss_meter.add(total_loss.item())
            now_lr = [param_group['lr'] for param_group in pra_optimizer.param_groups][0]
            
            pra_optimizer.zero_grad()
            total_loss.backward()
            pra_optimizer.step()

        my_print('|{}|{:>20}|\tIteration:{:>5}|\tLoss:{:.8f}|lr: {}|'.format(datetime.now(), pra_epoch_log, iteration, loss_meter.value()[0] ,now_lr))
    return loss_meter.value()[0]

        

def val_model(pra_model, pra_data_loader):
    # pra_model.to(dev)
    pra_model.eval()
    rescale_xy = torch.ones((1,2,1,1)).to(dev)
    rescale_xy[:,0] = max_x
    rescale_xy[:,1] = max_y
    all_overall_sum_list = []
    all_overall_num_list = []

    celossfunc = torch.nn.NLLLoss()
    # train model using training data
    for iteration, (ori_data, A, ori_mean_xy, ori_map_data, lane_label, ori_trajectory, seq_id_city) in enumerate(pra_data_loader):
        # data: (N, C, T, V)
        # C = 11: [frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading] + [mask]
        # max_xy = torch.max(torch.max(torch.max(torch.abs(ori_data[:,3:5]), dim=2)[0], dim=-1)[0], dim=0)[0]
        # rescale_xy[:,:,0,0] = max_xy
        # mean_xy = mean_xy.unsqueeze(-1).unsqueeze(-1)
        data, no_norm_loc_data, _ = preprocess_data(ori_data, rescale_xy)
        map_data, _, _, = preprocess_data(ori_map_data, rescale_xy)
        trajectory, no_norm_trajectory, _, = preprocess_data(ori_trajectory, rescale_xy, [1,2,3,4])
        lane_label = torch.argmax(lane_label, dim=1).long().to(dev)
        for now_history_frames in range(history_frames, history_frames + 1):
            input_data = data[:,:,:now_history_frames,:] # (N, C, T, V)=(N, 4, 6, 120)
            output_loc_GT = data[:,:2,now_history_frames:,:] # (N, C, T, V)=(N, 2, 6, 120)

            ori_output_loc_GT = no_norm_loc_data[:,:2,now_history_frames:,:]
            ori_history_output_loc_GT = no_norm_loc_data[:,:2,:now_history_frames,:]
            ori_output_last_loc = no_norm_loc_data[:,:2,now_history_frames-1:now_history_frames,:1]
            input_trajectory = trajectory[:,:,:now_history_frames,:]
            input_no_norm_trajectory = no_norm_trajectory[:,:,:now_history_frames,:]
            output_trajectory_GT = trajectory[:,:2,now_history_frames:,:] # (N, C, T, V)=(N, 2, 6, 120)
            output_mask = trajectory[:,-1:,now_history_frames:,:] # (N, C, T, V)=(N, 1, 6, 120)

            A = A.float().to(dev)
            pred, att = pra_model(pra_x=input_trajectory, pra_map=no_norm_trajectory, pra_A=A, pra_pred_length=output_trajectory_GT.shape[-2]) #, pra_teacher_forcing_ratio=0, pra_teacher_location=output_loc_GT (N, C, T, V)=(N, 2, 6, 120)

            argmax_att = torch.argmax(att, dim=-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            ori_trajectory = ori_trajectory.float().to(config.dev)
            ori_lane_id = ori_trajectory[:,0]
            # ori_trajectory = ori_trajectory.gather(dim=-1, index=argmax_att.repeat((1, ori_trajectory.shape[1], ori_trajectory.shape[2], 1)))
            # ori_trajectory = ori_trajectory[:,:,:,0:1].float().to(config.dev)

            ori_output_last_trajectory = ori_trajectory[:,1:3,now_history_frames-1:now_history_frames,:]
            
            predicted = pred * rescale_xy
            if config.vel_mode:
                for ind in range(1, predicted.shape[-2]):
                    predicted[:,:,ind] = torch.sum(predicted[:,:,ind-1:ind+1], dim=-2)
                predicted += ori_output_last_trajectory

            """ 
            predicted:   [C=2, T, V]
            ori_mean_xy: [2]
            seq_id_city: [2]
            ori_lane_id: [T, V]
            """
            now_ori_data = ori_data.detach().cpu().numpy() # (N, C, T, V)=(N, 11, 6, 120)
            now_ori_trajectory = ori_trajectory.detach().cpu().numpy() # (N, C, T, V)=(N, 11, 6, 120)
            now_map_data = ori_map_data.detach().cpu().numpy()
            now_att = att.detach().cpu().numpy()
            predicted_loc = torch.zeros_like(predicted)
            for n, (now_pred, now_mean_xy, now_seq_id_city, now_lane_id) in enumerate(zip(predicted, ori_mean_xy, seq_id_city, ori_lane_id)):
                for v in range(now_pred.shape[-1]):
                    if output_mask[n,0,0,v].item() != 0:
                        predicted_loc[n,:,:,v:v+1] = get_xy_trajectory(now_pred[:,:,v:v+1], now_mean_xy, now_seq_id_city, now_lane_id[:,v:v+1], now_history_frames)
                if config.view:
                    visulize(predicted_loc[n:n+1,:,:,:].detach().cpu().numpy(), now_ori_data[n:n+1, :5, :, :], now_ori_trajectory[n:n+1, -1:, 0:, :], now_map_data[n:n+1, 3:5], now_att[n:n+1], lane_label[n:n+1])

            # argmax_att = torch.argmax(att, dim=-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            # ori_output_loc_GT = ori_output_loc_GT.gather(dim=-1, index=argmax_att.repeat((1, ori_output_loc_GT.shape[1], ori_output_loc_GT.shape[2], 1)))

            predicted_loc = predicted_loc.gather(dim=-1, index=argmax_att.repeat((1, predicted_loc.shape[1], predicted_loc.shape[2], 1)))
            overall_sum_time, overall_num, x2y2 = compute_RMSE(predicted_loc[:,:,:,0:1], ori_output_loc_GT[:,:,:,0:1], output_mask[:,:,:,0:1])       
            now_x2y2 = x2y2.detach().cpu().numpy()
            now_x2y2 = now_x2y2.sum(axis=-1)
            overall_num = overall_num.detach().cpu().numpy()
            
            all_overall_num_list.extend(overall_num)
            all_overall_sum_list.extend(now_x2y2)

            # now_pred = predicted.detach().cpu().numpy() # (N, C, T, V)=(N, 2, 6, 120)
            # if config.view:
            #     visulize(now_pred[:,:,:,0:1], now_ori_data[:, :5, :, 0:1], now_ori_data[:, -1:, -1:, 0:1], now_map_data[:, 3:5], now_att, lane_label)

    all_overall_sum_list = np.array(all_overall_sum_list)
    all_overall_num_list = np.array(all_overall_num_list)
    return all_overall_sum_list, all_overall_num_list



def test_model(pra_model, pra_data_loader, all_pred_trajectory, all_gt_trajectory, all_city):
    # pra_model.to(dev)
    pra_model.eval()
    rescale_xy = torch.ones((1,2,1,1)).to(dev)
    rescale_xy[:,0] = max_x
    rescale_xy[:,1] = max_y
    all_overall_sum_list = []
    all_overall_num_list = []
    celossfunc = torch.nn.NLLLoss()
    with open(test_result_file, 'w') as writer:
        # train model using training data
        for ori_data, A, ori_mean_xy, ori_map_data, lane_label, ori_trajectory, seq_id_city  in tqdm(pra_data_loader):
            # data: (N, C, T, V)
            # C = 11: [frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading] + [mask]
            # max_xy = torch.max(torch.max(torch.max(torch.abs(ori_data[:,3:5]), dim=2)[0], dim=-1)[0], dim=0)[0]
            # rescale_xy[:,:,0,0] = max_xy
            data, no_norm_loc_data, _ = preprocess_data(ori_data, rescale_xy)
            map_data, _, _, = preprocess_data(ori_map_data, rescale_xy)
            trajectory, no_norm_trajectory, _, = preprocess_data(ori_trajectory, rescale_xy, [1,2,3,4])
            lane_label = torch.argmax(lane_label, dim=1).long().to(dev)

            now_history_frames = history_frames
            input_data = data[:,:,:now_history_frames,:] # (N, C, T, V)=(N, 4, 6, 120)
            output_loc_GT = data[:,:2,now_history_frames:,:] # (N, C, T, V)=(N, 2, 6, 120)
            output_mask = data[:,-1:,now_history_frames:,:] # (N, C, T, V)=(N, 1, 6, 120)

            ori_output_loc_GT = no_norm_loc_data[:,:2,now_history_frames:,:]
            ori_history_output_loc_GT = no_norm_loc_data[:,:2,:now_history_frames,:]
            ori_output_last_loc = no_norm_loc_data[:,:2,now_history_frames-1:now_history_frames,:1]
            input_trajectory = trajectory[:,:,:now_history_frames,:]
            output_trajectory_GT = trajectory[:,:2,now_history_frames:,:] # (N, C, T, V)=(N, 2, 6, 120)
            output_mask = trajectory[:,-1:,now_history_frames:,:] # (N, C, T, V)=(N, 1, 6, 120)

            A = A.float().to(dev)
            predicted, att = pra_model(pra_x=input_trajectory, pra_map=trajectory, pra_A=A, pra_pred_length=config.future_frames) #, pra_teacher_forcing_ratio=0, pra_teacher_location=output_loc_GT (N, C, T, V)=(N, 2, 6, 120)
            ########################################################
            # Compute details for training
            ########################################################

            ori_trajectory = ori_trajectory.float().to(config.dev)

            # argmax_att = torch.argmax(att, dim=-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat((1, ori_trajectory.shape[1], ori_trajectory.shape[2], 1))
            # ori_trajectory = ori_trajectory.float().gather(dim=-1, index=argmax_att)
            ori_trajectory = ori_trajectory[:,:,:,:1]

            ori_output_last_trajectory = ori_trajectory[:,1:3,now_history_frames-1:now_history_frames,:1]
            ori_lane_id = ori_trajectory[:,0]
            
            predicted = predicted * rescale_xy
            if config.vel_mode:
                for ind in range(1, predicted.shape[-2]):
                    predicted[:,:,ind] = torch.sum(predicted[:,:,ind-1:ind+1], dim=-2)
                predicted += ori_output_last_trajectory

            """ 
            predicted:   [N, C=2, T, V]
            ori_mean_xy: [N, 2]
            seq_id_city: [N, 2]
            ori_lane_id: [N, T, V]
            """

            ori_future_lane_id = get_lane_id(predicted, ori_lane_id, seq_id_city) # (N, K, L) K lane graphs，with L lanes
            for now_pred, now_seq_id_city, now_lane_id, now_future_lane_id, now_output_loc_GT in zip(predicted, seq_id_city, ori_lane_id, ori_future_lane_id, ori_output_loc_GT):
                now_output_loc_GT = now_output_loc_GT[:,:,0].permute((1,0)).detach().cpu().numpy()
                now_seq_id = now_seq_id_city[0].long().item()
                now_city = now_seq_id_city[1].long().item()
                now_city = "PIT" if now_city == 0 else "MIA"
                pred_trajectory = []
                if len(now_future_lane_id) > 0 :
                    for future_lane_id in now_future_lane_id:
                        now_pred_trajectory = get_xy_trajectory(now_pred, torch.tensor([0,0]), now_seq_id_city, now_lane_id, now_history_frames, future_lane_id)
                        now_pred_trajectory = now_pred_trajectory[:,:,0].permute((1,0)).detach().cpu().numpy()
                        pred_trajectory.append(now_pred_trajectory)
                else:
                    now_pred_trajectory = get_xy_trajectory(now_pred, torch.tensor([0,0]), now_seq_id_city, now_lane_id, now_history_frames)
                    now_pred_trajectory = now_pred_trajectory[:,:,0].permute((1,0)).detach().cpu().numpy()
                    pred_trajectory.append(now_pred_trajectory)
                zero_pred = np.zeros((config.future_frames, 2))
                while len(pred_trajectory) < config.sample_times:
                    pred_trajectory.append(zero_pred)
                all_pred_trajectory[now_seq_id] = pred_trajectory
                all_city[now_seq_id] = now_city
                all_gt_trajectory = now_output_loc_GT
    # return all_overall_sum_list, all_overall_num_list


def run_trainval(pra_model, pra_traindata_path, pra_testdata_path):
    my_print("train_data_file: [%s]"%pra_traindata_path)
    my_print("test_data_file: [%s]"%pra_testdata_path)
    learning_rate = base_lr
    optimizer = optim.Adam(
        [{'params':model.parameters()},],) # lr = 0.0001)
    pre_loss = float("inf")
    best_epoch = 0
    train_data_files = sorted([os.path.join(pra_traindata_path, f) for f in os.listdir(pra_traindata_path) if f.split(".")[-1] == "pkl"])
    # train_data_files = train_data_files[:1]
    for now_epoch in range(total_epoch):
        total_train_loss = 0
        total_val_loss = 0
        # all_loader_train = itertools.chain(loader_train, loader_test)
        for idx, train_data_file in enumerate(train_data_files):
            # train_data_file = os.path.join(data_root, train_data_path, train_data_file)
            loader_train = data_loader(train_data_file, pra_batch_size=batch_size_train, pra_shuffle=True, pra_drop_last=True, train_val_test='train')
            # loader_test = data_loader(pra_testdata_path, pra_batch_size=batch_size_train, pra_shuffle=True, pra_drop_last=True, train_val_test='all')

            # evaluate on testing data (observe 5 frame and predict 1 frame)
            loader_val = data_loader(train_data_file, pra_batch_size=batch_size_val, pra_shuffle=False, pra_drop_last=False, train_val_test='val') 
            my_print('#######################################Train_%02d/%02d'%(idx+1, len(train_data_files)))
            now_loss = train_model(pra_model, loader_train, pra_optimizer=optimizer, pra_epoch_log='Epoch:{:>4}/{:>4}'.format(now_epoch, total_epoch))
            # now_loss = train_model(pra_model, all_loader_train, pra_optimizer=optimizer, pra_epoch_log='Epoch:{:>4}/{:>4}'.format(now_epoch, total_epoch))
            total_train_loss += now_loss
            if now_epoch % 5 == 0:
                config.view = True
            else:
                config.view = False
            my_print('#######################################Test_%02d/%02d'%(idx+1, len(train_data_files)))
            overall_loss_time = display_result(
                val_model(pra_model, loader_val), 
                pra_pref='{}_Epoch{}'.format('Test', now_epoch)
                )
            total_val_loss += np.mean(overall_loss_time)
        total_train_loss /= len(train_data_files)
        total_val_loss /= len(train_data_files)
        if total_val_loss < pre_loss:
            pre_loss = total_val_loss
            best_epoch = now_epoch
        else:
            learning_rate *= lr_decay
            if learning_rate < 1e-10:
                my_print("best epoch: %d, best loss: %f"%(best_epoch, pre_loss))
                break
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate
        my_print("now_train_loss: %f, best epoch: %d, best val loss: %f"%(total_train_loss, best_epoch, pre_loss))
        my_save_model(pra_model, now_epoch)


def run_val(pra_model, pra_data_path):
    my_print("val_data_file: [%s]"%pra_data_path)
    pra_model.eval()
    val_data_files = sorted([os.path.join(pra_data_path, f) for f in os.listdir(pra_data_path) if f.split(".")[-1] == "pkl"])
    total_val_loss = 0
    val_data_files = val_data_files[:1]
    for idx, val_data_file in enumerate(val_data_files):
        my_print('#######################################Val_%02d/%02d'%(idx+1, len(val_data_files)))
        loader_val = data_loader(val_data_file, pra_batch_size=batch_size_val, pra_shuffle=False, pra_drop_last=False, train_val_test='test')
        # overall_loss_time = display_result(
        #     val_model(pra_model, loader_val), 
        #     pra_pref='{}_Epoch{}'.format('val', now_epoch)
        #     )
        overall_loss_time = display_result(
            val_model(pra_model, loader_val), 
            pra_pref='{}_Epoch{}'.format('Val', idx)
            )
        total_val_loss += overall_loss_time
    total_val_loss /= (len(val_data_files))
    pra_pref='{}_{}'.format('Val', "overall")
    overall_log = '|{}|[{}] All_All: {}'.format(datetime.now(), pra_pref, ' '.join(['{:.3f}'.format(x) for x in list(total_val_loss) + [np.mean(total_val_loss)]]))
    my_print(overall_log)


def run_test(pra_model, pra_data_path, save_result=True):
    my_print("test_data_file: [%s]"%pra_data_path)
    pra_model.eval()
    test_data_files = sorted([os.path.join(pra_data_path, f) for f in os.listdir(pra_data_path) if f.split(".")[-1] == "pkl"])
    # test_data_files = test_data_files[:1]
    num_files = len(test_data_files)
    all_gt_trajectory = {}
    all_pred_trajectory = {}
    all_city = {}
    for idx, test_data_file in enumerate(test_data_files):
        my_print('#######################################test_%02d/%02d'%(idx+1, num_files))
        print("test data file: [%s]"%test_data_file)
        loader_test = data_loader(test_data_file, pra_batch_size=batch_size_test, pra_shuffle=False, pra_drop_last=False, train_val_test='test')
        test_model(pra_model, loader_test, all_pred_trajectory, all_gt_trajectory, all_city)
    try:
        result = eval_forecasting.compute_forecasting_metrics(all_pred_trajectory, all_gt_trajectory, all_city, config.sample_times, future_frames, 2)
        my_print(result)
    except:
        pass
    if save_result:
        output_path = config.pretrained_model_path.replace("trained_models", "result")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # all_pred_trajectory = np.array(all_pred_trajectory)
        generate_forecasting_h5(all_pred_trajectory, output_path)



if __name__ == '__main__':
    if config.train:
        for name, value in vars(config).items():
            if not name.startswith("_") and not isinstance(value, ModuleType) :
                my_print("%s: [%s]"%(name, value))
    graph_args = config.graph_args
    model = Model(in_channels=config.in_channels, graph_args=graph_args, edge_importance_weighting=True, use_cuda=use_cuda, dropout=config.dropout)
    if config.use_cuda:
        model.cuda()
    # model.to(dev)

    if config.load_model or not config.train:
        pretrained_model_path = config.pretrained_model_path
        model = my_load_model(model, pretrained_model_path)

    if config.convert_model:
        model.eval().cpu()
        example = torch.rand(1, 4, 6, 120)
        example_A = torch.rand(3, 120, 120)
        example_l = 6
        my_print("convert jit script model from [%s]"%config.pretrained_model_path)
        script_model = torch.jit.script(model)
        # script_model = torch.jit.trace(model.cpu(), (example, example_A))
        torch.jit.save(script_model, "grip_predictor.pt")
        os._exit(0)

    data_root = config.data_root
    # train_data_file = os.path.join(data_root, train_data_path)
    # test_data_file = os.path.join(data_root, test_data_path)
    # print("train_data_file: [%s]"%train_data_file)
    # print("test_data_file: [%s]"%test_data_file)

    test_data_path = config.test_data_path
    test_data_file = config.test_data_file
    train_data_path = os.path.join(data_root, config.train_data_path)
    val_data_path = os.path.join(data_root, config.val_data_path)
    test_data_path = os.path.join(data_root, config.test_data_path)
    # test_data_file = os.path.join(data_root, test_data_path, test_data_file)
    # train and evaluate model
    if config.train:
        # train_data_files = sorted([os.path.join(train_data_path, f) for f in os.listdir(train_data_path) if f.split(".")[-1] == "pkl"])
        # for train_data_file in train_data_files[:1]:
            # train_data_file = os.path.join(data_root, train_data_path, train_data_file)
        run_trainval(model, pra_traindata_path=train_data_path, pra_testdata_path=test_data_file)
    else:
        run_val(model, val_data_path)
        # run_test(model, val_data_path)
        # run_test(model, test_data_path)
    
        
        

