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
from util.map_util import HDMap

from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils import centerline_utils
from shapely.geometry import LineString

import config.configure as config
# from util.map_util import HDMap, LaneSegment

if config.map_type == "argo":
    print("loading ArgoverseMap")
    avm = ArgoverseMap()

CUDA_VISIBLE_DEVICES='0'
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch()

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
    # att: N T*V, T'*V'
    # outputs *= 5 
    # targets *= 5
    # outputs = outputs.astype("int") + 255 
    # targets = targets.astype("int") + 255
    history_frame_num = targets.shape[2] - outputs.shape[2]
    idx = 0
    for outs, tars, mask in zip(outputs, targets, output_mask):
        for i in range(outs.shape[-1]):
            img = np.ones((512,512,3)) * 255
            out = outs[:,:,i].copy()
            tar = tars[-2:,:,i].copy()
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
                        if a[ii] == 0:
                            continue
                        segment = (lane[:,:,ii] - min_xy) / scale_val + offset
                        segment = segment.astype("int")
                        for jj in range(segment.shape[-1] - 1):
                            color = (0,0,0) 
                            thickness = 1
                            if abs(a[ii] - a[argmax_a]) < 1e-6: #config.use_map and 
                                color = (0,255,0)
                            if label[i] == ii:
                                thickness = 2   
                            p1 = segment[:,jj]
                            p2 = segment[:, jj + 1]
                            if (p1[0] != zero_pos[0] or p1[1] != zero_pos[1]) and (p2[0] != zero_pos[0] or p2[1] != zero_pos[1]):
                                img = cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), color, thickness)
                            if (a[ii] > 0.2 or abs(a[ii] - a[argmax_a]) < 1e-6) and jj == segment.shape[-1] * ii // lane.shape[-1]:
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


                obs_info = "obs[{:0>5d}]_frame[{:0>5d}]_type[{:d}]_[{:0>4d}].jpg".format(*(tars[(1,0,2),history_frame_num,i].astype("int")), count)
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
        try:
            # print("city: ", city)
            city = "PIT" if city == "MIA" else "MIA"
            lane = avm.city_lane_centerlines_dict[city][lane_id]
        except:
            # print(city, lane_id)
            return 0, 0
    
    segment = lane.centerline
    lane_string = LineString(segment)
    curr_point = lane_string.interpolate(s).bounds
    n_point = curr_point
    if (s + 1) < lane_string.length:
        n_point = lane_string.interpolate(s + 1).bounds
    elif (s - 1) > 0 :
        n_point = lane_string.interpolate(s - 1).bounds
    heading = np.arctan2((curr_point[1] - n_point[1]), (curr_point[0] - n_point[0])) or 0
    x = curr_point[0] + l * np.sin(heading)
    y = curr_point[1] - l * np.cos(heading)
    return x, y
    

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
        trajectory, _, _, = preprocess_data(ori_trajectory, rescale_xy, [1,2,3,4])
        lane_label = torch.argmax(lane_label, dim=1).long().to(dev)
        for now_history_frames in range(history_frames, history_frames + 1):
            # for now_history_frames in range(1, data.shape[-2]):
            input_data = data[:,:,:now_history_frames,:] # (N, C, T, V)=(N, 4, 6, 120)
            output_loc_GT = data[:,:2,now_history_frames:,:] # (N, C, T, V)=(N, 2, 6, 120)
            output_mask = data[:,-1:,now_history_frames:,:] # (N, C, T, V)=(N, 1, 6, 120)
            input_trajectory = trajectory[:,:,:now_history_frames,:]
            output_trajectory_GT = trajectory[:,:2,now_history_frames:,:] # (N, C, T, V)=(N, 2, 6, 120)
            output_mask = trajectory[:,-1:,now_history_frames:,:] # (N, C, T, V)=(N, 1, 6, 120)
            
            A = A.float().to(dev)
            # t1 = time.time()
            # print("load data time: %f"%(t1 - t3))
            predicted, att = pra_model(pra_x=input_trajectory, pra_map=trajectory, pra_A=A, pra_pred_length=output_trajectory_GT.shape[-2]) #, pra_teacher_forcing_ratio=0, pra_teacher_location=output_loc_GT (N, C, T, V)=(N, 2, 6, 120)
            # t2 = time.time()
            # print("pred time: %f"%(t2-t1))
            predicted = predicted * rescale_xy
            output_loc_GT = output_loc_GT * rescale_xy
            output_trajectory_GT = output_trajectory_GT * rescale_xy
            ########################################################
            # Compute loss for training
            ########################################################
            # We use abs to compute loss to backward update weights
            # (N, T), (N, T)
            # att = att[:,0]
            # celoss = 0
            # for i in range(att.shape[0]):
            #     a = att[i].clone().view((1, -1))
            #     label = lane_label[i:i+1].clone()
            #     celoss += celossfunc(torch.log(a), label)
            # celoss = celoss / att.shape[0]
            # log_att = torch.zeros_like(att).to(dev).float()
            # idx = att > 0
            # log_att[idx] = torch.log(att[idx])
            argmax_att = torch.argmax(att, dim=-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat((1, output_loc_GT.shape[1], output_loc_GT.shape[2], 1))
            output_trajectory_GT = output_trajectory_GT.gather(dim=-1, index=argmax_att)
            overall_sum_time, overall_num, _ = compute_RMSE(predicted[:,:,:,0:1], output_trajectory_GT[:,:,:,0:1], output_mask[:,:,:,0:1], pra_error_order=2)
            # overall_loss
            total_loss = torch.sum(overall_sum_time) / torch.max(torch.sum(overall_num), torch.ones(1,).to(dev)) #(1,)
            if config.use_map and config.use_celoss:
                celoss = celossfunc(att, lane_label)
                total_loss = config.loss_weight[0] * total_loss + config.loss_weight[1] * celoss
            else:
                total_loss = config.loss_weight[0] * total_loss
            loss_meter.add(total_loss.item())
            now_lr = [param_group['lr'] for param_group in pra_optimizer.param_groups][0]
            
            pra_optimizer.zero_grad()
            total_loss.backward()
            pra_optimizer.step()
            # t3 = time.time()
            # print("bp time: %f"%(t3-t2))
            # visulize(predicted.detach().cpu().numpy(), output_loc_GT.detach().cpu().numpy(), output_mask.detach().cpu().numpy())
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
            output_mask = data[:,-1:,now_history_frames:,:] # (N, C, T, V)=(N, 1, 6, 120)

            ori_output_loc_GT = no_norm_loc_data[:,:2,now_history_frames:,:]
            ori_output_last_loc = no_norm_loc_data[:,:2,now_history_frames-1:now_history_frames,:1]
            input_trajectory = trajectory[:,:,:now_history_frames,:]
            output_trajectory_GT = trajectory[:,:2,now_history_frames:,:] # (N, C, T, V)=(N, 2, 6, 120)
            output_mask = trajectory[:,-1:,now_history_frames:,:] # (N, C, T, V)=(N, 1, 6, 120)

            A = A.float().to(dev)
            predicted, att = pra_model(pra_x=input_trajectory, pra_map=trajectory, pra_A=A, pra_pred_length=output_trajectory_GT.shape[-2]) #, pra_teacher_forcing_ratio=0, pra_teacher_location=output_loc_GT (N, C, T, V)=(N, 2, 6, 120)
            ########################################################
            # Compute details for training
            ########################################################

            ori_trajectory = ori_trajectory.float().to(config.dev)

            # argmax_att = torch.argmax(att, dim=-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat((1, ori_trajectory.shape[1], ori_trajectory.shape[2], 1))
            # ori_trajectory = ori_trajectory.float().gather(dim=-1, index=argmax_att)
            ori_trajectory = ori_trajectory[:,:,:,:1]

            ori_output_trajectory_GT = ori_trajectory[:,1:3,now_history_frames:,:]
            ori_output_last_trajectory = ori_trajectory[:,1:3,now_history_frames-1:now_history_frames,:1]
            ori_lane_id = ori_trajectory[:,0]
            
            predicted = predicted * rescale_xy
            ori_output_trajectory_GT = ori_output_trajectory_GT*rescale_xy
            ori_output_loc_GT = ori_output_loc_GT * rescale_xy
            if config.vel_mode:
                for ind in range(1, predicted.shape[-2]):
                    predicted[:,:,ind] = torch.sum(predicted[:,:,ind-1:ind+1], dim=-2)
                predicted += ori_output_last_trajectory

            now_pred = torch.zeros(predicted.shape).to(config.dev)
            for n in range(predicted.shape[0]):
                accumulated_s = 0
                curr_pred = predicted[n,:,:,0].float().detach().cpu().numpy()
                lane_id = ori_lane_id[n,:,0].long().detach().cpu().numpy()
                mean_xy = ori_mean_xy[n].float().detach().cpu().numpy() 
                curr_seq_id_city = seq_id_city[n]
                city = curr_seq_id_city[1].long().item()
                city = "PIT" if city == 0 else "MIA" 
                for t in range(1, now_history_frames):
                    if lane_id[t] != lane_id[t - 1]:
                        try:
                            lane = avm.city_lane_centerlines_dict[city][lane_id[t - 1]]
                        except:
                            city = "PIT" if city == "MIA" else "MIA"
                            lane = avm.city_lane_centerlines_dict[city][lane_id[t - 1]]
                        segment = lane.centerline
                        lane_string = LineString(segment)
                        accumulated_s += lane_string.length
                for t in range(curr_pred.shape[1]):
                    if lane_id[t + now_history_frames] != lane_id[t + now_history_frames - 1]:
                        try:
                            lane = avm.city_lane_centerlines_dict[city][lane_id[t + now_history_frames - 1]]
                        except:
                            city = "PIT" if city == "MIA" else "MIA"
                            lane = avm.city_lane_centerlines_dict[city][lane_id[t + now_history_frames - 1]]
                        segment = lane.centerline
                        lane_string = LineString(segment)
                        accumulated_s += lane_string.length
                    l_id = lane_id[t + now_history_frames]
                    s = curr_pred[0, t] - accumulated_s
                    l = curr_pred[1, t]
                    x, y = point_from_lane(l_id, s, l, city)
                    if x != 0 or y != 0:
                        now_pred[n,0,t,0] = x - mean_xy[0]
                        now_pred[n,1,t,0] = y - mean_xy[1]

            overall_sum_time, overall_num, x2y2 = compute_RMSE(now_pred[:,:,:,0:1], ori_output_loc_GT[:,:,:,0:1], output_mask[:,:,:,0:1])       
            now_pred = now_pred.detach().cpu().numpy() # (N, C, T, V)=(N, 2, 6, 120)
            now_ori_data = ori_data.detach().cpu().numpy() # (N, C, T, V)=(N, 11, 6, 120)
            now_map_data = ori_map_data.detach().cpu().numpy()
            now_att = att.detach().cpu().numpy()
            if config.view:
                visulize(now_pred[:,:,:,0:1], now_ori_data[:, :5, :, 0:1], now_ori_data[:, -1:, -1:, 0:1], now_map_data[:, 3:5], now_att, lane_label)

            ### overall dist
            # overall_sum_time, overall_num, x2y2 = compute_RMSE(predicted, output_loc_GT, output_mask)        
            # all_overall_sum_list.extend(overall_sum_time.detach().cpu().numpy())
            all_overall_num_list.extend(overall_num.detach().cpu().numpy())
            # x2y2 (N, 6, 39)
            now_x2y2 = x2y2.detach().cpu().numpy()
            now_x2y2 = now_x2y2.sum(axis=-1)
            all_overall_sum_list.extend(now_x2y2)

    all_overall_sum_list = np.array(all_overall_sum_list)
    all_overall_num_list = np.array(all_overall_num_list)
    return all_overall_sum_list, all_overall_num_list



def test_model(pra_model, pra_data_loader):
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
        for ori_data, A, _, ori_map_data, lane_label, ori_trajectory,  in tqdm(pra_data_loader):
            # data: (N, C, T, V)
            # C = 11: [frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading] + [mask]
            # max_xy = torch.max(torch.max(torch.max(torch.abs(ori_data[:,3:5]), dim=2)[0], dim=-1)[0], dim=0)[0]
            # rescale_xy[:,:,0,0] = max_xy
            data, no_norm_loc_data, _ = preprocess_data(ori_data, rescale_xy)
            map_data, _, _, = preprocess_data(ori_map_data, rescale_xy)
            lane_label = torch.argmax(lane_label, dim=1).long().to(dev)
            input_data = data[:,:,:history_frames,:] # (N, C, T, V)=(N, 4, 6, 120)
            output_mask = data[:,-1:,history_frames:,:] # (N, V)=(N, 120)
            # print(data.shape, A.shape, mean_xy.shape, input_data.shape)

            ori_output_loc_GT = no_norm_loc_data[:,:2,history_frames:,:]
            ori_output_last_loc = no_norm_loc_data[:,:2,history_frames-1:history_frames,:1]
        
            A = A.float().to(dev)
            # time1 = time.clock()
            predicted, att = pra_model(pra_x=input_data, pra_map=map_data, pra_A=A, pra_pred_length=future_frames) #, pra_teacher_forcing_ratio=0, pra_teacher_location=None (N, C, T, V)=(N, 2, 6, 120)
            # time2 = time.clock()
            # print(time2 - time1)
            predicted = predicted *rescale_xy 
            if config.vel_mode:
                for ind in range(1, predicted.shape[-2]):
                    predicted[:,:,ind] = torch.sum(predicted[:,:,ind-1:ind+1], dim=-2)
                predicted += ori_output_last_loc
            # celoss = celossfunc(torch.log(att[:,0]), lane_label)

            now_pred = predicted.detach().cpu().numpy() # (N, C, T, V)=(N, 2, 6, 120)
            # now_mean_xy = mean_xy.detach().cpu().numpy() # (N, 2)
            now_ori_data = ori_data.detach().cpu().numpy() # (N, C, T, V)=(N, 11, 6, 120)
            now_ori_output_loc_GT = ori_output_loc_GT.detach().cpu().numpy()
            now_output_mask = output_mask.detach().cpu().numpy()

            ### overall dist
            # overall_sum_time, overall_num, x2y2 = compute_RMSE(predicted, output_loc_GT, output_mask)        
            overall_sum_time, overall_num, x2y2 = compute_RMSE(predicted[:,:,:,0:1], ori_output_loc_GT[:,:,:,0:1], output_mask[:,:,:,0:1])        
            # all_overall_sum_list.extend(overall_sum_time.detach().cpu().numpy())
            all_overall_num_list.extend(overall_num.detach().cpu().numpy())
            # x2y2 (N, 6, 39)
            now_x2y2 = x2y2.detach().cpu().numpy()
            now_x2y2 = now_x2y2.sum(axis=-1)
            # all_overall_sum_list.extend(now_x2y2)

            # now_mask = now_ori_data[:, -1, history_frames - 1, :] # (N, V)
            
            distance_error = (np.sum(((now_pred[:,:,:,0] - now_ori_output_loc_GT[:,:,:,0])*now_output_mask[:,:,:,0]) ** 2, axis=1) ** 0.5)
            all_overall_sum_list.extend(distance_error)

            now_map_data = ori_map_data.detach().cpu().numpy()
            now_att = att.detach().cpu().numpy()
            # visulize(now_pred[:,:,:,0:1], now_ori_data[:, :5, :, 0:1], now_ori_data[:, -1:, -1:, 0:1], now_map_data[:, 3:5], now_att)

    all_overall_sum_list = np.array(all_overall_sum_list)
    all_overall_num_list = np.array(all_overall_num_list)
    all_over_loss = display_result(
        [all_overall_sum_list, all_overall_num_list], 
        pra_pref='{}_{}'.format('Test', test_data_file), 
        pra_error_order=1
        )
    return all_over_loss
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
    train_data_files = train_data_files[:1]
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
            my_print('#######################################Train_%d'%idx)
            now_loss = train_model(pra_model, loader_train, pra_optimizer=optimizer, pra_epoch_log='Epoch:{:>4}/{:>4}'.format(now_epoch, total_epoch))
            # now_loss = train_model(pra_model, all_loader_train, pra_optimizer=optimizer, pra_epoch_log='Epoch:{:>4}/{:>4}'.format(now_epoch, total_epoch))
            total_train_loss += now_loss
            if now_epoch % 5 == 0:
                config.view = True
            else:
                config.view = False
            my_print('#######################################Test_%d'%idx)
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
    # val_data_files = val_data_files[:1]
    for idx, val_data_file in enumerate(val_data_files):
        my_print('#######################################Val_%d'%idx)
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


def run_test(pra_model, pra_data_path):
    my_print("test_data_file: [%s]"%pra_data_path)
    pra_model.eval()
    test_data_files = sorted([os.path.join(pra_data_path, f) for f in os.listdir(pra_data_path) if f.split(".")[-1] == "pkl"])
    total_test_loss = 0
    test_data_files = test_data_files[:1]
    for idx, test_data_file in enumerate(test_data_files):
        my_print('#######################################Test_%d'%idx)
        loader_test = data_loader(test_data_file, pra_batch_size=batch_size_test, pra_shuffle=False, pra_drop_last=False, train_val_test='test')
        # overall_loss_time = display_result(
        #     test_model(pra_model, loader_test), 
        #     pra_pref='{}_Epoch{}'.format('Test', now_epoch)
        #     )
        overall_loss_time = test_model(pra_model, loader_test)
        total_test_loss += overall_loss_time
    total_test_loss /= (len(test_data_files))
    pra_pref='{}_{}'.format('Test', "overall")
    overall_log = '|{}|[{}] All_All: {}'.format(datetime.now(), pra_pref, ' '.join(['{:.3f}'.format(x) for x in list(total_test_loss) + [np.mean(total_test_loss)]]))
    my_print(overall_log)



if __name__ == '__main__':
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
    
        
        

