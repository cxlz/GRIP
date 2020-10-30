# from torch.utils import data
#
#
# class MyData(data.Dataset):
#     r"""
#     self.x is all items' feature vectors.  [P, len]
#     self.id is the index of predicted agent. id
#     self.label is the future trajectory vector. [len]
#     """
#     def __init__(self, dataset, isTrain):
#         self.x = dataset
#         self.id = 0
#         self.label = dataset
#         raise NotImplementedError
#
#     def __getitem__(self, index):
#
#
#     def __len__(self):


# import pickle as pkl
#
# with open('data/feature/forecasting_features_test.pkl', "rb") as f:
#     grid_search = pkl.load(f)
#     print(grid_search)
import argparse
import glob
import logging
import os
import pickle
from tqdm import tqdm
import sys
try:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
except:
    pass
root_path = os.path.abspath(".")
source_list=  ['/cowarobot/src/map/proto',
               '/cowarobot/src/perception/prediction/proto',
               '/cowarobot/src',
               os.path.join(root_path, "src"),
               os.path.join(root_path, 'src/map/proto'),
               os.path.join(root_path, 'src/perception/prediction/proto')
                ]
for source in source_list:
  if (source not in sys.path):
    sys.path.append(source)


import pandas as pd
import numpy as np
import cv2

from perception.prediction.data.data_pipelines.data_preprocessing.map_util import HDMap, LaneSegment


X_ID = 3
Y_ID = 4
total_frames = 50
frame_steps = 3
time_resolution = 0.1
search_radius = 20
AVID = -1

# avm = ArgoverseMap()
root_path = os.path.abspath("./")
map_file = "data/map/huaxin/base_map.bin"
print(os.path.isfile(map_file))
hdmap = HDMap(os.path.join(root_path, map_file))


def vecLink(a, polyID):
    a = np.array(a)
    ans = []
    type = 0 if a[0, 1] == AVID else 1
    for i in range(a.shape[0] - 1):
        l, r = a[i], a[i + 1]
        now = [l[X_ID], l[Y_ID], r[X_ID], r[Y_ID], type,
               l[0],
               r[0],
               np.sqrt(np.square(l[X_ID]-r[X_ID])+np.square(l[Y_ID]-r[Y_ID])) / (r[0]-l[0]),
               polyID]
        ans.append(now)
    return ans





def work(file, name):
    with open(file, 'r') as reader:
        # print(train_file_path)
        content = np.array([x.strip().split(' ') for x in reader.readlines()]).astype(float)
    folder = file.split("/")[-1]
    now_dict = {}
    for row in content:
        # instance = {row[1]:row[2:]}
        n_dict = now_dict.get(int(row[0]), {})
        n_dict[row[1]] = row#[2:]
        # n_dict.append(instance)
        # now_dict[]
        now_dict[int(row[0])] = n_dict

    frame_id_set = sorted(set(now_dict.keys()))
    file_num = 0
    for index in tqdm(frame_id_set[:-total_frames+1:frame_steps]):
        for ind in range(index, index + total_frames):
            if ind not in frame_id_set or AVID not in now_dict[ind].keys():
                break
        else:
            curr_data = []
            for ind in range(index, index + total_frames):
                for obs_id, data in now_dict[ind].items():
                    curr_data.append(data)
            curr_data = sorted(sorted(curr_data, key=lambda x: x[0]), key=lambda x: x[1])
            curr_data = np.array(curr_data).astype("float")
            process_data(curr_data, "%s_%d.csv"%(folder, file_num))
            file_num += 1


def process_data(ans, file):

    track_id = 1


    # print(ans)

    AVX = 0
    AVY = 0
    AVTIME = 0

    tmp = []
    j = 0
    polyID = 0
    # img_half_scale = 20
    # img_resolution = 0.1
    # img_half_scale = round(img_half_scale / img_resolution)
    # img = np.ones((img_half_scale * 2, img_half_scale * 2, 3)) * 255
    for i in range(ans.shape[0]):
        if i + 1 == ans.shape[0] or ans[i, track_id] != ans[i + 1, track_id]:
            now = []
            while j <= i:
                now.append(ans[j])
                if j < i:
                    assert ans[j, 0] <= ans[j + 1, 0]
                j += 1
            vecList = vecLink(now, polyID)
            polyID += 1
            for vec in vecList:
                tmp.append(vec)
            if ans[i, 1] == AVID:
                if len(now) < 50:
                    return -1
                AVX, AVY = ans[i-30, 3], ans[i-30, 4]
                # distance = np.sqrt(np.square(now[0][3] - now[19][3]) + np.square(now[0][4] - now[19][4]))
                # angle = np.arctan2(now[0][4] - now[49][4], now[0][3] - now[49][3]) - np.arctan2(now[0][4] - now[24][4], now[0][3] - now[24][3])
                # angle = int(angle / np.pi * 180) % 360
                # if distance < 1 or angle > 350 or angle < 10:
                #     # print(name, distance, angle)
                #     return -1
                AVTIME = ans[i-30, 0]
            # if len(vecList) > 2:
            #     traj = np.array(vecList, dtype="float")
            #     traj[:, 0:3:2] -= AVX
            #     traj[:, 1:4:2] -= AVY
            #     traj = (traj / img_resolution).astype("int")
            #     for i in range(len(traj)):
            #         if traj[i, 4] == 0:
            #             line_color = (255, 0, 0)
            #             img = cv2.line(img, (traj[i, 0] + img_half_scale, traj[i, 1] + img_half_scale),
            #                             (traj[i, 2] + img_half_scale, traj[i, 3] + img_half_scale), line_color, thickness=2)
            #                 # img = cv2.circle(img, (traj[i, 0] + img_half_scale, traj[i, 1] + img_half_scale), 1, line_color, lineType=8)
            #         else:
            #             line_color = (0, 255, 0)
            #             img = cv2.circle(img, (traj[i, 0] + img_half_scale, traj[i, 1] + img_half_scale), 1, line_color, thickness= -1)
    # line_color = (0, 0, 0)

    lane_list = hdmap.get_lanes([AVX, AVY], search_radius)
    for lane in lane_list:
        lane_id = lane.id
        for segment in lane.segment:
            segment = np.array(segment).astype("float")
            for i in range(segment.shape[0] - 1):
                l, r = segment[i], segment[i + 1]

                t = 0
                if lane.turn == 2:
                    t = 1
                elif lane.turn == 3 or lane.turn == 4:
                    t = 2

                now = [l[0], l[1], r[0], r[1], 2,
                    0,
                    t,
                    0,
                    polyID]
                tmp.append(now)
                    # img = cv2.circle(img, (vecList[i, 0] + img_half_scale, vecList[i, 1] + img_half_scale), 1, line_color)
                # traj = np.array(now, dtype="float")
                # traj[0:3:2] -= AVX
                # traj[1:4:2] -= AVY
                # traj = (traj / img_resolution).astype("int")
                # img = cv2.line(img, (traj[0] + img_half_scale, traj[1] + img_half_scale),
                #                 (traj[2] + img_half_scale, traj[3] + img_half_scale), line_color, thickness=1)
        
            polyID += 1

    tmp = np.array(tmp)
    for i in range(tmp.shape[0]):
        tmp[i, 0] -= AVX
        tmp[i, 2] -= AVX
        tmp[i, 1] -= AVY
        tmp[i, 3] -= AVY
        for j in range(4):
            tmp[i , j] *= 100
        if tmp[i, 4] != 2:
            tmp[i, 5] = (tmp[i, 5] - AVTIME) * time_resolution
            tmp[i, 6] = (tmp[i, 6] - AVTIME) * time_resolution


    # print(tmp)
    # print(tmp.shape)
    # cv2.imshow("img", img)
    # cv2.waitKey(1)
    pf = pd.DataFrame(data=tmp)
    pf.to_csv(os.path.join(args.save_dir, 'data_' + file), header=False, index=False)
    return 1


# nameList = ['2645.csv','4791.csv']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="src/perception/prediction/data/prediction/EQ1-37-09.08-21.24/feature/txt", required=False, help="data load dir")
    parser.add_argument("-s", "--save_dir", type=str, default="src/perception/prediction/data/prediction/EQ1-37-09.08-21.24/feature/txt/vectornet_data", required=False, help="data save dir")
    parser.add_argument("-n", "--num", type=int, default=5000, required=False, help="num of files to load")
    args = parser.parse_args()
    # DATA_DIR = 'data/argo/forecasting_sample/data/'
    # nameList = ['2645.csv','3700.csv','3828.csv','3861.csv']
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    DATA_DIR = args.data_dir
    print("generate vectornet data from %s"%DATA_DIR)
    nameList = os.listdir(DATA_DIR)
    nameList = [name for name in nameList if name.endswith(".txt")]
    nameList = sorted(nameList, key=lambda x: int(x.split(".")[-2].split("_")[-1]))
    args.num = min(args.num, len(nameList))
    count = 0
    for name in tqdm(nameList):
        filepath = os.path.join(DATA_DIR, name)
        if os.path.isfile(filepath):
            work(filepath ,name)  

# df = pd.read_pickle(FEATURE_DIR + 'forecasting_features_test.pkl')

# feature_idx = [FEATURE_FORMAT["X"], FEATURE_FORMAT["Y"]]
# seq_id = df["SEQUENCE"].values
#
# obs_trajectory = np.stack(
#     df["FEATURES"].values)[:, :20, feature_idx].astype("float")

# print(obs_trajectory.shape)
# print(df.info())
# print(type(df))
# print(df)
#
# print("-------------")
# print(df["SEQUENCE"].values)
# print("-------------")
# print(df["FEATURES"].values)
# print(df["FEATURES"].values.shape)
# print(df["FEATURES"].values[0].shape)
# print(df["FEATURES"].values.shape)
# print(df["FEATURES"].values[0].shape)
# print("-------------")
# print(df["CANDIDATE_CENTERLINES"].values)
# print("-------------")
# print(df["ORACLE_CENTERLINE"].values)
# print("-------------")
# print(df["CANDIDATE_NT_DISTANCES"].values)

# print("-------------")
# print(df['FEATURES'])
# print("-------------")
# print(df["FEATURES"].values)
