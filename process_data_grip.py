import os
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

class data_process_grip:
    def __init__(self):
        self.prediction_data_list=[]
        self.save_path = args.save_dir
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        # self.save_path = os.path.join(self.save_path, args.save_dir)
        print(self.save_path)
        # self.test_save_path = os.path.join(self.save_path, "prediction_test")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        # if not os.path.exists(self.test_data_path):
        #     os.makedirs(self.test_data_path)
        self.prediction_data_count = 0

    def work(self, name, file):
        # print("Loading data from file: [%s]"%file)
        ans = pd.read_csv(name)
        ans = np.array(ans)
        self.prediction_data_list = []
        city = ans[0, -1]
        t = 0
        timestamp = ans[0][0]
        for feature_curr in ans:
            pred_data = []
            if timestamp != feature_curr[0]:
                timestamp = feature_curr[0]
                t += 1
            obs_id = feature_curr[1]
            obs_id = int("".join(obs_id.split("-")))
            obs_type = feature_curr[2]
            if obs_type == "AV" or obs_type == "AGENT":
                if obs_type == "AV":
                    AV_ID = obs_id
                    obs_type = 0
                else:
                    AGENT_ID = obs_id
                    obs_type = 1
                    continue
            else:
                continue
                obs_type = 5
            pos_x = feature_curr[3]
            pos_y = feature_curr[4]
            pos_z = 0
            heading = 0
            pred_data.append(t)
            pred_data.append(str(obs_id))
            pred_data.append(str(obs_type))
            pred_data.append(str(float(pos_x)))
            pred_data.append(str(float(pos_y)))
            pred_data.append(str(float(pos_z)))
            pred_data.append("0")
            pred_data.append("0")
            pred_data.append("0")
            pred_data.append(str(float(heading)))
            pred_data.append(city)

            self.prediction_data_list.append(pred_data)

        # if len(self.prediction_data_list) > 5000:
        self.prediction_data_list = sorted(sorted(self.prediction_data_list, key=lambda x: x[1]), key=lambda x: x[0])
        txt_file = os.path.join(self.save_path, "%05d_train_data_%s.txt"%(self.prediction_data_count, file))
        self.prediction_data_list = np.array(self.prediction_data_list)
        idx = self.prediction_data_list[:, 1] == str(AV_ID)
        now = self.prediction_data_list[idx]
        
        distance = np.sqrt(np.square(float(now[0][3]) - float(now[19][3])) + np.square(float(now[0][4]) - float(now[19][4])))
        angle = np.arctan2(float(now[0][4]) - float(now[-1][4]), float(now[0][3]) - float(now[-1][3])) - np.arctan2(float(now[0][4]) - float(now[24][4]), float(now[0][3]) - float(now[24][3]))
        angle = int(angle / np.pi * 180) % 360
        if distance < 1: #  or angle > 350 or angle < 10
            return 0
        with open(txt_file, "w") as f:
            # timestamp = float(self.prediction_data_list[0][0])    
            for data in self.prediction_data_list:
                # data[0] = str(round((float(data[0]) - timestamp) / 0.1))
                str_data = " ".join(data) + "\n"
                f.write(str_data)        
        self.prediction_data_count += 1
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="/datastore/data/cxl/argoverse/train/data/", required=False, help="data load dir")
    parser.add_argument("-s", "--save_dir", type=str, default="/datastore/data/cxl/GRIP/data/argo/all/prediction_train/raw_data_AV", required=False, help="data save dir")
    parser.add_argument("-n", "--num", type=int, default=10000, required=False, help="num of files to load")
    args = parser.parse_args()
    # DATA_DIR = 'data/argo/forecasting_sample/data/'
    # nameList = ['2645.csv','3700.csv','3828.csv','3861.csv']
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    DATA_DIR = args.data_dir
    nameList = sorted(os.listdir(DATA_DIR))
    args.num = min(args.num, len(nameList))
    count = 0
    processor = data_process_grip()
    for name in tqdm(nameList):
        if processor.work(os.path.join(DATA_DIR, name) ,name.split(".")[0]) > 0:
            count += 1
            if count % 500 == 0 and count != 0:
                print("[%4d/%4d] data loaded"%(count, args.num))
            if args.num == count:
                break