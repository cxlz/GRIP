import shutil
import os
data_path = "prediction_train"
for file in os.listdir(data_path):
    file_path = os.path.join(data_path, file)
    shutil.move(file_path, os.path.join(data_path, "turn_" + file))