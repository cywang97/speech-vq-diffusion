import os

string = "python train.py --name audiolm_train --config_file configs/audiolm_6k.yaml --num_node 1 --tensorboard --load_path OUTPUT/pretrained_model/CC_pretrained_pth"

os.system(string)

