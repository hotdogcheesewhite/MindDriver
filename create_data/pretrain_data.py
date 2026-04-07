import os
import re
import pickle
import ndjson
import json
import tiktoken
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from datetime import datetime, timedelta

system="""You're an autonomous vehicle's brain. Coordinates: X-axis is perpendicular, and Y-axis is parallel to the direction you're facing. You're at point (0,0). Units: meters. Based on the provided particulars, you can generate CAM_FRONT image at the 0.5 second in the future.\n"""

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
num_language_tokens = 0
num_system_tokens = 0
num_user_tokens = 0
num_assistant_tokens = 0
traj_only = True
train_messages = []

gt_indices=json.load(open('./MoVQGAN/gt_indices_pretrain.json'))
dataroot = './LLaMA-Factory/data/nuscenes'
nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)

cam_front_dir = os.path.join(dataroot, 'sweeps/CAM_FRONT')
supported_extensions = ('.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG')
image_files = [
    f for f in os.listdir(cam_front_dir)
    if os.path.isfile(os.path.join(cam_front_dir, f)) and f.endswith(supported_extensions)
]
image_files = sorted(image_files)

for i, now_path in enumerate(tqdm(image_files)):
        images_path=[]
        try:
            target_name=None
            filename = now_path
            parts = filename.split("__")
            timestamp_str = filename.split("__")[2].split(".")[0] 
            original_timestamp = int(timestamp_str) / 1e6  
            dt = datetime.fromtimestamp(original_timestamp)
            future_dt = dt + timedelta(seconds=0.5)
            new_timestamp = int(future_dt.timestamp() * 1e6) 
            new_filename = filename.replace(timestamp_str, str(new_timestamp))
            new_filepath = os.path.join(cam_front_dir, new_filename)

            if os.path.exists(new_filepath):
                target_name=new_filename
            else:
                prefix = str(new_timestamp)[:11]  

                for j in range(i+1,i+10):
                    if image_files[j].startswith(f"{parts[0]}__{parts[1]}__{prefix}"):
                        target_name=image_files[j]
                        break

                if target_name == None:
                    prefix_new = str(int(prefix)-1)
                    for j in range(i+1,i+10):
                        if image_files[j].startswith(f"{parts[0]}__{parts[1]}__{prefix_new}"):
                            target_name=image_files[j]
                            break

                if target_name == None:
                    prefix_new = str(int(prefix)+1)
                    for j in range(i+1,i+10):
                        if image_files[j].startswith(f"{parts[0]}__{parts[1]}__{prefix_new}"):
                            target_name=image_files[j]
                            break

            if target_name == None:
                continue

            next_img_token=gt_indices[target_name]
            next_img_token = str(next_img_token).replace(" ", "")
            numbers = next_img_token.strip('[]').split(',')
            next_img_token = ''.join([f'<|{num}|>' for num in numbers])

        except:
            continue

        images_path.append(os.path.join('data/nuscenes/sweeps/CAM_FRONT', filename))

        train_message = {
                            "id": filename,
                            "images": images_path,
                            "system": system,
                            "conversations": [
                                {
                                    "from": "human",
                                    "value": "This is the CAM_FRONT image of the current frame: <image>\n Please generate CAM_FRONT image at the 0.5 second in the future.\n"
                                },
                                {
                                    "from": "gpt",
                                    "value": next_img_token + " These are the visual tokens of CAM_FRONT image at the 0.5 second in the future.\n <|endoftext|><|im_end|>\n"
                                }
                            ]
                        }
        train_messages.append(train_message)


with open("./LLaMA-Factory/data/pretrain_data.json", "w") as f:
    json.dump(train_messages, f, indent=4)

