import pickle
import re
import json
import argparse
import tiktoken
from nuscenes.nuscenes import NuScenes
from prompt_message import  generate_user_message, generate_assistant_message, generate_goal_message
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from pathlib import Path

system="You're an autonomous vehicle's brain. Coordinates: X-axis is perpendicular, and Y-axis is parallel to the direction you're facing. You're at point (0,0). Units: meters. Based on the provided particulars, please output the CAM_FRONT image at the 0.5 second in the future and plan waypoints (0.5s intervals) for the next 3 seconds."

parser = argparse.ArgumentParser(description="Choose to use train or val tokens.")
parser.add_argument("--split", type=str, default="train", choices=["train", "val"], help="Select 'train' or 'val' token set")
args = parser.parse_args()

data = pickle.load(open('/mnt/nas-data-1/zhanglingjun.zlj1/data/nuscene/data/cached_nuscenes_info.pkl', 'rb'))
split = json.load(open('/mnt/nas-data-1/zhanglingjun.zlj1/MindDriver/create_data/full_split.json', 'r'))
tokens = split[args.split]

num_train_samples = len(tokens)
train_ratio = 1

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
num_language_tokens = 0
num_user_tokens = 0
num_assistant_tokens = 0
traj_only = True

with open('/mnt/nas-data-1/zhanglingjun.zlj1/ad_data_process/sft_data_api_explain/LLaMA-Factory-data/train_api_split.json', 'r') as file:
    full_train_token_prompt = json.load(file)
with open('/mnt/nas-data-1/zhanglingjun.zlj1/result_qwen_2_5_72b.json', 'r') as file:
    merged_data = json.load(file)

train_messages = []
dataroot = '/mnt/nas-data-1/zhanglingjun.zlj1/MindDriver/LLaMA-Factory/data/nuscenes'
nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
can_bus = NuScenesCanBus(dataroot='/mnt/nas-data-5/yuanyujian.yyj/data')
sft_indices = json.load(open('/mnt/nas-data-1/zhanglingjun.zlj1/MindDriver/MoVQGAN/gt_indices_sft.json'))

for tokeni,token in enumerate(merged_data.keys()):
    merged_item = merged_data[token]

    for item in full_train_token_prompt:
        if item['id'] == token:
                 = item['prompt']
            prompt1 = "1.Objective: You are controlling an autonomous vehicle in a complex urban traffic environment. Currently available inputs include: six camera view images, the ego vehicle's trajectory over the past 2 seconds, and historical images from the <CAM_FRONT> camera over the past 2 seconds. Your task is to generate a predicted front-view image for the next 0.5 seconds based on the navigation goal and perceptual information, and to plan a safe and reasonable trajectory for the ego vehicle over the next 3 seconds./n 2.Use the following standard format:<think>[Reasoning text][Image]</think>\n<answer>[Waypoints]</answer>"

            images_path = merged_item['images_path']
            reversed_last_four = images_path[-1:-5:-1]
            first_six = images_path[:-4]
            new_images_path = reversed_last_four + first_six
            assitant_message = item['traj']
            break

    sample = nusc.get('sample',token)
    sample_data_front = nusc.get('sample_data',sample['data']['CAM_FRONT'])
    sample_data_front_path = sample_data_front['filename']
    base_path = Path('/mnt/nas-data-5/yuanyujian.yyj/data/nuscenes/')
    full_path1 = base_path / sample_data_front_path

    scene = nusc.get('scene', sample['scene_token'])
    scene_name = scene['name']
    vehicle_msgs = can_bus.get_messages(scene_name, 'vehicle_monitor')
    accmsgs = can_bus.get_messages(scene_name, 'pose')
    sd_token = sample['data']['LIDAR_TOP']
    sample_timestamp = nusc.get('sample_data', sd_token)['timestamp']
    msg = min(
        vehicle_msgs,
        key=lambda m: abs(m['utime'] - sample_timestamp)
    )
    ego_speed = str(round(msg['vehicle_speed']/3.6,2))
    acc = min(
        accmsgs,
        key=lambda m: abs(m['utime'] - sample_timestamp)
    )
    acc_info = str(round(acc['accel'][0],2))

    try:
        next_token=sample['next']

        next_img_token=sft_indices[next_token]['CAM_FRONT']
        next_img_token = str(next_img_token).replace(" ", "")
        numbers = next_img_token.strip('[]').split(',')
        next_img_token = ''.join([f'<|{num}|>' for num in numbers])


    except:
        print("failed")
        continue
    answer = merged_data[token]['result']
    
    train_data_one = {
                    "messages": [
                        {
                            "content": prompt+"The current longitudinal velocity of the vehicle is"+ego_speed+" m/s\n"+"The current longitudinal acceleration is"+acc_info+" m/s^2\n"+"These are the images from the vehicle's CAM_FRONT over the past 2.0s<image>\n past 1.5s<image>\n past 1.0s<image>\n past 0.5s<image>\n"+"These are the six-view images of the vehicle for the current frame: CAM_FRONT:<image>\nCAM_FRONT_LEFT:<image>\nCAM_FRONT_RIGHT:<image>\nCAM_BACK:<image>\nCAM_BACK_LEFT:<image>\nCAM_BACK_RIGHT:<image>\n"+prompt1,
                            "role": "user"
                        },
                        {
                            "content": "<think>"+ answer + "\n This is the front-view image for the future 0.5s."+ next_img_token + "</think>" +"\n"+'<answer>'+ assitant_message+"</answer>",
                            "role": "assistant"
                        }
                    ],
                    "images": new_images_path,
                    "id": token

                }

    train_messages.append(train_data_one)


with open(f"./{args.split}_final.json", "w") as f:
    json.dump(train_messages, f, ensure_ascii=False, indent=4)

