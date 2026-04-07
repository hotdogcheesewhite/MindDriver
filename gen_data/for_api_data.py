import pickle
import re
import json
import argparse
import tiktoken
from nuscenes.nuscenes import NuScenes
from prompt_message import  generate_user_message, generate_assistant_message, generate_goal_message

parser = argparse.ArgumentParser(description="Choose to use train or val tokens.")
parser.add_argument("--split", type=str, default="train", choices=["train", "val"], help="Select 'train' or 'val' token set")
args = parser.parse_args()

data = pickle.load(open('/mnt/nas-data-1/zhanglingjun.zlj1/data/nuscene/data/cached_nuscenes_info.pkl', 'rb'))
split = json.load(open('/mnt/nas-data-1/zhanglingjun.zlj1/FSDrive-main/create_data/full_split.json', 'r'))
tokens = split[args.split]

num_train_samples = len(tokens)
train_ratio = 1

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
num_language_tokens = 0
num_user_tokens = 0
num_assistant_tokens = 0
traj_only = True

dataroot = '/mnt/nas-data-1/zhanglingjun.zlj1/FSDrive-main/LLaMA-Factory/data/nuscenes'
nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
sft_indices = json.load(open('/mnt/nas-data-1/zhanglingjun.zlj1/FSDrive-main/MoVQGAN/gt_indices_sft.json'))
train_messages = []

user_content  = ("""
1. Objective:
You are controlling an autonomous vehicle in a complex urban traffic environment with access to images fro
m six camera perspectives and 2 seconds of historical footage from <CAM_FRONT>. Your task is to plan a s
afe and reasonable driving trajectory for the next 3 seconds based on navigation targets. Navigation targets wi
ll be provided in the following form: [FORWARD] / [RIGHT] / [LEFT] / [STOP], which should guide the pri
oritization of actions.
2. Process Overview:
Follow the steps outlined below for reasoning:
a. [Scene Description]: Describe the weather, road conditions, visibility, and traffic signal states to determine
drivable areas.
b. [Risk Object Identification]: Identify 1-3 objects with the greatest impact on safety, analyze their position
s and motion states, and update drivable areas.
c. [Reasoning Autonomous Driving Behavior]: Propose three reasonable behavior combinations (direction
+ speed) and provide reasons.
d. [Summarizing Reasoning Results]: Select the optimal behavior and output it in standard format.
3. Scene Analysis:
Analyze the weather (sunny/rainy/foggy/night) and the impact of obstructions on visibility.
Determine the traffic signal state most relevant to the current driving direction: [Red Light, Yellow Light, Gre
en Light, Uncertain]. “Most relevant” refers to the traffic light controlling the right of way for the vehicle‘s la
ne; if navigating a right turn, prioritize the right turn signal. Summarize the initial drivable area: [Large, Medi
um, Small, Uncertain].Large: Multi-lane, unobstructed; Medium: Partially restricted; Small: Severely restric
ted; Uncertain: Insufficient visibility.
4. Latent Risk Assessment:
Use the vehicle's forward direction as the reference point, with <CAM_FRONT_LEFT> covering the left
front area and <CAM_FRONT_RIGHT> covering the right front area.
Combine multi-perspective and historical images to identify object categories (e.g., cars, buses, pedestrians,
construction zones) and motion states (e.g., stationary, constant speed, accelerating toward, moving away).
Select 1-3 highest-risk objects (priority: dynamic > static, lateral proximity > distant).
Update drivable areas across perspectives, using the smallest value principle (if any direction is rated
<Small>, the overall drivable area is <Small>).
5. Behavior Reasoning:
Propose three safe and reasonable behavior combinations, each containing:
Direction Change (select one): [Maintain Current Lane, Change Lane Left, Change Lane Right, Turn Left,
Turn Right].
Speed Change (select one): [Smooth Deceleration, Emergency Brake, Maintain Current Speed, Smooth
Acceleration, Stop, Remain Stationary].
Provide reasoning, considering traffic rules, obstacles, signal lights, and navigation targets.
6. Summarizing Reasoning Results:
Select the optimal behavior from the proposed options and present it in the following format:
Self-driving vehicle's optimal direction and speed change: <Direction>, <Speed>.
7. Output Format:
[Scene Analysis]:
<Description Results>, Drivable Area: <Large>
[Latent Risk Assessment]:
<CAM_FRONT>: <Object Category>, <Motion State>
<CAM_FRONT_RIGHT>: <Object Category>, <Motion State>
Combined Drivable Area: <Medium>
[Behavior Reasoning]:
<Maintain Current Lane>, <Smooth Deceleration>, <Reason>
<Change Lane Left>, <Smooth Acceleration>, <Reason>
<Turn Right>, <Smooth Deceleration>, <Reason>
[Action Decision]:
Self-driving vehicle's optimal direction and speed change: <Maintain Current Lane>, <Smooth Deceleration>.""")

for token_i, token in enumerate(tokens):
    if token_i >= train_ratio * num_train_samples:
        break 
    assitant_message, STOP = generate_assistant_message(data, token, traj_only=traj_only)
    user_message, images_path = generate_user_message(data, token)
    goal = generate_goal_message(data,token)

    if len(assitant_message.split("\n")) > 6:
        print()
        print(token)
        print(user_message)
        print(assitant_message)
    num_language_tokens += len(encoding.encode(user_message))
    num_user_tokens += len(encoding.encode(user_message))
    num_language_tokens += len(encoding.encode(assitant_message))
    num_assistant_tokens += len(encoding.encode(assitant_message))

    try:
        next_token=nusc.get('sample', token)['next']
        next_img_token=sft_indices[next_token]['CAM_FRONT']
        next_img_token = str(next_img_token).replace(" ", "")
        numbers = next_img_token.strip('[]').split(',')
        next_img_token = ''.join([f'<|{num}|>' for num in numbers])
    except:
        continue

    if STOP:
        goal = "STOP"

    train_message = {
                        "id": token,
                        "images": images_path,
                        "prompt": user_content + goal+ "Please reason based on the driving instructions and the current six-ring view.",
                        "traj":assitant_message
    }
    train_messages.append(train_message)

print("#### Cost Summarization ####")
print(f"Number of user tokens: {num_user_tokens}")
print(f"Number of assistant tokens: {num_assistant_tokens}")
print(f"Number of total tokens: {num_language_tokens}")



with open(f"./LLaMA-Factory-data/{args.split}_api_split.json", "w") as f:
    json.dump(train_messages, f, ensure_ascii=False, indent=4)