import pickle
import re
import json
import argparse
import tiktoken
from nuscenes.nuscenes import NuScenes
from prompt_message import  generate_user_message, generate_assistant_message, generate_goal_message

# system="You're an autonomous vehicle's brain. Coordinates: X-axis is perpendicular, and Y-axis is parallel to the direction you're facing. You're at point (0,0). Units: meters. Based on the provided particulars, please output the CAM_FRONT image at the 0.5 second in the future and plan waypoints (0.5s intervals) for the next 3 seconds."
text_prompt_half =("""
1.Objective:
You are controlling an autonomous vehicle in a complex urban traffic environment with multiple camera views. Your task is to plan a reasonable and safe driving trajectory for the next few seconds.
2.Process Overview:
This process provides the steps for final decision-making reasoning for the autonomous vehicle. You need to follow this sequence for reasoning:
a. Scene Description: Describe the traffic conditions, including relevant environmental cues such as traffic lights, lane markings, and the behavior of surrounding vehicles or pedestrians.
b. Identification of Important Traffic Participants and Obstacles: Identify two to three key road users or obstacles, specify their relative position to the self-vehicle, and infer their possible motion state.
c. Infer Autonomous Driving Behavior: Based on the observed environment and current vehicle state, deduce the intended actions of the self-vehicle.
d. Summarize the Inference Results: Output the results in a standard format.
3.Elements of Scene Description:
Each scene description must include:
Current weather and road conditions, and analyze their impact on visibility and braking distance, such as night, low visibility, or presence of obstructions.
Identify the traffic light most relevant to the current vehicle direction and analyze how its signal state affects the current decision. The traffic light status should be selected from [Red, Yellow, Green, and Uncertain].
4.Identification of Important Traffic Participants and Obstacles:
Evaluate the distance of surrounding objects and potential collision risks. Based on actual visual information from images, determine the relative distance of objects such as vehicles, motorcycles, or pedestrians in front of the self-vehicle, and classify the risk level based on safety distance standards.
Predict the motion state of detected objects, prioritizing the analysis of high-risk objects.
5.Infer Autonomous Driving Behavior:
Prioritize avoiding collisions in the self-vehicle, then ensure driving within the drivable area. If collision avoidance is necessary, you may change lanes.
When an object is very close to the self-vehicle, prioritize deceleration or emergency braking to avoid collision risks. If dynamic objects are extremely near, immediately engage in risk avoidance processes rather than just allowing reaction time.
Prioritize ensuring the vehicle does not collide. If collision avoidance is necessary, the vehicle may leave the current lane and choose [Move Forward to the Left] or [Move Forward to the Right].
6.Summarize the Inference Results:
Combine the above information to logically deduce the best driving decision:
Lateral Action (choose one): [Go Straight, Move Forward to the Left, Move Forward to the Right, Turn Left, Change Lane to the Left, Turn Right, Change Lane to the Right]
Longitudinal Action (choose one): [Stop, Decelerate to Zero, Maintain Constant Speed, Rapid Deceleration, Decelerate, Rapid Acceleration, Accelerate]
Example:
a. Scene Description:
<Description>
b. Identification of Important Traffic Participants and Obstacles:
<CAM_FRONT>: ...
c. Infer Autonomous Driving Behavior:
<To ensure collision avoidance...>
d. Summarize the Inference Results:
In summary, the optimal driving decision is *Go Straight and *Decelerate to Zero.""")

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

dataroot = './LLaMA-Factory/data/nuscenes'
nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
sft_indices = json.load(open('./MoVQGAN/gt_indices_sft.json'))
train_messages = []

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
                        "system": text_prompt_half,
                        "conversations": [
                            {
                                "from": "human",
                                "value": "Here are current six images from the car: 'CAM_FRONT': <image>\n,'CAM_FRONT_LEFT': <image>\n, 'CAM_FRONT_RIGHT': <image>\n,'CAM_BACK': <image>\n,'CAM_BACK_LEFT': <image>\n, 'CAM_BACK_RIGHT': <image>\n Your current driving instruction is " + goal + ". Based on the provided particulars, please deduce the optimal driving decision and Summarize the Inference Results. Please ensure to answer in English.\n"                                
                            },
                            {
                                "from": "gpt",                                
                                "value": " These are the visual tokens of CAM_FRONT image at the 0.5 second in the future. \n" + assitant_message + " These are the future waypoints. \n " 
                            },                    
                        ]
                    }
    train_messages.append(train_message)

print("#### Cost Summarization ####")
print(f"Number of user tokens: {num_user_tokens}")
print(f"Number of assistant tokens: {num_assistant_tokens}")
print(f"Number of total tokens: {num_language_tokens}")


# with open(f"./LLaMA-Factory/data/{args.split}_cot_motion.json", "w") as f:
#     json.dump(train_messages, f, indent=4)
with open(f"./LLaMA-Factory/data/{args.split}_text_cot_motion.json", "w") as f:
    json.dump(train_messages, f, indent=4)


# python tools/evaluation/evaluation.py \
# --metric uniad \
# --result_file /mnt/nas-data-1/zhanglingjun.zlj1/FSDrive-main/LLaMA-Factory/data/eval_traj.json

# python scripts/vllm_infer.py \
# --model_name_or_path /mnt/nas-data-5/yuanyujian.yyj/checkpoints_fsdrive_sft \
# --dataset val_cot_motion \
# --template qwen2_vl \
# --cutoff_len 32768 \
# --max_new_tokens 2048 \
# --max_samples 100000 \
# --image_resolution 524288 \
# --save_name results.jsonl \
# --temperature 0.1 \
# --top_p 0.1 \
# --top_k 10

# python tools/visualization/visualize_planning.py \
# --pred-trajs-path /mnt/nas-data-1/zhanglingjun.zlj1/FSDrive-main/LLaMA-Factory/data/generated_predictions.jsonl \
# --tokens-path /mnt/nas-data-1/zhanglingjun.zlj1/FSDrive-main/LLaMA-Factory/data/eval_traj.json \
# --output-path ./vis_traj_gt
#  you'll need:\nHistorical Trajectory (last 2 seconds): [(-0.11,-11.70), (-0.12,-9.12), (-0.10,-6.23), (-0.05,-3.06)]\nMission Goal: FORWARD\nTraffic Rules: Avoid collision with other objects.\n- Always drive on drivable regions.\n- Avoid driving on occupied regions.


# python tools/visualization/visualize_planning.py \
# --pred-trajs-path /mnt/nas-data-1/zhanglingjun.zlj1/FSDrive-main/LLaMA-Factory/data/generated_predictions.jsonl \
# --tokens-path /mnt/nas-data-1/zhanglingjun.zlj1/img/train.json \
# --output-path ./vis_traj_six_one
#  /mnt/nas-data-1/zhanglingjun.zlj1/img/train.json
