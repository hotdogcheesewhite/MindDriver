import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor,AutoModelForCausalLM
from vispro import process_vision_info
import pickle
import re
import json
import argparse
import tiktoken
import torch
from nuscenes.nuscenes import NuScenes
from prompt_message import  generate_user_message, generate_assistant_message, generate_goal_message
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
# default: Load the model on the available device(s)
# model_path = "/mnt/nas-data-1/zhanglingjun.zlj1/qwen25-vl"

model_path = "/mnt/nas-data-3/wanjiaxu.wjx/code/embody/llm/Qwen2.5-VL/Qwen2.5-VL-72B-Instruct"
with init_empty_weights():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,torch_dtype=torch.bfloat16)
no_split_modules = model._no_split_modules
print(f"no_split_modules: {no_split_modules}")

map_list = {
    0: "60GB",  # GPU 0 最多用 40GB 显存
    1: "60GB",  # GPU 1 最多用 40GB 显存
    2: "60GB",
    3: "60GB",
    4: "60GB",  # GPU 0 最多用 40GB 显存
    5: "60GB",  # GPU 1 最多用 40GB 显存
    6: "60GB",
    7: "60GB",
}
device_map = infer_auto_device_map(model,max_memory=map_list,no_split_module_classes=no_split_modules)
model = load_checkpoint_and_dispatch(model, checkpoint=model_path, device_map=device_map)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,torch_dtype="auto",max_memory=max_memory,device_map="auto")
# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     model_path,
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="cuda:1",
# )

# default processor
processor = AutoProcessor.from_pretrained(model_path)


# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
text_prompt_half_chinese =("""
1. 目标:
你正在复杂的城市交通环境中控制一辆自动驾驶车辆，当前拥有几个摄像头视角。你的任务是规划未来几秒内合理且安全的行驶轨迹
2. 过程概述:
该过程提供了自动驾驶车辆进行最终决策推理的步骤，你需要按照这样的顺序进行推理：
a. 场景描述: 描述交通状况，包括相关的环境线索，如交通灯、车道标记，以及周围车辆或行人的行为。
b. 重要交通参与者与障碍物识别: 识别两到三个关键的道路使用者或障碍物，说明他们与自我车辆的相对位置，以及可能的运动状态。
c. 推理自动驾驶行为: 根据观察到的环境和当前车辆状态，推断自我车辆的期望意图。
d. 总结推理结果: 输出具有标准格式的结果。
3. 场景描述要素:
每个场景描述需要包括：
当前的天气和路面状况，并分析其对视野和制动距离的影响，如黑夜、能见度低、存在遮挡等情况。
判断与当前车辆行驶方向最密切的红绿灯，并分析其信号状态对当前决策的影响，红绿灯的状态应该从[红灯、黄灯、绿灯和不确定]中挑选一个
4. 重要交通参与者与障碍物识别:
评估周边物体的距离与潜在碰撞风险。请根据图像实际视觉信息，判断前方车辆、摩托车、行人等物体与本车的相对距离，并结合安全车距标准进行风险等级划分。
对预测出的物体进行运动状态预测，优先考虑风险等级高的物体进行分析
5. 推理自动驾驶行为:
优先保证自车不发生碰撞，再保证行驶在可驾驶区域内，若需要避撞，可以离开当前车道，选择[向左前方前进]或[向右前方前进]。
当发现有物体距离本车非常近，必须优先考虑减速或紧急刹车以避免碰撞风险。动态物体距离本车极近时，应立即进入风险规避流程，而不是仅预留反应时间。
如果有锥桶等施工标志场景，需要认定为道路变窄，采用合理动作避免碰撞。
6. 总结推理结果:
综合以上信息，通过逻辑推理得出最佳的驾驶决策：
横向动作（只选一个）：[前进，向左前方前进，向右前方前进，左转，变车道到左，右转，变车道到右]
纵向动作（只选一个）：[停止，减速到零，保持恒定速度，快速减速，减速，快速加速，加速]
示例：
a. 场景描述:
<描述>
b. 重要交通参与者与障碍物识别:
<CAM_FRONT>：……
c. 推理自动驾驶行为:
<为保证避免碰撞……>
d. 总结推理结果:
综上，最终可采取的最佳驾驶决策为，前进，保持恒定速度
""")
# text_prompt_half =("""
# 1.Objective:
# You are controlling an autonomous vehicle in a complex urban traffic environment with multiple camera views. Your task is to plan a reasonable and safe driving trajectory for the next few seconds.
# 2.Process Overview:
# This process provides the steps for final decision-making reasoning for the autonomous vehicle. You need to follow this sequence for reasoning:
# a. Scene Description: Describe the traffic conditions, including relevant environmental cues such as traffic lights, lane markings, and the behavior of surrounding vehicles or pedestrians.
# b. Identification of Important Traffic Participants and Obstacles: Identify two to three key road users or obstacles, specify their relative position to the self-vehicle, and infer their possible motion state.
# c. Infer Autonomous Driving Behavior: Based on the observed environment and current vehicle state, deduce the intended actions of the self-vehicle.
# d. Summarize the Inference Results: Output the results in a standard format.
# 3.Elements of Scene Description:
# Each scene description must include:
# Current weather and road conditions, and analyze their impact on visibility and braking distance, such as night, low visibility, or presence of obstructions.
# Identify the traffic light most relevant to the current vehicle direction and analyze how its signal state affects the current decision. The traffic light status should be selected from [Red, Yellow, Green, and Uncertain].
# 4.Identification of Important Traffic Participants and Obstacles:
# Evaluate the distance of surrounding objects and potential collision risks. Based on actual visual information from images, determine the relative distance of objects such as vehicles, motorcycles, or pedestrians in front of the self-vehicle, and classify the risk level based on safety distance standards.
# Predict the motion state of detected objects, prioritizing the analysis of high-risk objects.
# 5.Infer Autonomous Driving Behavior:
# Prioritize avoiding collisions in the self-vehicle, then ensure driving within the drivable area. If collision avoidance is necessary, you may change lanes.
# When an object is very close to the self-vehicle, prioritize deceleration or emergency braking to avoid collision risks. If dynamic objects are extremely near, immediately engage in risk avoidance processes rather than just allowing reaction time.
# Prioritize ensuring the vehicle does not collide. If collision avoidance is necessary, the vehicle may leave the current lane and choose [Move Forward to the Left] or [Move Forward to the Right].
# 6.Summarize the Inference Results:
# Combine the above information to logically deduce the best driving decision:
# Lateral Action (choose one): [Go Straight, Move Forward to the Left, Move Forward to the Right, Turn Left, Change Lane to the Left, Turn Right, Change Lane to the Right]
# Longitudinal Action (choose one): [Stop, Decelerate to Zero, Maintain Constant Speed, Rapid Deceleration, Decelerate, Rapid Acceleration, Accelerate]
# Example:
# a. Scene Description:
# <Description>
# b. Identification of Important Traffic Participants and Obstacles:
# <CAM_FRONT>: ...
# c. Infer Autonomous Driving Behavior:
# <To ensure collision avoidance...>
# d. Summarize the Inference Results:
# In summary, the optimal driving decision is *Go Straight and *Decelerate to Zero.""")

parser = argparse.ArgumentParser(description="Choose to use train or val tokens.")
parser.add_argument("--split", type=str, default="train", choices=["train", "val"], help="Select 'train' or 'val' token set")
args = parser.parse_args()

data = pickle.load(open('/mnt/nas-data-1/zhanglingjun.zlj1/data/nuscene/data/cached_nuscenes_info.pkl', 'rb'))
split = json.load(open('/mnt/nas-data-1/zhanglingjun.zlj1/FSDrive-main/create_data/splits/split_3.json', 'r'))
prefix_path = "/mnt/nas-data-1/zhanglingjun.zlj1/FSDrive-main/LLaMA-Factory/"
tokens = split[args.split]

output_file = 'xingyun_3.json'
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as json_file:
        results = json.load(json_file)
else:
    results = {}

num_train_samples = len(tokens)
train_ratio = 1

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
num_language_tokens = 0
num_user_tokens = 0
num_assistant_tokens = 0
traj_only = True

# dataroot = './LLaMA-Factory/data/nuscenes'
dataroot ="/mnt/nas-data-1/zhanglingjun.zlj1/FSDrive-main/LLaMA-Factory/data/nuscenes"
nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)

train_messages = []
for token_i, token in enumerate(tokens):
    if token in results:
        print(f"Skipping already processed token: {token}")
        continue
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

    if STOP:
        goal = "STOP"
    
    # text_prompt = text_prompt_half + "Here are current six images from the car: 'CAM_FRONT': <image>\n,'CAM_FRONT_LEFT': <image>\n, 'CAM_FRONT_RIGHT': <image>\n,'CAM_BACK': <image>\n,'CAM_BACK_LEFT': <image>\n, 'CAM_BACK_RIGHT': <image>\n Your current driving instruction is " + goal + ". Based on the provided particulars, please deduce the optimal driving decision and Summarize the Inference Results. Please ensure to answer in English.\n"
    text_prompt = text_prompt_half_chinese + "这是该车当前帧的六视图: 'CAM_FRONT': <image>\n,'CAM_FRONT_LEFT': <image>\n, 'CAM_FRONT_RIGHT': <image>\n,'CAM_BACK': <image>\n,'CAM_BACK_LEFT': <image>\n, 'CAM_BACK_RIGHT': <image>\n 你的驾驶指令是 " + goal + ". 基于以上信息，推理出最佳的横向动作与纵向动作的拼接结果.\n"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": prefix_path+images_path[0],
                },
                {
                    "type": "image",
                    "image": prefix_path+images_path[1],
                },
                {
                    "type": "image",
                    "image": prefix_path+images_path[2],
                },
                {
                    "type": "image",
                    "image": prefix_path+images_path[3],
                },
                {
                    "type": "image",
                    "image": prefix_path+images_path[4],
                },
                {
                    "type": "image",
                    "image": prefix_path+images_path[5],
                },
                {"type": "text", "text": text_prompt},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    # inputs = accelerator.prepare(inputs)
    inputs = inputs.to(model.device)
    

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    results[token] = {
            "token": token,
            "images_path": images_path,
            "result": output_text
        }
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)
    print(output_text)

# import pickle
# import re
# import json
# import argparse
# import tiktoken
# from nuscenes.nuscenes import NuScenes
# from prompt_message import  generate_user_message, generate_assistant_message, generate_goal_message

# # system="You're an autonomous vehicle's brain. Coordinates: X-axis is perpendicular, and Y-axis is parallel to the direction you're facing. You're at point (0,0). Units: meters. Based on the provided particulars, please output the CAM_FRONT image at the 0.5 second in the future and plan waypoints (0.5s intervals) for the next 3 seconds."
# text_prompt_half =("""
# 1.Objective:
# You are controlling an autonomous vehicle in a complex urban traffic environment with multiple camera views. Your task is to plan a reasonable and safe driving trajectory for the next few seconds.
# 2.Process Overview:
# This process provides the steps for final decision-making reasoning for the autonomous vehicle. You need to follow this sequence for reasoning:
# a. Scene Description: Describe the traffic conditions, including relevant environmental cues such as traffic lights, lane markings, and the behavior of surrounding vehicles or pedestrians.
# b. Identification of Important Traffic Participants and Obstacles: Identify two to three key road users or obstacles, specify their relative position to the self-vehicle, and infer their possible motion state.
# c. Infer Autonomous Driving Behavior: Based on the observed environment and current vehicle state, deduce the intended actions of the self-vehicle.
# d. Summarize the Inference Results: Output the results in a standard format.
# 3.Elements of Scene Description:
# Each scene description must include:
# Current weather and road conditions, and analyze their impact on visibility and braking distance, such as night, low visibility, or presence of obstructions.
# Identify the traffic light most relevant to the current vehicle direction and analyze how its signal state affects the current decision. The traffic light status should be selected from [Red, Yellow, Green, and Uncertain].
# 4.Identification of Important Traffic Participants and Obstacles:
# Evaluate the distance of surrounding objects and potential collision risks. Based on actual visual information from images, determine the relative distance of objects such as vehicles, motorcycles, or pedestrians in front of the self-vehicle, and classify the risk level based on safety distance standards.
# Predict the motion state of detected objects, prioritizing the analysis of high-risk objects.
# 5.Infer Autonomous Driving Behavior:
# Prioritize avoiding collisions in the self-vehicle, then ensure driving within the drivable area. If collision avoidance is necessary, you may change lanes.
# When an object is very close to the self-vehicle, prioritize deceleration or emergency braking to avoid collision risks. If dynamic objects are extremely near, immediately engage in risk avoidance processes rather than just allowing reaction time.
# Prioritize ensuring the vehicle does not collide. If collision avoidance is necessary, the vehicle may leave the current lane and choose [Move Forward to the Left] or [Move Forward to the Right].
# 6.Summarize the Inference Results:
# Combine the above information to logically deduce the best driving decision:
# Lateral Action (choose one): [Go Straight, Move Forward to the Left, Move Forward to the Right, Turn Left, Change Lane to the Left, Turn Right, Change Lane to the Right]
# Longitudinal Action (choose one): [Stop, Decelerate to Zero, Maintain Constant Speed, Rapid Deceleration, Decelerate, Rapid Acceleration, Accelerate]
# Example:
# a. Scene Description:
# <Description>
# b. Identification of Important Traffic Participants and Obstacles:
# <CAM_FRONT>: ...
# c. Infer Autonomous Driving Behavior:
# <To ensure collision avoidance...>
# d. Summarize the Inference Results:
# In summary, the optimal driving decision is *Go Straight and *Decelerate to Zero.""")

# parser = argparse.ArgumentParser(description="Choose to use train or val tokens.")
# parser.add_argument("--split", type=str, default="train", choices=["train", "val"], help="Select 'train' or 'val' token set")
# args = parser.parse_args()

# data = pickle.load(open('/mnt/nas-data-1/zhanglingjun.zlj1/data/nuscene/data/cached_nuscenes_info.pkl', 'rb'))
# split = json.load(open('/mnt/nas-data-1/zhanglingjun.zlj1/FSDrive-main/create_data/full_split.json', 'r'))
# explain = json.load(open('/mnt/nas-data-1/zhanglingjun.zlj1/0818zlj.json', 'r'))
# tokens = split[args.split]

# num_train_samples = len(tokens)
# train_ratio = 1

# encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
# num_language_tokens = 0
# num_user_tokens = 0
# num_assistant_tokens = 0
# traj_only = True

# dataroot = './LLaMA-Factory/data/nuscenes'
# nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
# sft_indices = json.load(open('./MoVQGAN/gt_indices_sft.json'))
# train_messages = []

# valid_tokens = [token for token in tokens if token in explain]

# for token_i, token in enumerate(valid_tokens):
#     if token_i >= train_ratio * num_train_samples:
#         break 
#     assitant_message, STOP = generate_assistant_message(data, token, traj_only=traj_only)
#     user_message, images_path = generate_user_message(data, token)
#     goal = generate_goal_message(data,token)

#     if len(assitant_message.split("\n")) > 6:
#         print()
#         print(token)
#         print(user_message)
#         print(assitant_message)
#     num_language_tokens += len(encoding.encode(user_message))
#     num_user_tokens += len(encoding.encode(user_message))
#     num_language_tokens += len(encoding.encode(assitant_message))
#     num_assistant_tokens += len(encoding.encode(assitant_message))

#     try:
#         next_token=nusc.get('sample', token)['next']
#         next_img_token=sft_indices[next_token]['CAM_FRONT']
#         next_img_token = str(next_img_token).replace(" ", "")
#         numbers = next_img_token.strip('[]').split(',')
#         next_img_token = ''.join([f'<|{num}|>' for num in numbers])
#     except:
#         continue

#     if STOP:
#         goal = "STOP"
    

#     result_content = explain[token]["result"]
#     match = re.search(r'"content":"(.*?)",\"tool_calls', result_content)
#     match_content = match.group(1)

#     train_message = {
#                         "id": token,
#                         "images": images_path,
#                         "system": text_prompt_half,
#                         "conversations": [
#                             {
#                                 "from": "human",
#                                 "value": "Here are current six images from the car: 'CAM_FRONT': <image>\n,'CAM_FRONT_LEFT': <image>\n, 'CAM_FRONT_RIGHT': <image>\n,'CAM_BACK': <image>\n,'CAM_BACK_LEFT': <image>\n, 'CAM_BACK_RIGHT': <image>\n Your current driving instruction is " + goal + ". Based on the provided particulars, please deduce the optimal driving decision and Summarize the Inference Results. Finally, plan waypoints (0.5s intervals) for the next 3 seconds.\nPlease ensure to answer in English.\n"                                
#                             },
#                             {
#                                 "from": "gpt",                                
#                                 "value": match_content + assitant_message + " These are the future waypoints. \n " 
#                             },                    
#                         ]
#                     }
#     train_messages.append(train_message)

# print("#### Cost Summarization ####")
# print(f"Number of user tokens: {num_user_tokens}")
# print(f"Number of assistant tokens: {num_assistant_tokens}")
# print(f"Number of total tokens: {num_language_tokens}")


# # with open(f"./LLaMA-Factory/data/{args.split}_cot_motion.json", "w") as f:
# #     json.dump(train_messages, f, indent=4)
# with open(f"./LLaMA-Factory/data/{args.split}_text_cot_plan_motion.json", "w") as f:
#     json.dump(train_messages, f, indent=4)