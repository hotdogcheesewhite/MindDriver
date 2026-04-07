import pickle
import re
import json
import argparse
import tiktoken
from nuscenes.nuscenes import NuScenes
from prompt_message import  generate_user_message, generate_assistant_message, generate_goal_message

# system="You're an autonomous vehicle's brain. Coordinates: X-axis is perpendicular, and Y-axis is parallel to the direction you're facing. You're at point (0,0). Units: meters. Based on the provided particulars, please output the CAM_FRONT image at the 0.5 second in the future and plan waypoints (0.5s intervals) for the next 3 seconds."
system="You're an autonomous vehicle's brain. Coordinates: X-axis is perpendicular, and Y-axis is parallel to the direction you're facing. You're at point (0,0). Units: meters. Based on the provided particulars, please plan waypoints (0.5s intervals) for the next 3 seconds."


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

user_content  = ("""
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
你当前的驾驶指令是""")

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


    # train_message = {
    #                     "id": token,
    #                     "images": images_path,
    #                     "system": system,
    #                     "conversations": [
    #                         {
    #                             "from": "human",
    #                             "value": "Here are current six images from the car: 'CAM_FRONT': <image>\n,'CAM_FRONT_LEFT': <image>\n, 'CAM_FRONT_RIGHT': <image>\n,'CAM_BACK': <image>\n,'CAM_BACK_LEFT': <image>\n, 'CAM_BACK_RIGHT': <image>\n" + user_message + "Based on the provided particulars, please output explaina and plan waypoints (0.5s intervals) for the next 3 seconds.\n"                                
    #                         },
    #                         {
    #                             "from": "gpt",                                
    #                             "value": next_img_token + " These are the visual tokens of CAM_FRONT image at the 0.5 second in the future. \n" + assitant_message + " These are the future waypoints. \n <|endoftext|><|im_end|>" 
    #                         },                    
    #                     ]
    #                 }
    train_message = {
                        "id": token,
                        "images": images_path,
                        "prompt": user_content + goal+ "请基于驾驶指令，与当前的六环视图，进行推理。",
                        "traj":assitant_message
    }
    train_messages.append(train_message)

print("#### Cost Summarization ####")
print(f"Number of user tokens: {num_user_tokens}")
print(f"Number of assistant tokens: {num_assistant_tokens}")
print(f"Number of total tokens: {num_language_tokens}")


# with open(f"./LLaMA-Factory/data/{args.split}_cot_motion.json", "w") as f:
#     json.dump(train_messages, f, indent=4)
with open(f"./LLaMA-Factory/data/{args.split}_api_split_0822.json", "w") as f:
    json.dump(train_messages, f, ensure_ascii=False, indent=4)