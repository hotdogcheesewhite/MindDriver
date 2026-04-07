from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from vispro import process_vision_info
import pickle
import re
import json
import argparse
import tiktoken
from nuscenes.nuscenes import NuScenes
from prompt_message import  generate_user_message, generate_assistant_message, generate_goal_message
from accelerate import Accelerator
accelerator = Accelerator()
# default: Load the model on the available device(s)
model_path = "/mnt/nas-data-1/zhanglingjun.zlj1/qwen25-vl"
# model_path = "/mnt/nas-data-3/wanjiaxu.wjx/code/embody/llm/Qwen2.5-VL/Qwen2.5-VL-72B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="cuda:6"
)
model = accelerator.prepare(model)
# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processor
processor = AutoProcessor.from_pretrained(model_path)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
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
# prefix_path = "/mnt/nas-data-1/zhanglingjun.zlj1/FSDrive-main/LLaMA-Factory/"
prefix_path = "/mnt/nas-data-5/yuanyujian.yyj/"
tokens = split[args.split]

output_file = 'pure_text_cot_0819.json'
results = {}

num_train_samples = len(tokens)
train_ratio = 1

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
num_language_tokens = 0
num_user_tokens = 0
num_assistant_tokens = 0
traj_only = True

# dataroot = './LLaMA-Factory/data/nuscenes'
# dataroot ="/mnt/nas-data-1/zhanglingjun.zlj1/FSDrive-main/LLaMA-Factory/data/nuscenes"
dataroot = "/mnt/nas-data-5/yuanyujian.yyj/data/nuscenes"
nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)


train_messages = []
for token_i, token in enumerate(tokens):
    if token_i >= train_ratio * num_train_samples:
        break 
    assitant_message, STOP = generate_assistant_message(data, token, traj_only=traj_only)
    user_message, images_path = generate_user_message(data, token)
    goal = generate_goal_message(data,token)
    # import pdb;pdb.set_trace()

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
    
    text_prompt = text_prompt_half + "Here are current six images from the car: 'CAM_FRONT': <image>\n,'CAM_FRONT_LEFT': <image>\n, 'CAM_FRONT_RIGHT': <image>\n,'CAM_BACK': <image>\n,'CAM_BACK_LEFT': <image>\n, 'CAM_BACK_RIGHT': <image>\n Your current driving instruction is " + goal + ". Based on the provided particulars, please deduce the optimal driving decision and Summarize the Inference Results. Please ensure to answer in English.\n"
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

    # import pdb;pdb.set_trace()

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
    inputs = inputs.to(model.device)

    import pdb;pdb.set_trace()

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=8192)
    import pdb;pdb.set_trace()
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    results[token] = {
            "token": token,
            "images_path": prefix_path+images_path,
            "result": output_text
        }
    import pdb;pdb.set_trace()
    with open(output_file, 'w') as json_file:
        json.dump(results, json_file, indent=4)
    print(output_text)