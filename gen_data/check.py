import subprocess
import time
from nuscenes.nuscenes import NuScenes
import os
import pickle
import json


def call_multi_image_api(text_prompt):
    try:
        print(f"质检开始")
        result = subprocess.run(
            ["./api_check.sh", text_prompt],  
            capture_output=True,
            text=True,
            timeout=1200  
        )
        if result.returncode == 0:
            print("=" * 60)
            print(result.stdout)
            return result.stdout
        else:
            print(f"return: {result.returncode}")
            print(f"error: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        print("timeout")
        return None
    except Exception as e:
        print(f"error: {e}")
        return None

if __name__ == "__main__":
    with open('/mnt/nas-data-1/zhanglingjun.zlj1/ad_data_process/v3_check_train_json_files/split_1.json', 'r') as file:
        data = json.load(file)
    output_file = '../v3_0903_result_check/check_right_or_wrong5.json'
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    else:
        results = {}


    for item in data:
        token = item['token']


        text = "1. Objective: You are an expert in detecting the quality of reasoning chains for autonomous driving models, tasked with determining whether the input reasoning process is correct or flawed. 2. Output Format: Reason + [Correct/Incorrect] 3. Judgment Criteria: Determine whether the input reasoning chain is logically correct and free from logical flaws. At the end of your response, specify the optimal direction and speed change for the ego vehicle as: <Direction Change>, <Speed Change>. Direction Change (select one): [Maintain Current Lane, Change Lane Left, Change Lane Right, Turn Left, Turn Right]. Speed Change (select one): [Smooth Deceleration, Emergency Brake, Maintain Current Speed, Smooth Acceleration, Stop, Remain Stationary].4. Input reasoning process:"
        user_content = item['result']
        text1 = "Please evaluate the quality of this response according to the template."
        # import pdb; pdb.set_trace()
        start_time = time.time()
        user_content = text + user_content + text1
        response = call_multi_image_api(user_content)
        result_json = json.loads(response)
        content = result_json['choices'][0]['message']['content']
        end_time = time.time()
        results[token] = {
            "token": token,
            "result": content
        }
        with open(output_file, 'w', encoding='utf-8') as json_file:
            json.dump(results, json_file, ensure_ascii=False, indent=4)
        print(f"API time: {end_time - start_time:.2f}秒")
        if response:
            print("\nDone!")
        else:
            print("\nFailed!")