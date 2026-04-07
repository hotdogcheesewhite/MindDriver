import subprocess
import time
from nuscenes.nuscenes import NuScenes
import os
import pickle
import json


def call_multi_image_api(text_prompt,images):
    try:
        print(f"process {len(images)}")
        result = subprocess.run(
            ["/mnt/nas-data-1/zhanglingjun.zlj1/ad_data_process/api_call/api_his.sh", text_prompt] + images,  # text_prompt
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
        print("TIMEOUT")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None

if __name__ == "__main__":
    with open('/mnt/nas-data-1/zhanglingjun.zlj1/train_api_split_0822.json', 'r') as file:
        data = json.load(file)

    output_file = '../result_qwen_2_5_72b.json'
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    else:
        results = {}
    for item in data:
        id = item['id']
        if id in results:
            print(f"Token {id} already processed, skipping...")
            continue
        images_path = item['images']
        user_content = item['prompt']
        # import pdb; pdb.set_trace()
        start_time = time.time()
        response = call_multi_image_api(user_content,images_path)
        result_json = json.loads(response)
        content = result_json['choices'][0]['message']['content']
        end_time = time.time()
        results[id] = {
            "token": id,
            "images_path": images_path,
            "result": content
        }
        with open(output_file, 'w', encoding='utf-8') as json_file:
            json.dump(results, json_file, ensure_ascii=False, indent=4)
        print(f"Timecost: {end_time - start_time:.2f}秒")
        if response:
            print("\nDone!")
        else:
            print("\nFailed!")