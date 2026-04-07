import os
import re
import json
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import argparse
import torchvision.transforms as T
from torch.nn.functional import mse_loss, l1_loss
from nuscenes.nuscenes import NuScenes
from movqgan import get_movqgan_model
from itertools import islice

def read_json_or_jsonl(file_path):

    with open(file_path, 'r', encoding='utf-8') as f:
        first_char = None
        for line in f:
            stripped = line.strip()
            if stripped:
                first_char = stripped[0]
                f.seek(0)  # 回到开头
                break

        if first_char == '[':
            # 整个文件是一个 JSON 数组
            data = json.load(f)  # 直接加载整个列表
            if isinstance(data, list):
                for item in data:
                    yield item
            else:
                raise ValueError("JSON file starts with '[' but is not a valid list.")
        else:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

def show_images(batch, file_path):
    """ Display a batch of images inline. """
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(torch.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    image = Image.fromarray(reshaped.numpy())
    image.save(file_path)


device = torch.device('cuda:1')
model = get_movqgan_model('270M', ckpt_path='/mnt/nas-data-1/zhanglingjun.zlj1/data/MoVQGAN/movqgan_270M.ckpt', device=device)

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--input_json', type=str, required=True, help='Path to input JSON file')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output images')
args = parser.parse_args()

idxs_path = args.input_json
save_dir = args.output_dir
os.makedirs(save_dir, exist_ok=True)

# idxs = json.load(open(idxs_path, "r"))
# for key, value in islice(idxs.items(), 10):
token = 0
for item in list(read_json_or_jsonl(idxs_path)):
    # try:
    predict_text = item["predict"]
    # token = item["id"]
    idx = predict_text
    numbers = re.findall(r'<\|(\d+)\|>', idx)
    idx = [int(num) for num in numbers]
    idx = torch.tensor(idx).to(model.device)
    idx = torch.clamp(idx, min=0, max=16383)
    
    current_length = idx.size(0)
    required_length = 384

    if current_length < required_length:
        pad_length = required_length - current_length
        padding = torch.randint(0, 16384, (pad_length,), device=idx.device, dtype=idx.dtype)
        idx = torch.cat([idx, padding], dim=0)

    with torch.no_grad():
        out = model.decode_code(idx[:required_length].long().unsqueeze(0))#.long()
    

    save_path = os.path.join(save_dir, f"{token}.png")

    show_images(out, save_path)
    token +=1
