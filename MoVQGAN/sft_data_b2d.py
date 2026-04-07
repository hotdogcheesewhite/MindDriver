import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from movqgan import get_movqgan_model

def prepare_image(img):
    """ Transform and normalize PIL Image to tensor. """
    transform = T.Compose([
        T.Resize((128, 192), interpolation=T.InterpolationMode.BICUBIC),
    ])
    pil_image = transform(img)
    arr = np.array(pil_image.convert("RGB"))
    arr = arr.astype(np.float32) / 127.5 - 1
    
    return torch.from_numpy(np.transpose(arr, [2, 0, 1]))

# 设置设备和模型
device = torch.device('cuda:5')
model = get_movqgan_model('270M', ckpt_path='/mnt/nas-data-1/zhanglingjun.zlj1/data/MoVQGAN/movqgan_270M.ckpt', device=device)

# 读取您的JSON文件
json_path = '/mnt/nas-data-1/zhanglingjun.zlj1/data/reasonplan/train_final_modified.json'
with open(json_path, 'r') as f:
    data = json.load(f)

# 准备输出字典
gt_indices = {}

# 处理每个图像对
for key, image_path in tqdm(data.items(), desc="Processing images"):
    try:
        # 打开并预处理图像
        img = Image.open(image_path)
        img_tensor = prepare_image(img)
        
        # 使用MoVQGAN编码图像
        with torch.no_grad():
            # 添加batch维度并移到GPU
            out = model(img_tensor.to(device).unsqueeze(0))
            # 转换为列表格式并保存
            indices = str(out.cpu().tolist())
        # 保存结果，使用原始的key作为标识
        next_img_token = str(indices).replace(" ", "")
        numbers = next_img_token.strip('[]').split(',')
        next_img_token = ''.join([f'<|{num}|>' for num in numbers])
        gt_indices[key] = {
            'CAM_FRONT': next_img_token
        }
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        continue

# 保存结果
output_path = "./MoVQGAN/gt_indices_reasonplan_train.json"
with open(output_path, "w") as f:
    json.dump(gt_indices, f, indent=4)

print(f"Processing complete! Results saved to {output_path}")
print(f"Total images processed: {len(gt_indices)}")