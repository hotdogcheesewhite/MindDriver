import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from movqgan import get_movqgan_model

def prepare_image(img):
    """ Convert PIL Image to normalized tensor without resizing. """
    transform = T.Compose([
        T.Resize((128, 128), interpolation=T.InterpolationMode.BICUBIC),
    ])
    pil_image = transform(img)
    arr = np.array(pil_image.convert("RGB")).astype(np.float32)
    arr = arr / 127.5 - 1  # [-1, 1]
    tensor = torch.from_numpy(np.transpose(arr, (2, 0, 1)))  # HWC -> CHW
    return tensor


device = torch.device('cuda:5')
model = get_movqgan_model('270M', ckpt_path='/mnt/nas-data-1/zhanglingjun.zlj1/data/MoVQGAN/movqgan_270M.ckpt', device=device)


root_dir = '/mnt/nas-data-1/zhanglingjun.zlj1/data/bench2drive-base'
output_jsonl = './MoVQGAN/gt_indices_bench2drive_bev.jsonl'


os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)


image_paths = []
for scene_name in os.listdir(root_dir):
    scene_path = os.path.join(root_dir, scene_name)
    if not os.path.isdir(scene_path):
        continue
    rgb_bev_dir = os.path.join(scene_path, 'camera', 'rgb_bev')
    if not os.path.exists(rgb_bev_dir):
        continue
    for fname in os.listdir(rgb_bev_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_paths.append(os.path.join(rgb_bev_dir, fname))

print(f"Found {len(image_paths)} images.")


with open(output_jsonl, 'w') as f_out:
    for img_path in tqdm(image_paths, desc="Processing and saving"):
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = prepare_image(img)
            with torch.no_grad():
                input_tensor = img_tensor.unsqueeze(0).to(device)
                indices_tensor = model(input_tensor)  
                indices_list = indices_tensor.cpu().tolist()  


            result = {
                img_path: {
                    "CAM_BEV": indices_list
                }
            }


            f_out.write(json.dumps(result) + "\n")
            f_out.flush() 

        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue

print(f"✅ Streaming save complete! File saved to: {output_jsonl}")
