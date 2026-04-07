import os
import shutil

# 1. 读出所有token/mnt/nas-data-1/zhanglingjun.zlj1/FSDrive-main/l2_tokens_4.0-5.0.txt/mnt/nas-data-1/zhanglingjun.zlj1/FSDrive-main/collision_tokens.txt
tokens = set()
with open('/mnt/nas-data-1/zhanglingjun.zlj1/FSDrive-main/l2_tokens_4.0-5.0.txt', 'r') as f:
    for line in f:
        if line.strip():
            token = line.split('+"+"+')[0]
            tokens.add(token)

# 2. 创建目标文件夹
src_dir = '/mnt/nas-data-1/zhanglingjun.zlj1/FSDrive-main/vis_trajbev'
dst_dir = '/mnt/nas-data-1/zhanglingjun.zlj1/FSDrive-main/l2_zengshaung_5+'
os.makedirs(dst_dir, exist_ok=True)

# 3. 遍历并复制
for fname in os.listdir(src_dir):
    if fname.endswith('.png') or fname.endswith('.jpg'):
        token_prefix = fname.split('_')[0]
        if token_prefix in tokens:
            shutil.copy(os.path.join(src_dir, fname),
                        os.path.join(dst_dir, fname))