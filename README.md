<div align="center">
<a id="readme-top"></a>
<h1>MindDriver: Introducing Progressive Multimodal Reasoning for Autonomous Driving </h1>
<h3 align="center"><strong>🎉🎉CVPR 2026 🎉🎉</strong></h3>
<a href="https://arxiv.org/abs/2602.21952"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
  
Lingjun Zhang<sup>1*</sup>,
Yujian Yuan<sup>1,2*</sup>,
Changjie Wu<sup>1†</sup>, 
Xinyuan Chang<sup>1</sup>, 
Xin Cai<sup>3</sup>,
Shuang Zeng<sup>1,4</sup>, 
Linzhe Shi<sup>1</sup>, 
Sijin Wang<sup>1</sup>, 
Hang Zhang<sup>1</sup>, 
Mu Xu<sup>1</sup>,

<sup>1</sup>Amap, Alibaba Group,
<sup>2</sup>The Hong Kong University of Science and Technology,
<sup>3</sup>The Chinese University of Hong Kong,
<sup>4</sup>Xi'an Jiaotong University

(*) Equal contribution. (†) Project leader.
</div>

<div align="center">
<img width="800" alt="image" src="assets/figintro1.png">
<p>Comparison of different reasoning methods. Text reasoning struggles with space misalignment, while image reasoning suffers from guideless image prediction. Our proposed progressive multimodal reasoning conducts aligned smooth reasoning.</p>
</div>


**MindDriver**: The proposed multimodal reasoning framework that enables VLM to imitate human-like progressive thinking for autonomous driving. MindDriver presents semantic understanding, semantic-to-physical space imagination, and physical-space trajectory planning.

## 🗓️ Release Plan
- **`2026/02`**: ✅ MindDriver paper.
- **`2026/04`**: ✅ MindDriver annotation and training code.
- **`2026/06`**: MindDriver checkpoints.
<p align="right"><a href="#readme-top"><img src=https://img.shields.io/badge/back%20to%20top-red?style=flat
></a></p>

## 🛠️ Installation

Create the required environment through the following steps:

```bash
git clone https://github.com/hotdogcheesewhite/MindDriver.git && cd MindDriver

conda create -n MindDriver python=3.10 -y && conda activate MindDriver

# CUDA 12.4
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

cd LLaMA-Factory && pip install -e ".[metrics,deepspeed,liger-kernel,bitsandbytes]" --no-build-isolation

cd .. && pip install -r requirements.txt
```
<p align="right"><a href="#readme-top"><img src=https://img.shields.io/badge/back%20to%20top-red?style=flat
></a></p>

## 📦 Data Preparation

1、Download nuScenes

Download the complete dataset from [nuScenes](https://www.nuscenes.org/nuscenes#download) and extract it to `./LLaMA-Factory/data/nuscenes`

Or establish a soft connection：

```bash
ln -s /path/to/your/nuscenes LLaMA-Factory/data
```

We used pre-cached data from the nuScenes dataset. The data can be downloaded at [Google Drive](https://drive.google.com/file/d/1Pc3vKtNHwZVY2mB9xBOOKiMIMr4hJFj7/view?usp=drive_link). The file `cached_nuscenes_info.pkl` is located in the directory `./create_data`. The `metrics` folder is placed in the directory `./tools/data`.

2、Extract visual tokens

Separately extract the visual tokens of the front view from fine-tuned data, to facilitate supervised MLLM:

```bash
python MoVQGAN/sft_data.py
```

3、Construct data

Construct fine-tuning data that conform to the LLaMA-Factory format respectively:

```bash
python create_data/sft_data.py --split train # Change to "val" for constructing the validation set
python gen_data/for_api_data.py
python gen_data/api_call_mutil.py
python gen_data/check.py
python gen_data.py/convert_to_qwen_img.py
NuScenes Raw Images
    │
    ▼
[Step 1] MoVQGAN/sft_data.py
    │ Encode CAM_FRONT images → discrete visual tokens
    │ Output: gt_indices_sft.json
    ▼
[Step 2] create_data/sft_data.py
    │ Build prompt + image paths + ground-truth trajectory
    │ Output: {split}_api_split.json (training data template)
    ▼
[Step 3] gen_data/api_call_mutil.py  ──── OR ────  create_data/singleinfer.py (local)
    │ Send images + prompt → LLM → get reasoning text
    │ Output: result_qwen_2_5_72b.json
    ▼
[Step 4] gen_data/check.py
    │ Quality-check each reasoning result via API
    │ Output: check_right_or_wrong.json
    ▼
[Step 5] gen_data/convert_to_qwen_img.py
    │ Merge: API results + check results + MoVQGAN tokens + CAN bus data
    │ Output: {split}_final.json (ready for training)
    ▼
[Step 6] LLaMA-Factory (configs/sft.yaml)
    │ Input: {split}_final.json + Qwen2.5-VL checkpoint
    │ Output: Fine-tuned model weights
```

```

Follow the [LLaMA-Factory tutorial](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md) and add the dataset information in the file `./LLaMA-Factory/data/dataset_info.json`.
<p align="right"><a href="#readme-top"><img src=https://img.shields.io/badge/back%20to%20top-red?style=flat
></a></p>

## 🚀 Training
Enter the working directory of LLaMA-Factory:
```bash
cd LLaMA-Factory
```
During the SFT stage, we assist the model in achieving two-stage alignment.
```bash
llamafactory-cli train ../configs/sft.yaml
```
<p align="right"><a href="#readme-top"><img src=https://img.shields.io/badge/back%20to%20top-red?style=flat
></a></p>

## 🎯 Infer
Run the following command in the LLaMA-Factory directory to infer test dataset:
```bash
python scripts/vllm_infer.py \ 
--model_name_or_path saves/qwen25_vl-3b/sft \
--dataset val_cot_motion \
--template qwen2_vl \
--cutoff_len 32768 \
--max_new_tokens 2048 \
--max_samples 100000 \
--image_resolution 524288 \
--save_name results.jsonl \
--temperature 0.1 \
--top_p 0.1 \
--top_k 10
```
<p align="right"><a href="#readme-top"><img src=https://img.shields.io/badge/back%20to%20top-red?style=flat
></a></p>

## 📈 Evaluation
First, under the MindDriver directory, match the predicted results with the tokens to facilitate the evaluation:
```bash
cd ..

python tools/match.py \
--pred_trajs_path ./LLaMA-Factory/results.jsonl \
--token_traj_path ./LLaMA-Factory/data/val_cot_motion.json
```

Then evaluate the L2 and collision rate indicators for the end-to-end trajectory planning:
```bash
python tools/evaluation/evaluation.py \
# Change to "stp3" and use the ST-P3 calculation method
--metric uniad \  
--result_file ./LLaMA-Factory/eval_traj.json
```
<p align="right"><a href="#readme-top"><img src=https://img.shields.io/badge/back%20to%20top-red?style=flat
></a></p>

## 👀 Visualization
Use the following command under the MindDriver directory to visualize the trajectory:
```bash
python tools/visualization/visualize_planning.py \
--pred-trajs-path ./LLaMA-Factory/results.jsonl \
--tokens-path ./LLaMA-Factory/eval_traj.json \  
--output-path ./vis_traj
```

Use the following command under the MindDriver directory to restore the visual tokens to the pixel space and visualize the CoT:
```bash
python ./MoVQGAN/vis.py \
--input_json ./LLaMA-Factory/eval_traj.json \
--output_dir ./vis_cot
```
<p align="right"><a href="#readme-top"><img src=https://img.shields.io/badge/back%20to%20top-red?style=flat
></a></p>

## 🙏 Acknowledgement
Our work is primarily based on the following codebases:[FSDrive](https://github.com/MIV-XJTU/FSDrive), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [MoVQGAN](https://github.com/ai-forever/MoVQGAN), [GPT-Driver](https://github.com/PointsCoder/GPT-Driver), [Agent-Driver](https://github.com/USC-GVL/Agent-Driver). We are sincerely grateful for their work.

<p align="right"><a href="#readme-top"><img src=https://img.shields.io/badge/back%20to%20top-red?style=flat
></a></p>
