import os
import numpy as np
import pickle
from pathlib import Path
import json
import ast
import re
import argparse
from PIL import Image, ImageDraw, ImageFont

from cam_render import CameraRender
from utils import AgentPredictionData, color_mapping
from visual_tokens import tokens_for_viz, tokens_for_main, viz_scenes
from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from nuscenes.utils.data_classes import LidarPointCloud, Box
from pyquaternion import Quaternion
import imageio


def draw_plan(sample_token, nusc, plan_trajs_dict, output_path):
    sdc_pred_color = np.array([255, 51, 51]) / 255.0
    sdc_gt_color = np.array([51, 255, 51]) / 255.0
    
    cam_render = CameraRender(show_gt_boxes=False)
    cam_render.reset_canvas(dx=2, dy=3, tight_layout=True)
    cam_render.render_image_data(sample_token, nusc)

    plan_traj = plan_trajs_dict['predict']
    gt_traj = plan_trajs_dict['label']

    try:
        numbers = re.findall(r'\((\d+\.\d+),(\d+\.\d+)\)', plan_traj)
        plan_traj = [(float(x), float(y)) for x, y in numbers]
        numbers = re.findall(r'\((\d+\.\d+),(\d+\.\d+)\)', gt_traj)
        gt_traj = [(float(x), float(y)) for x, y in numbers]
        plan_traj = np.concatenate([plan_traj, np.ones((6,1))], axis=-1)
        gt_traj = np.concatenate([gt_traj, np.ones((6,1))], axis=-1)
    except:
        return

    gt_agent_list = [
        AgentPredictionData(
        pred_score=1.0,
        pred_label=0,
        pred_center=[0, 0, 0],
        pred_dim=[4.5, 2.0, 2.0],
        pred_yaw=0,
        pred_vel=0,
        pred_traj=gt_traj,
        is_sdc=True
        )
    ]
    cam_render.render_pred_traj(
        gt_agent_list, sample_token, nusc, sdc_color=sdc_gt_color, render_sdc=True)

    pred_agent_list = [
        AgentPredictionData(
            pred_score=1.0,
            pred_label=0,
            pred_center=[0, 0, 0],
            pred_dim=[4.5, 2.0, 2.0],
            pred_yaw=0,
            pred_vel=0,
            pred_traj=plan_traj,
            is_sdc=True
        )
    ]
    cam_render.render_pred_traj(
        pred_agent_list, sample_token, nusc, sdc_color=sdc_pred_color, render_sdc=True)
    
    save_path = Path(output_path) / Path(sample_token + '.jpg')
    cam_render.save_fig(save_path)


def load_pred_trajs_from_json(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process NuScenes trajectory visualization.')
    parser.add_argument('--dataroot', type=str, default='./LLaMA-Factory/data/nuscenes',
                        help='Path to NuScenes dataset root directory')
    parser.add_argument('--pred-trajs-path', type=str, required=True,
                        help='Path to prediction trajectories JSONL file')
    parser.add_argument('--tokens-path', type=str, required=True,
                        help='Path to evaluation tokens JSON file')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Path to save visualization results')
    args = parser.parse_args()


    nusc = NuScenes(version="v1.0-trainval", dataroot=args.dataroot, verbose=True)
    plan_trajs_dict = load_pred_trajs_from_json(args.pred_trajs_path)
    tokens = json.load(open(args.tokens_path, 'r'))
    os.makedirs(args.output_path, exist_ok=True)

    for i, token in enumerate(tokens.keys()):
        draw_plan(token, nusc, plan_trajs_dict[i], args.output_path)