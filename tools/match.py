import json
import os
import argparse

def load_pred_trajs_from_json(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

parser = argparse.ArgumentParser(description="Process trajectories and generate evaluation JSON.")
parser.add_argument("--pred_trajs_path", type=str, required=True, 
                    help="Path to the generated predictions JSONL file (e.g., generated_predictions.jsonl)")
parser.add_argument("--token_traj_path", type=str, required=True, 
                    help="Path to the token trajectories JSON file (e.g., val_cot_motion_single.json)")
parser.add_argument("--output_dir", type=str, 
                    help="Directory to save the output file. If not provided, uses the same directory as pred_trajs_path.")
args = parser.parse_args()

pred_trajs = load_pred_trajs_from_json(args.pred_trajs_path)
token_traj = json.load(open(args.token_traj_path, 'r'))


eval_traj = {}
for i, traj in enumerate(token_traj):
    eval_traj[traj['id']] = pred_trajs[i]['predict']


output_dir = args.output_dir if args.output_dir else os.path.dirname(args.pred_trajs_path)
output_path = os.path.join(output_dir, "eval_traj_0811.json")

with open(output_path, "w") as f:
    json.dump(eval_traj, f, ensure_ascii=False,indent=4)

print(f"Evaluation trajectories saved to {output_path}")