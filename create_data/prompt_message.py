import numpy as np

def generate_user_message(data, token, perception_range=20.0, short=True):

    # user_message  = f"You have received new input data to help you plan your route.\n"
    user_message  = f"Here's some information you'll need:\n"
    
    data_dict = data[token]
    camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_LEFT',
            'CAM_FRONT_RIGHT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
    images_path = []
    for cam in camera_types:
        images_path.append(data_dict['cams'][cam]['data_path'].replace('/localdata_ssd/nuScenes', 'data/nuscenes', 1))

    """
    Historical Trjectory:
        gt_ego_his_trajs: [5, 2] last 2 seconds 
        gt_ego_his_diff: [4, 2] last 2 seconds, differential format, viewed as velocity 
    """
    xh1 = data_dict['gt_ego_his_trajs'][0][0]
    yh1 = data_dict['gt_ego_his_trajs'][0][1]
    xh2 = data_dict['gt_ego_his_trajs'][1][0]
    yh2 = data_dict['gt_ego_his_trajs'][1][1]
    xh3 = data_dict['gt_ego_his_trajs'][2][0]
    yh3 = data_dict['gt_ego_his_trajs'][2][1]
    xh4 = data_dict['gt_ego_his_trajs'][3][0]
    yh4 = data_dict['gt_ego_his_trajs'][3][1]
    user_message += f"Historical Trajectory (last 2 seconds):"
    user_message += f" [({xh1:.2f},{yh1:.2f}), ({xh2:.2f},{yh2:.2f}), ({xh3:.2f},{yh3:.2f}), ({xh4:.2f},{yh4:.2f})]\n"
    
    """
    Mission goal:
        gt_ego_fut_cmd
    """
    cmd_vec = data_dict['gt_ego_fut_cmd']
    right, left, forward = cmd_vec
    if right > 0:
        mission_goal = "RIGHT"
    elif left > 0:
        mission_goal = "LEFT"
    else:
        assert forward > 0
        mission_goal = "FORWARD"
    user_message += f"Mission Goal: "
    user_message += f"{mission_goal}\n"

    """
    Traffic Rule
    """
    user_message += f"Traffic Rules: Avoid collision with other objects.\n- Always drive on drivable regions.\n- Avoid driving on occupied regions.\n"
    
    return user_message, images_path


def generate_assistant_message(data, token, traj_only = False):

    data_dict = data[token]
    if traj_only:
        assitant_message = ""
    else:
        assitant_message = generate_chain_of_thoughts(data_dict)

    x1 = data_dict['gt_ego_fut_trajs'][1][0]
    x2 = data_dict['gt_ego_fut_trajs'][2][0]
    x3 = data_dict['gt_ego_fut_trajs'][3][0]
    x4 = data_dict['gt_ego_fut_trajs'][4][0]
    x5 = data_dict['gt_ego_fut_trajs'][5][0]
    x6 = data_dict['gt_ego_fut_trajs'][6][0]
    y1 = data_dict['gt_ego_fut_trajs'][1][1]
    y2 = data_dict['gt_ego_fut_trajs'][2][1]
    y3 = data_dict['gt_ego_fut_trajs'][3][1]
    y4 = data_dict['gt_ego_fut_trajs'][4][1]
    y5 = data_dict['gt_ego_fut_trajs'][5][1]
    y6 = data_dict['gt_ego_fut_trajs'][6][1]
    STOP = False
    if (x1 == 0 and y1 == 0 and
        x2 == 0 and y2 == 0 and
        x3 == 0 and y3 == 0 and
        x4 == 0 and y4 == 0 and
        x5 == 0 and y5 == 0 and
        x6 == 0 and y6 == 0):
        STOP = True
    if not traj_only:
        assitant_message += f"Trajectory:\n"    
    assitant_message += f"[({x1:.2f},{y1:.2f}), ({x2:.2f},{y2:.2f}), ({x3:.2f},{y3:.2f}), ({x4:.2f},{y4:.2f}), ({x5:.2f},{y5:.2f}), ({x6:.2f},{y6:.2f})]"
    return assitant_message, STOP

def generate_goal_message(data, token, perception_range=20.0, short=True):

    # user_message  = f"You have received new input data to help you plan your route.\n"
    
    data_dict = data[token]
    """
    Mission goal:
        gt_ego_fut_cmd
    """
    cmd_vec = data_dict['gt_ego_fut_cmd']
    right, left, forward = cmd_vec
    if right > 0:
        mission_goal = "RIGHT"
    elif left > 0:
        mission_goal = "LEFT"
    else:
        assert forward > 0
        mission_goal = "FORWARD"
    user_message = mission_goal
    return user_message

def generate_histraj_message(data, token, perception_range=20.0, short=True):

    # user_message  = f"You have received new input data to help you plan your route.\n"
    user_message  = f"Here's some information you'll need:\n"
    
    data_dict = data[token]
    """
    Historical Trjectory:
        gt_ego_his_trajs: [5, 2] last 2 seconds 
        gt_ego_his_diff: [4, 2] last 2 seconds, differential format, viewed as velocity 
    """
    xh1 = data_dict['gt_ego_his_trajs'][0][0]
    yh1 = data_dict['gt_ego_his_trajs'][0][1]
    xh2 = data_dict['gt_ego_his_trajs'][1][0]
    yh2 = data_dict['gt_ego_his_trajs'][1][1]
    xh3 = data_dict['gt_ego_his_trajs'][2][0]
    yh3 = data_dict['gt_ego_his_trajs'][2][1]
    xh4 = data_dict['gt_ego_his_trajs'][3][0]
    yh4 = data_dict['gt_ego_his_trajs'][3][1]
    user_message += f"Historical Trajectory (last 2 seconds):"
    user_message += f" [({xh1:.2f},{yh1:.2f}), ({xh2:.2f},{yh2:.2f}), ({xh3:.2f},{yh3:.2f}), ({xh4:.2f},{yh4:.2f})]\n"
    return user_message

