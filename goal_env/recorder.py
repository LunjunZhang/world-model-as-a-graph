import cv2
import torch
import numpy as np


## This is used to store a video for remote visualization
def play(env, policy, video_path="tmp.avi", time_limit=500, device='cpu'):
    out = None
    obs = env.reset()
    num = 0
    
    rew = None
    action = None
    info = None
    flag = False
    while True:
        img = env.unwrapped.render(mode='rgb_array')[:, :, ::-1].copy()
        '''
        if True and isinstance(obs, dict):
            np.set_printoptions(precision=3)
            achieved = (float(obs['achieved_goal'][0]), float(obs['achieved_goal'][1]))
            desired = (float(obs['desired_goal'][0]), float(obs['desired_goal'][1]))

            cv2.putText(img, " obs: {:.3f} {:.3f}".format(achieved[0], achieved[1]), (400,25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
            cv2.putText(img, "goal: {:.3f} {:.3f}".format(desired[0], desired[1]), (400,50), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
            if rew is not None:
                cv2.putText(img, "rew: {:.3f}".format(rew), (400,75), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
            if action is not None:
                action = [float(i) for i in action][:2]
                cv2.putText(img, "rew: {:.3f} {:.3f}".format(action[0], action[1]), (400,100), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
            if info is not None:
                if 'is_success' in info:
                    cv2.putText(img, "success? {}".format(info['is_success']), (400,125), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
            cv2.putText(img, "step {}".format(num), (400,150), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
            flag = True
        '''
        if out is None:
            out = cv2.VideoWriter(
                video_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (img.shape[1], img.shape[0]))
        out.write(img)
        if isinstance(obs, dict):
            goal = torch.tensor(obs['desired_goal'], dtype=torch.float32).to(device)
            obs = torch.tensor(obs['observation'], dtype=torch.float32).to(device)
            action = policy(obs.unsqueeze(0), goal.unsqueeze(0))
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()
        else:
            action = policy(np.array(obs)[None]).action[0].detach().cpu().numpy()
        obs, rew, done, info = env.step(action)
        if done:
            obs = env.reset()
        num += 1
        # assert not info['is_success']
        flag = True
        if not flag:
            print(num, info, rew, done, env.goal, action)
        if num == time_limit - 1:
            break
