"""Replay a trajectory with live visualization."""
import argparse
import sys
import h5py
import numpy as np
import gymnasium as gym

sys.path.append('.')
from dsynth.envs import *
from dsynth.robots import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("traj_path", help="Path to .h5 trajectory file")
    parser.add_argument("--sim-backend", type=str, default="cpu")
    parser.add_argument("--render-backend", type=str, default="cpu")
    parser.add_argument("--no-shadow", action="store_true")
    parser.add_argument("--camera-width", type=int, default=320)
    parser.add_argument("--camera-height", type=int, default=320)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()

    # Load trajectory metadata
    with h5py.File(args.traj_path, 'r') as f:
        env_id = f['traj_0'].attrs['env_id']
        env_kwargs_str = f['traj_0'].attrs['env_kwargs']
        actions = f['traj_0/actions'][:]
        env_states = f['traj_0/env_states'][:]
        n_steps = len(actions)

    # Parse env kwargs from stored JSON
    import json
    env_kwargs = json.loads(env_kwargs_str)
    robot_uids = env_kwargs.get('robot_uids', 'ds_fetch_basket')
    config_dir = env_kwargs.get('config_dir_path', 'generated_envs/ds_small_scene/')
    control_mode = env_kwargs.get('control_mode', 'pd_joint_pos')

    print(f"Trajectory: {args.traj_path}")
    print(f"  Task: {env_id}")
    print(f"  Robot: {robot_uids}")
    print(f"  Steps: {n_steps}")
    print(f"  Scene: {config_dir}")

    camera_cfg = dict(width=args.camera_width, height=args.camera_height)

    env = gym.make(
        env_id,
        robot_uids=robot_uids,
        config_dir_path=config_dir,
        num_envs=1,
        control_mode=control_mode,
        render_mode="human",
        enable_shadow=not args.no_shadow,
        obs_mode="none",
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        human_render_camera_configs=dict(render_camera=camera_cfg),
    )

    # Reset and set initial state
    obs, info = env.reset(options={'reconfigure': True})
    viewer = env.render()

    if env_states is not None and len(env_states) > 0:
        env.agent.robot.set_state(env_states[0])

    env.render()

    print(f"\nPress 'p' to play/pause, ESC to quit")
    print(f"Playing {n_steps} steps...")

    step = 0
    playing = True
    dt = 1.0 / args.fps
    import time

    while step < n_steps:
        # Handle viewer events
        if viewer.window.key_press('escape'):
            break
        if viewer.window.key_press('p'):
            playing = not playing
            print(f"{'Playing' if playing else 'Paused'}")

        if playing:
            action = actions[step]
            try:
                obs, reward, terminated, truncated, info = env.step(action)
            except Exception as e:
                print(f"Step {step} error: {e}")
            step += 1
            if step % 50 == 0:
                print(f"  Step {step}/{n_steps}")

            time.sleep(dt)

    print(f"Done. Replayed {step} steps.")
    env.close()


if __name__ == "__main__":
    main()
