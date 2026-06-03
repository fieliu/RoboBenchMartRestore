import multiprocessing as mp
import os
import json
from copy import deepcopy
import time
import argparse
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import os.path as osp

from mani_skill.utils.structs.pose import to_sapien_pose
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.trajectory.merge_trajectory import merge_trajectories

import mplib
from mplib.collision_detection import fcl
import sys
sys.path.append('.')
from dsynth.envs import *
from dsynth.robots import *

from dsynth.planning import MP_SOLUTIONS, R1_MP_SOLUTIONS
from dsynth.planning.utils import BAD_ENV_ERROR_CODE

OPEN = 1
CLOSED = -1


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PickToCartEnv", help=f"Environment to run motion planning solver on. Available options are {list(MP_SOLUTIONS.keys())}")
    parser.add_argument("--scene-dir", type=str)
    parser.add_argument("-o", "--obs-mode", type=str, default="none", help="Observation mode to use. Usually this is kept as 'none' as observations are not necesary to be stored, they can be replayed later via the mani_skill.trajectory.replay_trajectory script.")
    parser.add_argument("-n", "--num-traj", type=int, default=10, help="Number of trajectories to generate.")
    parser.add_argument("--only-count-success", action="store_true", help="If true, generates trajectories until num_traj of them are successful and only saves the successful trajectories/videos")
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-b", "--sim-backend", type=str, default="auto", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("--render-mode", type=str, default="rgb_array", help="can be 'sensors' or 'rgb_array' which only affect what is saved to videos")
    parser.add_argument("--vis", action="store_true", help="whether or not to open a GUI to visualize the solution live")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save-video", action="store_true", help="whether or not to save videos locally")
    parser.add_argument("--traj-name", type=str, help="The name of the trajectory .h5 file that will be created.")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("-r", "--robot-uids", type=str, default="ds_fetch_basket", choices=["ds_fetch_basket", "ds_r1"], help="Robot to use for motion planning")
    parser.add_argument("--num-procs", type=int, default=1, help="Number of processes to use to help parallelize the trajectory replay process. This uses CPU multiprocessing and only works with the CPU simulation backend at the moment.")
    parser.add_argument("--env-kwargs", type=str, default=None, help="Extra env kwargs as a JSON object, forwarded to gym.make. e.g. '{\"num_basket_distractors\": 3}'")
    parser.add_argument("--no-retry", action="store_true", help="Run the solver exactly once (no retry loop) and force-save the video regardless of success/failure. For debugging what goes wrong.")
    return parser.parse_args()


def _main(args, proc_id=0):
    env_id = args.env_id
    scene_dir = args.scene_dir
    robot_uids = args.robot_uids

    extra_env_kwargs = json.loads(args.env_kwargs) if args.env_kwargs else {}

    env = gym.make(env_id,
                    robot_uids=robot_uids,
                   config_dir_path = scene_dir,
                   num_envs=1,
                   control_mode="pd_joint_pos",
                   viewer_camera_configs={'shader_pack': args.shader},
                    human_render_camera_configs={'shader_pack': args.shader},
                    sensor_configs={'shader_pack': args.shader},
                   render_mode="rgb_array",
                   enable_shadow=True,
                   obs_mode=args.obs_mode,
                   parallel_in_single_scene = False,
                   sim_backend=args.sim_backend,
                   render_backend='cpu' if args.sim_backend == 'cpu' else 'gpu',
                   **extra_env_kwargs,
                   )

    # Select solver based on robot type
    if robot_uids == 'ds_r1':
        if env_id not in R1_MP_SOLUTIONS:
            raise RuntimeError(f"No R1 motion planning solutions for {env_id}. Available: {list(R1_MP_SOLUTIONS.keys())}")
        solve = R1_MP_SOLUTIONS[env_id]
    else:
        if env_id not in MP_SOLUTIONS:
            raise RuntimeError(f"No motion planning solutions for {env_id}. Available: {list(MP_SOLUTIONS.keys())}")
        solve = MP_SOLUTIONS[env_id]

    if not args.traj_name:
        new_traj_name = time.strftime("%Y%m%d_%H%M%S")
    else:
        new_traj_name = args.traj_name

    if args.num_procs > 1:
        new_traj_name = new_traj_name + "." + str(proc_id)

    output_dir = osp.join(scene_dir, "demos", env_id, new_traj_name)

    env = RecordEpisode(
        env,
        output_dir=output_dir,
        trajectory_name=new_traj_name,
        save_video=args.save_video,
        source_type="motionplanning",
        source_desc="official motion planning solution from dsynth contributors",
        record_reward=False,
        save_on_reset=False,
    )

    if args.vis:
        env = gym.wrappers.HumanRendering(env)

    print(f"Motion Planning Running on {env_id}")

    solve = MP_SOLUTIONS[env_id] if robot_uids != 'ds_r1' else R1_MP_SOLUTIONS[env_id]

    pbar = tqdm(total=args.num_traj)
    n_success = 0
    solution_episode_lengths = []
    failed_motion_plan = 0
    while n_success < args.num_traj:
        seed = np.random.randint(2**31)

        res = solve(env, seed=seed, debug=args.debug, vis=args.vis)

        if args.no_retry:
            # Debug mode: run exactly once, force-save video, stop regardless of outcome.
            print(f"[no-retry] solver returned res={res} (one-shot debug run, not retrying)")
            break

        if res == BAD_ENV_ERROR_CODE:
            failed_motion_plan += 1
            continue
        if res == -1:
            failed_motion_plan += 1
            continue

        if args.only_count_success:
            # Check success from the latest info
            obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape[0]))
            if not info.get('success', [False])[0]:
                failed_motion_plan += 1
                continue

        n_success += 1
        pbar.update(1)
        env.flush_trajectory()

    pbar.close()

    if args.save_video or args.no_retry:
        env.flush_video(verbose=True)

    # Merge trajectories
    all_trajs = sorted([f for f in os.listdir(output_dir) if f.startswith(new_traj_name) and f.endswith('.h5')])
    if len(all_trajs) > 1:
        merge_trajectories(output_dir, new_traj_name)

    print(f"success_rate={n_success}/{args.num_traj}, failed_motion_plan_rate={failed_motion_plan}")
    print(f"Output saved to: {output_dir}")


def main(args=None):
    if args is None:
        args = parse_args()

    if args.num_procs > 1:
        pool = mp.Pool(args.num_procs)
        procs = []
        for i in range(args.num_procs):
            procs.append(pool.apply_async(_main, (args, i)))
        pool.close()
        pool.join()
        for p in procs:
            p.get()
    else:
        _main(args)


if __name__ == "__main__":
    main(parse_args())
