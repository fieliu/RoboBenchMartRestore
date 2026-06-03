"""Verify --lateral-shift direction: for several shift values, call the real
find_corridor_and_goal() and re-measure obstacle->shelf gaps. Tells us which
sign/magnitude opens a clearly-passable gap (>1.2m) on one side.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsynth.envs import *
from dsynth.robots import *
import gymnasium as gym
from run_navdp_nav import find_corridor_and_goal

ROBOT_W = 1.0   # Fetch footprint diameter (relaxed pose)


def gaps_for(env, shlv, obstacle, fwd):
    lateral = np.array([-fwd[1], fwd[0], 0.0]); lateral /= np.linalg.norm(lateral)
    left_gap, right_gap = 1e9, 1e9
    for actor in shlv.values():
        c = actor.pose.sp.p
        if abs(np.dot(c - obstacle, fwd)) > 1.5:
            continue
        lat = float(np.dot(c - obstacle, lateral))
        if lat < -0.1:  left_gap = min(left_gap, abs(lat))
        elif lat > 0.1: right_gap = min(right_gap, abs(lat))
    return left_gap - 0.25, right_gap - 0.25   # subtract obstacle half-width


def main():
    env = gym.make("DarkstoreContinuousBaseEnv", robot_uids="ds_fetch_basket",
                   config_dir_path="demo_envs/pick_to_basket", num_envs=1,
                   control_mode="pd_joint_pos", render_mode="rgb_array",
                   obs_mode="state", sim_backend="cpu", render_backend="cpu")
    env.reset(options={"reconfigure": True})
    sb = env.unwrapped.scene_builder
    actv = env.unwrapped.active_shelves[0]
    shlv = env.unwrapped.actors["fixtures"]["shelves"]

    print(f"\nFetch needs ~{ROBOT_W:.1f}m. '+' free gap = passable; want one side > 1.2m")
    print(f"{'shift':>7}{'obs_y':>9}{'freeL':>8}{'freeR':>8}   verdict")
    print("-" * 48)
    for shift in [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6]:
        start, goal, obstacle, fwd, dist = find_corridor_and_goal(sb, actv, shlv, lateral_shift=shift)
        fl, fr = gaps_for(env, shlv, obstacle, fwd)
        best = max(fl, fr)
        side = "LEFT" if fl >= fr else "RIGHT"
        ok = "PASS %s(%.2f)" % (side, best) if best > 1.2 else "tight (%.2f)" % best
        print(f"{shift:>7.2f}{obstacle[1]:>9.2f}{fl:>8.2f}{fr:>8.2f}   {ok}")
    env.close()


if __name__ == "__main__":
    main()
