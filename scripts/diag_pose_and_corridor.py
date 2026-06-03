"""Combined diagnostic:
  A) evaluate candidate nav arm poses -> radius, ground clearance, camera occlusion
  B) measure corridor: obstacle box vs left/right shelf -> free gap width

Camera occlusion heuristic: express each arm link in robot frame; flag if a link
is in FRONT of the robot (x>0.1), ABOVE z=0.5 (near camera height), and within the
camera half-FOV cone of the forward axis -> would block head_camera.
"""
import sys, os
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dsynth.envs import *
from dsynth.robots import *
import gymnasium as gym

PAN, LIFT, UROLL, ELBOW, FROLL, WFLEX, WROLL = 5, 7, 8, 9, 10, 11, 12
ARM_KEYS = ("shoulder", "upperarm", "elbow", "forearm", "wrist", "gripper")
HALF_FOV = 1.442 / 2.0   # head_camera fov

# candidates: (name, pan, lift, uroll, elbow, froll, wflex, wroll)
CANDIDATES = [
    ("current(bad)",   -1.57, 1.518, 0.0,  0.0,  0.0, 1.57, 0.0),
    ("search_win_up",  -1.00, 1.520, 0.0,  2.0,  0.0, 1.50, 0.0),
    ("hang_side",      -1.57, 1.518, 0.0, -0.3,  0.0, 0.0,  0.0),
    ("hang_tuck",      -1.57, 1.518, 0.0, -1.0,  0.0, 0.0,  0.0),
    ("hang_tuck2",     -1.57, 1.300, 0.0, -1.5,  0.0, 0.5,  0.0),
    ("relaxed_down",   -1.30, 1.518, 0.0, -0.8,  0.0, 0.3,  0.0),
]


def eval_pose(robot, q0, links, vals):
    q = q0.copy()
    q[3] = q[4] = q[6] = 0.0   # torso, head_pan, head_tilt = 0
    q[PAN], q[LIFT], q[UROLL], q[ELBOW], q[FROLL], q[WFLEX], q[WROLL] = vals
    robot.set_qpos(torch.tensor(q[None], dtype=torch.float32))

    base = next(l for l in links if l.get_name() == "base_link")
    bp = base.pose.p[0].cpu().numpy()
    brot = base.pose.to_transformation_matrix()[0].cpu().numpy()[:3, :3]

    max_r, min_z, occl = 0.0, 1e9, False
    for lk in links:
        nm = lk.get_name()
        if not any(k in nm for k in ARM_KEYS):
            continue
        p = lk.pose.p[0].cpu().numpy()
        rel = brot.T @ (p - bp)            # link in robot frame
        r = float(np.hypot(rel[0], rel[1]))
        max_r = max(max_r, r); min_z = min(min_z, float(p[2]))
        # camera occlusion: in front (x>0.1), near/above camera height, inside FOV cone
        if rel[0] > 0.1 and p[2] > 0.5:
            ang = abs(np.arctan2(rel[1], rel[0]))
            if ang < HALF_FOV:
                occl = True
    return max_r, min_z, occl


def measure_corridor(env):
    """Measure free gap between obstacle position and nearest shelves on each side."""
    sb = env.unwrapped.scene_builder
    actv = env.unwrapped.active_shelves[0]
    shlv = env.unwrapped.actors["fixtures"]["shelves"]
    sp = shlv[actv[0]].pose.sp
    facing = sp.to_transformation_matrix()[:3, 1]
    perp = sp.to_transformation_matrix()[:3, 0]
    start = sp.p - 1.5 * facing
    fwd = perp.copy()
    if fwd[0] < 0: fwd = -fwd
    obstacle = start + 2.5 * fwd
    # lateral axis = perpendicular to travel direction, in ground plane
    lateral = np.array([-fwd[1], fwd[0], 0.0]); lateral /= np.linalg.norm(lateral)
    obs_half = 0.25   # obstacle box half-size

    # project all shelf centers onto lateral axis relative to obstacle
    print(f"\n=== CORRIDOR (travel dir fwd=({fwd[0]:.2f},{fwd[1]:.2f})) ===")
    print(f"obstacle at ({obstacle[0]:.2f},{obstacle[1]:.2f})  half-size {obs_half}m")
    left_gap, right_gap = 1e9, 1e9
    for i in range(len(shlv)):
        c = shlv[i].pose.sp.p
        d_fwd = np.dot(c - obstacle, fwd)
        if abs(d_fwd) > 1.5:        # only shelves near the obstacle band
            continue
        lat = np.dot(c - obstacle, lateral)
        if lat < -0.1:
            left_gap = min(left_gap, abs(lat))
        elif lat > 0.1:
            right_gap = min(right_gap, abs(lat))
    return obstacle, obs_half, left_gap, right_gap


def main():
    env = gym.make("DarkstoreContinuousBaseEnv", robot_uids="ds_fetch_basket",
                   config_dir_path="demo_envs/pick_to_basket", num_envs=1,
                   control_mode="pd_joint_pos", render_mode="rgb_array",
                   obs_mode="state", sim_backend="cpu", render_backend="cpu")
    env.reset(options={"reconfigure": True})
    robot = env.unwrapped.agent.robot
    q0 = robot.get_qpos()[0].cpu().numpy().copy()
    links = robot.get_links()

    print(f"\n{'pose':<16}{'radius':>8}{'min_z':>8}{'cam_blocked':>13}")
    print("-" * 45)
    for name, *vals in CANDIDATES:
        r, z, occl = eval_pose(robot, q0, links, vals)
        flag = "YES-BLOCKED" if occl else "clear"
        drag = " DRAG!" if z < 0.1 else ""
        print(f"{name:<16}{r:>8.3f}{z:>8.3f}{flag:>13}{drag}")

    obstacle, obs_half, lg, rg = measure_corridor(env)
    print(f"left shelf gap  = {lg:.2f} m   (obstacle edge to left shelf)")
    print(f"right shelf gap = {rg:.2f} m")
    free_left = lg - obs_half
    free_right = rg - obs_half
    print(f"\nfree passage LEFT of obstacle  = {free_left:.2f} m")
    print(f"free passage RIGHT of obstacle = {free_right:.2f} m")
    print(f"Fetch needs ~1.24 m diameter (current pose) / ~0.7 m (compact pose)")
    env.close()


if __name__ == "__main__":
    main()
