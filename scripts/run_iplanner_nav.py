"""
iPlanner point-goal navigation in darkstore corridor.

Pipeline:
  1. init_nav_pose()   → PD drive arm+body to nav-ready posture
  2. rotate_in_place() → differential drive turn toward goal
  3. set_head_tilt()   → PD drive camera 40° down
  4. iPlanner loop     → 30 Hz replan, 0.6 m look-ahead
  5. set_head_tilt()   → PD drive camera back to level
"""
import argparse, os, time, sys
import numpy as np, cv2
import torch, gymnasium as gym, sapien

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dsynth.envs import *
from dsynth.robots import *
from dsynth.navigation.iplanner_controller import IPlannerController
from mani_skill.utils.building import actors
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils import sapien_utils

# ── low-level helpers ──────────────────────────────────────────────
def _arm_qpos(env):  return env.unwrapped.agent.controller.controllers["arm"].qpos[0].cpu().numpy()
def _body_qpos(env): return env.unwrapped.agent.controller.controllers["body"].qpos[0].cpu().numpy()

def get_depth(obs, cam="head_camera"):
    return obs["sensor_data"][cam]["depth"][0].cpu().numpy()

def get_robot_pose_matrix(env):
    return env.unwrapped.agent.base_link.pose.to_transformation_matrix()[0].cpu().numpy()

def make_action(env, lv, av):
    return np.hstack([_arm_qpos(env), 0.015, _body_qpos(env), [lv, av]])

def update_camera(env, rp, distance=4.0, height=2.4, look_ahead=2.5):
    p, f = rp[:3, 3], rp[:3, 0]
    src = p - distance * f + np.array([0, 0, height])
    tgt = p + look_ahead * f + np.array([0, 0, 0.8])
    n = sapien_utils.look_at(src, tgt).raw_pose[0].cpu().numpy()
    sc = env.unwrapped.scene.human_render_cameras["render_camera"].camera._render_cameras[0]
    sc.set_local_pose(sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]))
    sc.set_entity_pose(sapien.Pose(p=n[:3], q=n[3:]))

def write_frame(vw, env, obs, text_lines=None, depth_scale=8000.0):
    frame = env.render()
    if frame is None or vw is None:
        return
    fb = frame if isinstance(frame, np.ndarray) else frame[0].cpu().numpy()
    fb = cv2.cvtColor(fb, cv2.COLOR_RGB2BGR)
    dv = get_depth(obs).squeeze().astype(np.float32)
    dc = cv2.applyColorMap(np.clip(dv / depth_scale * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    dc = cv2.resize(dc, (fb.shape[1], fb.shape[0]))
    cmb = np.hstack([fb, dc])
    if text_lines:
        for i, txt in enumerate(text_lines):
            cv2.putText(cmb, txt, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    vw.write(cmb)

# ═══════════════════════════════════════════════════════════════════
# 1.  init_nav_pose  — arm to left side, torso minimum (PD)
# ═══════════════════════════════════════════════════════════════════
_ROOT_XYZ = slice(0, 3)
_BODY     = slice(3, 6)   # torso, head_pan, head_tilt  (Wait — URDF interleaves head+arm)
# DSFetchBasket full qpos: [root_x,root_y,root_z, torso, head_pan, shoulder_pan,
#                           head_tilt, shoulder_lift, upperarm_roll, elbow_flex,
#                           forearm_roll, wrist_flex, wrist_roll, r_gripper, l_gripper]

# action layout [arm(7) + gripper(1) + body(3) + base(2)]
#   arm   = [shoulder_pan, shoulder_lift, upperarm_roll, elbow_flex, forearm_roll, wrist_flex, wrist_roll]
#   body  = [head_pan, head_tilt, torso]
#   base  = [lv, av]
_ARM_INDICES  = slice(0, 7)
_BODY_INDICES = slice(8, 11)   # gripper takes index 7

_NAV_ARM = np.array([
    -1.57,   # shoulder_pan  = -90°   → arm to RIGHT side
    1.518,   # shoulder_lift = +87°   → upper arm hanging DOWN (max limit)
    0.0,     # upperarm_roll
    0.0,     # elbow_flex    = 0°     → forearm continues DOWN
    0.0,     # forearm_roll
    1.57,    # wrist_flex    = +90°   → wrist horizontal (hand points right)
    0.0,     # wrist_roll
], dtype=np.float32)

_NAV_BODY = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # head_pan, head_tilt=0, torso=0


def init_nav_pose(env, steps=150, vw=None, obs=None):
    """PD-drive arm & body to nav posture.  No teleport — actuators move the joints."""
    for i in range(steps):
        act = np.hstack([_NAV_ARM, 0.015, _NAV_BODY, [0.0, 0.0]])
        obs, _, _, _, _ = env.step(act.astype(np.float32))
        update_camera(env, get_robot_pose_matrix(env))
        if vw is not None and i % 3 == 0:
            write_frame(vw, env, obs, ["INIT: moving arm to nav posture"])
    print(f"  init_nav_pose ✓  ({steps} PD steps)")
    return obs

# ═══════════════════════════════════════════════════════════════════
# 2.  rotate_in_place — P-control differential-drive turn toward goal
#    Δθ = atan2(dy, dx)       ← bearing to goal in robot frame
#    ω  = K_p * Δθ            ← proportional angular velocity
#    Terminate when |Δθ| < ε  ← goal is ahead
# ═══════════════════════════════════════════════════════════════════
def rotate_in_place(env, target_world, K_p=0.8, tol_rad=0.087, max_steps=1000, vw=None, obs=None):
    """Differential-drive rotation — wheels spin opposite directions.
    Terminates when goal bearing is within tol_rad of the robot's forward axis.
    """
    print("[TURN]  P-control  Kp=0.8  tol=5°")
    t0 = time.time()
    for i in range(max_steps):
        T          = get_robot_pose_matrix(env)   # base_link in world
        robot_pos  = T[:3, 3]
        robot_fwd  = T[:3, 0]                     # robot's X-axis
        robot_yaw  = float(np.arctan2(robot_fwd[1], robot_fwd[0]))

        to_goal    = target_world - robot_pos
        goal_azim  = float(np.arctan2(to_goal[1], to_goal[0]))  # α
        dtheta     = goal_azim - robot_yaw                      # Δθ
        # normalise to [-π, π]
        dtheta     = float((dtheta + np.pi) % (2 * np.pi) - np.pi)

        if abs(dtheta) < tol_rad:
            print(f"  [TURN] ✓  {i+1} steps ({time.time()-t0:.1f}s)  "
                  f"robot_yaw={np.degrees(robot_yaw):.0f}°  goal_azim={np.degrees(goal_azim):.0f}°  "
                  f"Δθ={np.degrees(dtheta):.1f}°")
            return True, obs

        av = float(np.clip(K_p * dtheta, -0.7, 0.7))
        obs, _, _, _, _ = env.step(make_action(env, 0.0, av))
        update_camera(env, get_robot_pose_matrix(env))
        if vw is not None and i % 3 == 0:
            write_frame(vw, env, obs, [f"TURN: yaw={np.degrees(robot_yaw):.0f}° dtheta={np.degrees(dtheta):.0f}°"])
        if i % 80 == 0:
            print(f"  [TURN] {i:4d}  robot_yaw={np.degrees(robot_yaw):.0f}°  "
                  f"goal_azim={np.degrees(goal_azim):.0f}°  Δθ={np.degrees(dtheta):.0f}°  "
                  f"av={av:.3f}")
    print(f"  [TURN] ✗  timeout ({max_steps} steps)")
    return False, obs

# ═══════════════════════════════════════════════════════════════════
# 3.  set_head_tilt  — tilt camera down / up (PD)
# ═══════════════════════════════════════════════════════════════════
def set_head_tilt(env, tilt_rad, steps=25, vw=None, obs=None):
    """PD-drive head_tilt_joint.  tilt>0 = down, tilt<0 = up.  Range [-0.76, 1.45]."""
    tilt = float(np.clip(tilt_rad, -0.76, 1.45))
    for i in range(steps):
        a = _arm_qpos(env)
        b = _body_qpos(env).copy(); b[1] = tilt   # head_tilt is body index 1
        obs, _, _, _, _ = env.step(np.hstack([a, 0.015, b, [0.0, 0.0]]).astype(np.float32))
        update_camera(env, get_robot_pose_matrix(env))
        if vw is not None and i % 3 == 0:
            direction = "DOWN" if tilt > 0 else "UP"
            write_frame(vw, env, obs, [f"HEAD: tilting {np.degrees(tilt):.0f}° {direction}"])
    print(f"  set_head_tilt ✓  ({np.degrees(tilt):.0f}°)  ({steps} PD steps)")
    return obs

# ═══════════════════════════════════════════════════════════════════
# Scene  setup
# ═══════════════════════════════════════════════════════════════════
def find_corridor_and_goal(sb, active, shelves):
    sp     = shelves[active[0]].pose.sp
    facing = sp.to_transformation_matrix()[:3, 1]
    perp   = sp.to_transformation_matrix()[:3, 0]
    xs, ys = sb.x_size[0], sb.y_size[0]
    start  = sp.p - 1.5 * facing
    fwd = perp.copy()
    if fwd[0] < 0: fwd = -fwd
    free = xs - 2.0 - start[0] if abs(fwd[0]) > abs(fwd[1]) else ys - 2.0 - start[1]
    dist = max(3.0, min(free, 10.0))
    goal = start + dist * fwd
    goal[:2] = np.clip(goal[:2], 2.0, [xs - 2.0, ys - 2.0])
    obstacle = start + 2.5 * fwd
    return start, goal, obstacle, fwd, dist

# ═══════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene-dir",      default="demo_envs/pick_to_basket")
    p.add_argument("--goal-threshold", type=float, default=0.3)
    p.add_argument("--max-steps",      type=int,   default=2000)
    p.add_argument("--linear-speed",   type=float, default=0.3)
    p.add_argument("--angular-speed",  type=float, default=0.3)
    p.add_argument("--look-ahead",     type=float, default=0.5)
    p.add_argument("--model-path",     default=None)
    p.add_argument("--device",         default="cpu")
    p.add_argument("--output-dir",     default="navigation_results")
    p.add_argument("--sim-backend",    default="cpu")
    p.add_argument("--shader",         default="default")
    args = p.parse_args()

    env = gym.make("DarkstoreContinuousBaseEnv", robot_uids="ds_fetch_basket",
                   config_dir_path=args.scene_dir, num_envs=1,
                   control_mode="pd_joint_pos", render_mode="rgb_array",
                   obs_mode="rgbd", enable_shadow=False, parallel_in_single_scene=False,
                   sim_backend=args.sim_backend,
                   render_backend="cpu" if args.sim_backend == "cpu" else "gpu")
    obs, _ = env.reset(options={"reconfigure": True})
    print("Environment reset done.")

    sb   = env.unwrapped.scene_builder
    actv = env.unwrapped.active_shelves[0]
    shlv = env.unwrapped.actors["fixtures"]["shelves"]
    start, goal_world, obs_pos, fwd, goal_dist = find_corridor_and_goal(sb, actv, shlv)

    print(f"Scene {sb.x_size[0]:.0f}x{sb.y_size[0]:.0f}m  "
          f"start=({start[0]:.2f},{start[1]:.2f})  "
          f"goal=({goal_world[0]:.2f},{goal_world[1]:.2f})  dist={goal_dist:.1f}m  "
          f"obs=({obs_pos[0]:.2f},{obs_pos[1]:.2f})")

    actors.build_box(env.unwrapped.scene, half_sizes=[0.25, 0.25, 0.75],
                     color=[0.8, 0.4, 0.1, 1.0], name="obs",
                     initial_pose=sapien.Pose(p=obs_pos + [0, 0, 0.75]), scene_idxs=[0])
    print("Obstacle placed.")

    # ── create video writer early ─────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    vid = os.path.join(args.output_dir, f"nav_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
    sample_frame = env.render()
    sf = sample_frame if isinstance(sample_frame, np.ndarray) else sample_frame[0].cpu().numpy()
    frame_w = sf.shape[1] * 2
    frame_h = sf.shape[0]
    vw = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"mp4v"), 30, (frame_w, frame_h))

    # ── 1. init nav pose ──────────────────────────────────────
    print("[INIT] moving arm to nav posture …")
    obs = init_nav_pose(env, vw=vw, obs=obs)

    T0 = get_robot_pose_matrix(env)
    print(f"       pos=({T0[0,3]:.2f},{T0[1,3]:.2f})  yaw={np.degrees(np.arctan2(T0[1,0],T0[0,0])):.0f}°")

    # ── 2. rotate to face goal ────────────────────────────────
    print("[TURN] rotating toward goal …")
    _, obs = rotate_in_place(env, goal_world, vw=vw, obs=obs)

    # ── 3. head level (iPlanner trained with camera_tilt=0) ────
    print("[HEAD] setting level (0°) for iPlanner …")
    obs = set_head_tilt(env, np.radians(0), vw=vw, obs=obs)

    # ── 4. iPlanner ───────────────────────────────────────────
    iplanner = IPlannerController(model_path=args.model_path, device=args.device)
    print(f"iPlanner loaded  offset=({iplanner.sensor_offset_x:.2f},{iplanner.sensor_offset_y:.2f})")

    rpos = get_robot_pose_matrix(env)[:3, 3]
    initial_dist = np.linalg.norm(rpos[:2] - goal_world[:2])
    print(f"Robot pos=({rpos[0]:.2f},{rpos[1]:.2f})  dist_to_goal={initial_dist:.2f}m")
    print(f"goal_robot = {iplanner.compute_goal_in_robot_frame(goal_world, get_robot_pose_matrix(env))}")

    step = 0; reached = False
    fear_buf, fear_thr = 0, 3
    stuck_ctr = 0
    last_pos = rpos.copy(); total_dist = 0.0
    STUCK_THRESH = 60
    STUCK_RECOVER_STEPS = 40
    recover_step = 0
    recover_lv, recover_av = 0.0, 0.0

    print(f"\nNavigation start (max {args.max_steps} steps)\n")

    plan_interval = 7
    last_traj = None; last_fear = 0.0

    while step < args.max_steps:
        rp   = get_robot_pose_matrix(env)
        rpos = rp[:3, 3].copy()
        d2g  = np.linalg.norm(rpos[:2] - goal_world[:2])
        if d2g < args.goal_threshold:
            print(f"[DONE] step={step}  dist={d2g:.3f}m"); reached = True; break

        if step % plan_interval == 0 or last_traj is None:
            depth = get_depth(obs)
            gr    = iplanner.compute_goal_in_robot_frame(goal_world, rp)
            grc   = gr[:3]
            kp, last_traj, last_fear = iplanner.plan(depth, grc)

            if step % 80 == 0:
                d_np = depth.squeeze().astype(np.float32)
                valid = d_np[d_np > 0]
                print(f"  [DEPTH] dtype={depth.dtype} shape={depth.shape} "
                      f"range=[{d_np.min():.0f},{d_np.max():.0f}] "
                      f"valid={len(valid)} mean={valid.mean():.0f}" if len(valid) > 0 else
                      f"  [DEPTH] dtype={depth.dtype} shape={depth.shape} ALL ZEROS!")
                print(f"  [PLAN] fear={last_fear:.3f} keypoints={kp[:3].tolist()}")

        if recover_step > 0:
            lv, av = recover_lv, recover_av
            recover_step -= 1
            if recover_step == 0:
                stuck_ctr = 0
                last_traj = None
        else:
            is_forward = True
            if last_traj is not None and len(last_traj) > 0:
                for wp in last_traj:
                    if np.linalg.norm(wp[:2]) > 0.5:
                        heading = np.array([wp[0], wp[1]])
                        heading = heading / np.linalg.norm(heading)
                        if heading.dot(np.array([1.0, 0.0])) < 1.0 - 0.5:
                            is_forward = False
                        break

            if last_fear > 0.5 and is_forward:
                fear_buf = fear_buf + 1
            elif fear_buf > 0:
                fear_buf = fear_buf - 1
            if fear_buf > fear_thr:
                lv, av = 0.0, 0.0
            else:
                lv, av = iplanner.compute_base_velocity(
                    last_traj, rp, args.look_ahead,
                    args.linear_speed, args.angular_speed)

            mv = np.linalg.norm(rpos[:2] - last_pos[:2])
            total_dist += mv
            stuck_ctr = stuck_ctr + 1 if mv < 0.001 else 0
            last_pos = rpos.copy()

            if stuck_ctr > STUCK_THRESH:
                print(f"  [STUCK] detected at ({rpos[0]:.1f},{rpos[1]:.1f})  recovering …")
                recover_lv = -0.15
                recover_av = 0.4
                recover_step = STUCK_RECOVER_STEPS
                stuck_ctr = 0
                lv, av = recover_lv, recover_av
                recover_step -= 1

        obs, _, _, _, _ = env.step(make_action(env, lv, av))
        step += 1

        update_camera(env, get_robot_pose_matrix(env))
        write_frame(vw, env, obs, [
            f"D:{d2g:.1f}m V:{lv:.2f},{av:.2f} F:{last_fear:.2f}",
            f"Tr:{total_dist:.1f}m St:{step}",
        ])

        if step % 80 == 0:
            gr = iplanner.compute_goal_in_robot_frame(goal_world, rp)
            print(f"  {step:5d}| ({rpos[0]:.1f},{rpos[1]:.1f})  d2g={d2g:.1f}m  "
                  f"lv={lv:.2f} av={av:.2f}  fear={last_fear:.3f}  gr=({gr[0]:.1f},{gr[1]:.1f})")

    if not reached: print(f"[END] dist={d2g:.1f}m traveled={total_dist:.1f}m")

    # ── 5. head back to level ─────────────────────────────────
    print("[HEAD] back to level …")
    obs = set_head_tilt(env, 0.0, vw=vw, obs=obs)

    for _ in range(60):
        obs, _, _, _, _ = env.step(make_action(env, 0.0, 0.0))
        update_camera(env, get_robot_pose_matrix(env))
        write_frame(vw, env, obs, [f"FINAL - Traveled: {total_dist:.1f}m"], depth_scale=5000.0)

    if vw: vw.release(); print(f"Video: {vid}")
    env.close()

if __name__ == "__main__":
    main()
