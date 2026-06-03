"""NavDP point-goal navigation in darkstore corridor.

Same pipeline as run_iplanner_nav.py, but the perception->trajectory box is
NavDP (diffusion policy) instead of iPlanner:
  1. init_nav_pose()   -> PD drive arm+body to nav-ready posture
  2. rotate_in_place() -> differential drive turn toward goal
  3. set_head_tilt(0)  -> camera level (NavDP trained on front-view)
  4. NavDP loop        -> N-step replan, world-frame trajectory tracking
  5. set_head_tilt(0)  -> settle

NavDP specifics:
  - needs RGB + depth (both from head_camera, aligned)
  - stateful: reset() once, then step frame-by-frame in order
  - trajectory cached in world frame so replan interval N can be > 1
"""
import argparse, os, time, sys
import numpy as np, cv2
import torch, gymnasium as gym, sapien

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dsynth.envs import *
from dsynth.robots import *
from dsynth.navigation.navdp_controller import NavDPController
from mani_skill.utils.building import actors
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils import sapien_utils

# ── low-level helpers (shared with run_iplanner_nav.py) ────────────
def _arm_qpos(env):  return env.unwrapped.agent.controller.controllers["arm"].qpos[0].cpu().numpy()
def _body_qpos(env): return env.unwrapped.agent.controller.controllers["body"].qpos[0].cpu().numpy()

def get_depth(obs, cam="head_camera"):
    return obs["sensor_data"][cam]["depth"][0].cpu().numpy()

def get_rgb(obs, cam="head_camera"):
    return obs["sensor_data"][cam]["rgb"][0].cpu().numpy()

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

# ── relaxed "human" arm pose ───────────────────────────────────────
# Drive ANY starting arm posture to a relaxed hang-down pose:
#   shoulder_pan = 0      -> arm stays in FRONT, on centerline (0% cam occlusion, verified)
#   shoulder_lift= 1.518  -> upper arm hangs straight DOWN
#   elbow_flex   = 0      -> forearm continues straight DOWN
#   wrist_flex   = 0      -> hand continues down (palm forward)
# Torso is NOT raised (torso_height=0): keeping the torso low fixes the head
# camera height, which NavDP's obstacle avoidance depends on. The arm clears the
# floor via its hang-down geometry, not by lifting the body. This pose is
# gravity-assisted so the PD controller actually holds it (unlike folded poses).
_RELAX_ARM = np.array([0.0, 1.518, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

def relax_arm_to_human_pose(env, steps=150, torso_height=0.0, shoulder_pan=0.0,
                            vw=None, obs=None):
    """Drive the arm from whatever posture it's in to a relaxed human-like
    hang-down pose (upper arm + forearm vertical, hand forward, off the floor).

    Works from any start posture — PD pulls the joints to the target. `shoulder_pan`
    can be nudged off 0 if you want the arm slightly to one side; 0 keeps it
    centered in front (verified not to occlude the head camera).
    """
    arm = _RELAX_ARM.copy()
    arm[0] = shoulder_pan
    body = np.array([0.0, 0.0, float(torso_height)], dtype=np.float32)  # head_pan, head_tilt, torso
    for i in range(steps):
        act = np.hstack([arm, 0.015, body, [0.0, 0.0]]).astype(np.float32)
        obs, _, _, _, _ = env.step(act)
        update_camera(env, get_robot_pose_matrix(env))
        if vw is not None and i % 3 == 0:
            write_frame(vw, env, obs, [f"RELAX: arm hang-down  torso={torso_height:.2f}"])
    print(f"  relax_arm_to_human_pose done  (torso={torso_height:.2f}, {steps} PD steps)")
    return obs

def rotate_in_place(env, target_world, K_p=0.8, tol_rad=0.087, max_steps=1000, vw=None, obs=None):
    print("[TURN]  P-control  Kp=0.8  tol=5deg")
    t0 = time.time()
    for i in range(max_steps):
        T = get_robot_pose_matrix(env)
        robot_pos = T[:3, 3]; robot_fwd = T[:3, 0]
        robot_yaw = float(np.arctan2(robot_fwd[1], robot_fwd[0]))
        to_goal = target_world - robot_pos
        goal_azim = float(np.arctan2(to_goal[1], to_goal[0]))
        dtheta = float((goal_azim - robot_yaw + np.pi) % (2 * np.pi) - np.pi)
        if abs(dtheta) < tol_rad:
            print(f"  [TURN] done {i+1} steps ({time.time()-t0:.1f}s)  "
                  f"yaw={np.degrees(robot_yaw):.0f} azim={np.degrees(goal_azim):.0f}")
            return True, obs
        av = float(np.clip(K_p * dtheta, -0.7, 0.7))
        obs, _, _, _, _ = env.step(make_action(env, 0.0, av))
        update_camera(env, get_robot_pose_matrix(env))
        if vw is not None and i % 3 == 0:
            write_frame(vw, env, obs, [f"TURN: yaw={np.degrees(robot_yaw):.0f} dtheta={np.degrees(dtheta):.0f}"])
    print(f"  [TURN] timeout ({max_steps} steps)")
    return False, obs

def set_head_tilt(env, tilt_rad, steps=25, vw=None, obs=None):
    """PD-drive head_tilt_joint. tilt>0=down, tilt<0=up. Range [-0.76, 1.45]."""
    tilt = float(np.clip(tilt_rad, -0.76, 1.45))
    for i in range(steps):
        a = _arm_qpos(env)
        b = _body_qpos(env).copy(); b[1] = tilt   # head_tilt is body index 1
        obs, _, _, _, _ = env.step(np.hstack([a, 0.015, b, [0.0, 0.0]]).astype(np.float32))
        update_camera(env, get_robot_pose_matrix(env))
        if vw is not None and i % 3 == 0:
            write_frame(vw, env, obs, [f"HEAD: tilt {np.degrees(tilt):.0f}deg"])
    print(f"  set_head_tilt done  ({np.degrees(tilt):.0f}deg)")
    return obs

def find_corridor_and_goal(sb, active, shelves, lateral_shift=0.0):
    """Find start/goal/obstacle for the corridor.

    lateral_shift: signed metres to move start+goal+obstacle TOGETHER along the
    lateral axis (perpendicular to travel). Positive shifts toward +lateral.
    Use it to push the whole path off a too-narrow side toward the wider side.
    """
    sp = shelves[active[0]].pose.sp
    facing = sp.to_transformation_matrix()[:3, 1]
    perp = sp.to_transformation_matrix()[:3, 0]
    xs, ys = sb.x_size[0], sb.y_size[0]
    start = sp.p - 1.5 * facing
    fwd = perp.copy()
    if fwd[0] < 0: fwd = -fwd
    free = xs - 2.0 - start[0] if abs(fwd[0]) > abs(fwd[1]) else ys - 2.0 - start[1]
    dist = max(3.0, min(free, 10.0))
    goal = start + dist * fwd
    obstacle = start + 2.5 * fwd

    # lateral axis (perpendicular to travel, in ground plane)
    lateral = np.array([-fwd[1], fwd[0], 0.0])
    n = np.linalg.norm(lateral)
    if n > 1e-6:
        lateral = lateral / n
        shift_vec = lateral_shift * lateral
        start = start + shift_vec
        goal = goal + shift_vec
        obstacle = obstacle + shift_vec

    goal[:2] = np.clip(goal[:2], 2.0, [xs - 2.0, ys - 2.0])
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
    p.add_argument("--angular-speed",  type=float, default=0.4)
    p.add_argument("--look-ahead",     type=float, default=0.7)
    p.add_argument("--plan-interval",  type=int,   default=1, help="N: replan every N control steps")
    p.add_argument("--log-every",      type=int,   default=40, help="print key-info log every K control steps")
    p.add_argument("--stop-threshold", type=float, default=-3.0, help="NavDP critic stop threshold")
    p.add_argument("--head-tilt-deg",  type=float, default=15.0, help="head tilt (>0 down); look down 15deg during nav")
    p.add_argument("--torso-height",   type=float, default=0.0, help="torso lift (m); 0 = no raise (keep camera height fixed)")
    p.add_argument("--shoulder-pan",   type=float, default=0.0, help="arm pan: 0=front-center, -1.57=right side")
    p.add_argument("--lateral-shift",  type=float, default=0.0, help="shift start/goal/obstacle along lateral axis (m)")
    p.add_argument("--obstacle-height", type=float, default=0.3, help="obstacle box height (m)")
    p.add_argument("--obstacle-widen",  type=float, default=0.2, help="widen obstacle along lateral axis (m); position unchanged")
    p.add_argument("--model-path",     required=True, help="path to navdp checkpoint .ckpt")
    p.add_argument("--device",         default="cuda:0")
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
    start, goal_world, obs_pos, fwd, goal_dist = find_corridor_and_goal(
        sb, actv, shlv, lateral_shift=args.lateral_shift)

    print(f"Scene {sb.x_size[0]:.0f}x{sb.y_size[0]:.0f}m  "
          f"start=({start[0]:.2f},{start[1]:.2f})  "
          f"goal=({goal_world[0]:.2f},{goal_world[1]:.2f})  dist={goal_dist:.1f}m")

    # obstacle: height = args.obstacle_height, widened by --obstacle-widen along
    # the lateral axis so avoidance is clearer (position unchanged).
    half_h = args.obstacle_height / 2.0
    half_w = 0.25 + args.obstacle_widen / 2.0
    actors.build_box(env.unwrapped.scene, half_sizes=[0.25, half_w, half_h],
                     color=[0.8, 0.4, 0.1, 1.0], name="obs",
                     initial_pose=sapien.Pose(p=obs_pos + [0, 0, half_h]), scene_idxs=[0])
    print(f"Obstacle placed  h={args.obstacle_height}m  width={2*half_w:.2f}m "
          f"at ({obs_pos[0]:.2f},{obs_pos[1]:.2f})")

    # ── video writer ──────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    vid = os.path.join(args.output_dir, f"navdp_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
    sf = env.render()
    sf = sf if isinstance(sf, np.ndarray) else sf[0].cpu().numpy()
    vw = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"mp4v"), 30, (sf.shape[1] * 2, sf.shape[0]))

    # ── 1. relax arm to human-like hang-down pose ────────────
    print("[INIT] relaxing arm to human pose …")
    obs = relax_arm_to_human_pose(env, torso_height=args.torso_height,
                                  shoulder_pan=args.shoulder_pan, vw=vw, obs=obs)

    T0 = get_robot_pose_matrix(env)
    print(f"       pos=({T0[0,3]:.2f},{T0[1,3]:.2f})  yaw={np.degrees(np.arctan2(T0[1,0],T0[0,0])):.0f}°")

    # ── 2. head tilt FIRST (look down 15° before turning, per spec) ──
    print(f"[HEAD] setting tilt {args.head_tilt_deg:.0f}° for NavDP …")
    obs = set_head_tilt(env, np.radians(args.head_tilt_deg), vw=vw, obs=obs)

    # ── 3. rotate to face goal ────────────────────────────────
    print("[TURN] rotating toward goal …")
    _, obs = rotate_in_place(env, goal_world, vw=vw, obs=obs)

    # ── 4. NavDP ──────────────────────────────────────────────
    # head_camera intrinsic (640x360, fov=1.442) for NavDP's internal vis only
    navdp = NavDPController(model_path=args.model_path, device=args.device,
                            stop_threshold=args.stop_threshold)
    navdp.reset(threshold=args.stop_threshold)
    print(f"NavDP loaded  device={args.device}  N={args.plan_interval}  stop_thr={args.stop_threshold}")

    rpos = get_robot_pose_matrix(env)[:3, 3]
    initial_dist = np.linalg.norm(rpos[:2] - goal_world[:2])
    print(f"Robot pos=({rpos[0]:.2f},{rpos[1]:.2f})  dist_to_goal={initial_dist:.2f}m")

    step = 0; reached = False
    last_pos = rpos.copy(); total_dist = 0.0
    N = max(1, args.plan_interval)

    print(f"\nNavigation start (max {args.max_steps} steps, replan every {N})\n")

    while step < args.max_steps:
        rp   = get_robot_pose_matrix(env)
        rpos = rp[:3, 3].copy()
        d2g  = np.linalg.norm(rpos[:2] - goal_world[:2])
        if d2g < args.goal_threshold:
            print(f"[DONE] step={step}  dist={d2g:.3f}m"); reached = True; break

        # ── replan every N control steps ──────────────────────
        if step % N == 0 or navdp.traj_world is None:
            rgb        = get_rgb(obs)
            depth      = get_depth(obs)
            goal_robot = navdp.compute_goal_in_robot_frame(goal_world, rp)
            navdp.plan(rgb, depth, goal_robot, rp)

        # ── pure-pursuit over world-frame trajectory (every step) ──
        lv, av = navdp.compute_base_velocity(
            rp, look_ahead_dist=args.look_ahead,
            max_linear_vel=args.linear_speed, max_angular_vel=args.angular_speed)

        obs, _, _, _, _ = env.step(make_action(env, lv, av))
        step += 1

        mv = np.linalg.norm(rpos[:2] - last_pos[:2])
        total_dist += mv
        last_pos = rpos.copy()

        update_camera(env, get_robot_pose_matrix(env))
        vmax = float(np.max(navdp.last_values)) if navdp.last_values is not None else 0.0
        write_frame(vw, env, obs, [
            f"D:{d2g:.1f}m V:{lv:.2f},{av:.2f} C:{vmax:.2f}",
            f"Tr:{total_dist:.1f}m St:{step}",
        ])

        # ── key-info log (every --log-every steps) ────────────────
        if step % args.log_every == 0:
            tw = navdp.last_target_world
            gr = navdp.compute_goal_in_robot_frame(goal_world, rp)
            print(f"\n[step {step:4d}]  stop={navdp.is_stopped}  critic_max={vmax:.2f}")
            print(f"  goal(world)  =({goal_world[0]:.2f},{goal_world[1]:.2f})  "
                  f"dist={d2g:.2f}m  goal(robot)=({gr[0]:.2f},{gr[1]:.2f})")
            print(f"  robot(world) =({rpos[0]:.2f},{rpos[1]:.2f})  "
                  f"yaw={np.degrees(np.arctan2(rp[1,0],rp[0,0])):.0f}°")
            if navdp.traj_world is not None:
                pts = navdp.traj_world[:10]
                wp_s = "  ".join(f"({p[0]:.2f},{p[1]:.2f})" for p in pts)
                print(f"  waypoints[0:10] (world): {wp_s}")
            if tw is not None:
                print(f"  -> heading to wp(world)=({tw[0]:.2f},{tw[1]:.2f})  "
                      f"bearing={np.degrees(navdp.last_bearing):.0f}°  "
                      f"cmd lv={lv:.2f} av={av:.2f}")

    if not reached: print(f"[END] dist={d2g:.1f}m traveled={total_dist:.1f}m")

    # ── 5. settle ─────────────────────────────────────────────
    print("[HEAD] settle …")
    obs = set_head_tilt(env, 0.0, vw=vw, obs=obs)
    for _ in range(60):
        obs, _, _, _, _ = env.step(make_action(env, 0.0, 0.0))
        update_camera(env, get_robot_pose_matrix(env))
        write_frame(vw, env, obs, [f"FINAL - Traveled: {total_dist:.1f}m"], depth_scale=5000.0)

    if vw: vw.release(); print(f"Video: {vid}")
    env.close()

if __name__ == "__main__":
    main()
