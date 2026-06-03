"""Offline NavDP checkpoint load + inference smoke test.

Does NOT touch the simulator. Verifies:
  1. checkpoint loads (reports missing/unexpected keys — backbone health check)
  2. GPU inference runs
  3. output shapes are correct (traj (1,24,3), values (1,16))
"""
import sys, os
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "..", "dsynth", "navigation", "navdp_models")
sys.path.insert(0, MODELS)

CKPT = os.path.join(MODELS, "navdp-cross-modal.ckpt")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def main():
    print(f"device={DEVICE}")
    print(f"ckpt={CKPT}  exists={os.path.exists(CKPT)}")

    # ── 1. inspect raw checkpoint keys ─────────────────────────
    raw = torch.load(CKPT, map_location="cpu")
    sd = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    keys = list(sd.keys()) if hasattr(sd, "keys") else []
    print(f"\n[CKPT] {len(keys)} tensors")
    has_rgb = any(k.startswith("rgbd_encoder.rgb_model") for k in keys)
    has_depth = any(k.startswith("rgbd_encoder.depth_model") for k in keys)
    print(f"[CKPT] rgb_model present={has_rgb}  depth_model present={has_depth}")

    # ── 2. build agent + load via controller ───────────────────
    from policy_network import NavDP_Policy
    net = NavDP_Policy(image_size=224, memory_size=8, predict_size=24,
                       temporal_depth=16, heads=8, token_dim=384, device=DEVICE)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print(f"\n[LOAD] missing={len(missing)}  unexpected={len(unexpected)}")
    # spot critical missing
    crit = [m for m in missing if "rgb_model" in m or "depth_model" in m
            or "action_head" in m or "decoder" in m]
    print(f"[LOAD] critical-ish missing (first 8): {crit[:8]}")
    net.to(DEVICE).eval()
    print("[LOAD] model on device, eval mode")

    # ── 3. inference smoke test via NavDP_Agent ────────────────
    from policy_agent import NavDP_Agent
    K = np.array([[200.0, 0, 112.0], [0, 200.0, 112.0], [0, 0, 1.0]], dtype=np.float32)
    agent = NavDP_Agent(image_intrinsic=K, navi_model=CKPT, device=DEVICE)
    agent.reset(batch_size=1, threshold=-3.0)
    print("\n[INFER] agent ready, running step_pointgoal …")

    # fake one frame: 360x640 RGB (0-255), depth in METRES (already converted)
    H, W = 360, 640
    rgb = (np.random.rand(1, H, W, 3) * 255).astype(np.float32)
    depth = (np.random.rand(1, H, W, 1) * 3.0 + 0.5).astype(np.float32)  # 0.5–3.5 m
    goal = np.array([[3.0, 0.5, 0.0]], dtype=np.float32)                 # 3m ahead, 0.5m left

    traj, all_traj, values, vis = agent.step_pointgoal(goal, rgb, depth)
    print(f"[INFER] traj shape={traj.shape}      (expect (1,24,3))")
    print(f"[INFER] all_traj shape={all_traj.shape}  (expect (1,16,24,3))")
    print(f"[INFER] values shape={values.shape}    (expect (1,16))")
    print(f"[INFER] traj[0] first 3 waypoints:\n{np.round(traj[0][:3], 3)}")
    print(f"[INFER] critic max={values.max():.2f} min={values.min():.2f}")
    ok = traj.shape == (1, 24, 3) and values.shape[0] == 1
    print(f"\n{'SMOKE TEST PASS' if ok else 'SMOKE TEST FAIL'}")
    return net


if __name__ == "__main__":
    main()
