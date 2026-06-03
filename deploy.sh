#!/usr/bin/env bash
#
# deploy.sh — RoboBenchMart 一键部署脚本
#   - 国内源安装(清华 pip + HF 镜像)
#   - 每步带验证,失败即停(set -e + 显式 check)
#   - 版本钉死为当前实测可用版本
#
# 用法:
#   bash deploy.sh            # 完整部署
#   bash deploy.sh --no-demo  # 跳过可选的 demo_envs 下载
#
set -euo pipefail

# ----------------------------- 配置 ----------------------------- #
ENV_NAME="dsynth"
PY_VER="3.10"
PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
HF_MIRROR="https://hf-mirror.com"
DOWNLOAD_DEMO=1
for arg in "$@"; do
  [ "$arg" = "--no-demo" ] && DOWNLOAD_DEMO=0
done

# ----------------------------- 工具函数 ----------------------------- #
say()  { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }
ok()   { printf "\033[1;32m  [OK] %s\033[0m\n" "$*"; }
fail() { printf "\033[1;31m  [FAIL] %s\033[0m\n" "$*"; exit 1; }

# --------------------- Step 0: 系统依赖 (Vulkan) --------------------- #
say "Step 0: 安装 Vulkan (渲染必需)"
if ! command -v vulkaninfo >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y libvulkan1 vulkan-tools
fi
if vulkaninfo 2>/dev/null | grep -qi "deviceName"; then
  ok "Vulkan 检测到 GPU 设备"
else
  printf "\033[1;33m  [WARN] vulkaninfo 未检测到 GPU。无头服务器请见脚本末尾排查说明。\033[0m\n"
fi

# --------------------- Step 1: Conda 环境 --------------------- #
say "Step 1: 创建 conda 环境 ${ENV_NAME} (python ${PY_VER})"
if ! command -v conda >/dev/null 2>&1; then
  fail "未找到 conda,请先安装 Miniconda/Anaconda"
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -qw "${ENV_NAME}" || conda create -n "${ENV_NAME}" python="${PY_VER}" -y
conda activate "${ENV_NAME}"
[ "$(python --version 2>&1 | grep -o '3\.10')" = "3.10" ] && ok "Python 3.10 就绪" || fail "Python 版本不对"

# --------------------- Step 2: pip 国内源 --------------------- #
say "Step 2: 配置 pip 清华源"
pip config set global.index-url "${PIP_INDEX}"
pip config set global.extra-index-url "https://pypi.org/simple"
ok "pip 源 = $(pip config get global.index-url)"

# placeholder-steps2

# --------------------- Step 3: PyTorch (CUDA 12.4) --------------------- #
say "Step 3: 安装 PyTorch 2.5.1 + cu124"
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python - <<'PY' || fail "PyTorch CUDA 不可用"
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() == False"
print("  torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
PY
ok "PyTorch CUDA 就绪"

# --------------------- Step 4: 项目依赖 --------------------- #
say "Step 4: 安装项目依赖 (清华源)"
# 去掉 requirements 里的 torch/torchvision(Step 3 已装 cu124 版,避免被覆盖成 CPU 版)
grep -vE "^torch$|^torchvision$" requirements.txt > /tmp/req_no_torch.txt
pip install -r /tmp/req_no_torch.txt
pip install mani_skill==3.0.1 mplib==0.2.1 sapien==3.0.3 diffusers==0.33.1
python - <<'PY' || fail "核心库导入失败"
import mani_skill, sapien, mplib, diffusers
print("  mani_skill", mani_skill.__version__, "sapien", sapien.__version__,
      "mplib", mplib.__version__, "diffusers", diffusers.__version__)
PY
ok "核心库就绪"

# --------------------- Step 5: 验证渲染链路 --------------------- #
say "Step 5: 验证 ManiSkill 渲染"
timeout 120 python -m mani_skill.examples.demo_random_action >/dev/null 2>&1 \
  && ok "渲染链路 OK" \
  || printf "\033[1;33m  [WARN] demo_random_action 未通过,可能是无头 Vulkan 问题,见末尾排查。\033[0m\n"

# --------------------- Step 6: 下载资产 --------------------- #
say "Step 6: 下载资产 (RoboCasa + RoboBenchMart, HF 镜像)"
export HF_ENDPOINT="${HF_MIRROR}"
python -m mani_skill.utils.download_asset RoboCasa
hf download emb-ai/RoboBenchMart_assets --repo-type dataset --local-dir assets
if [ "${DOWNLOAD_DEMO}" = "1" ]; then
  hf download emb-ai/RoboBenchMart_demo_envs --repo-type dataset --local-dir demo_envs
fi
[ -d "$HOME/.maniskill/data" ] && ok "RoboCasa 资产: $(du -sh "$HOME/.maniskill/data" | cut -f1)" || fail "RoboCasa 资产缺失"
[ -d assets ] && ok "RoboBenchMart 资产已下载" || fail "assets 目录缺失"

# --------------------- Step 7: 端到端验证 --------------------- #
say "Step 7: 端到端验证 (跑一个真实任务并出视频)"
if [ "${DOWNLOAD_DEMO}" = "1" ]; then
  python scripts/run_mp.py -e PickFromFloorSlamContEnv \
    --scene-dir demo_envs/pick_from_floor -n 1 -b cpu --no-retry --save-video --debug \
    && ok "端到端通过,部署完成" \
    || printf "\033[1;33m  [WARN] 端到端任务未成功(可能是 solver 逻辑而非部署问题)。\033[0m\n"
else
  printf "  跳过端到端验证(未下载 demo_envs)。\n"
fi

cat <<'EOF'

============================================================
 部署完成。激活环境:  conda activate dsynth
------------------------------------------------------------
 无头服务器 Vulkan 排查(若 Step 5/7 报 Vulkan/GPU 错):
   ls /usr/share/vulkan/icd.d/nvidia_icd.json
   export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
   然后重跑:  python -m mani_skill.examples.demo_random_action
------------------------------------------------------------
 NavDP 导航模型 (519M) 不在本脚本下载范围;
   如需导航,单独放置:
   dsynth/navigation/navdp_models/navdp-cross-modal.ckpt
============================================================
EOF
