#!/bin/bash
# Live progress of all running replays. Re-run anytime, or: watch -n5 bash bash/replay_progress.sh
cd "$(dirname "$0")/.." || exit 1

running=$(pgrep -fc "replay_trajectory" 2>/dev/null || echo 0)
echo "===== replay 进度  (在跑进程: $running)  $(date +%H:%M:%S) ====="
free -h | awk '/Mem/{print "内存可用: "$7" / "$2}'
echo "-----------------------------------------------------------"

# files currently being replayed (truth for "running")
RUNNING_FILES=$(ps -C python -o args= 2>/dev/null | grep -oE "traj-path \S+" | awk '{print $2}')

shopt -s nullglob globstar
for log in generated_data/**/*.replay.log; do
  base="${log%.replay.log}"
  h5="${base}.h5"
  last=$(tr '\r' '\n' < "$log" 2>/dev/null | grep -oE "traj_[0-9]+: *[0-9]+%[^]]*]" | tail -1)
  if echo "$RUNNING_FILES" | grep -qF "$h5"; then
    state="⏳ running"
  elif grep -q "Killed\|Traceback\|Error" "$log" 2>/dev/null; then
    state="❌ failed"
  elif ls "${base}".rgbd.*.h5 >/dev/null 2>&1; then
    state="✅ done"
  else
    state="·  pending"
  fi
  printf "  %-10s %-52s %s\n" "$state" "${log#generated_data/}" "$last"
done