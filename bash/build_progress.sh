#!/bin/bash
# Live progress for build_dataset_mp.py: shows each in-flight trajectory's latest
# frame-extraction line (from its .replay.log) + overall done count.
# Usage:  watch -n 5 bash bash/build_progress.sh
SPLIT_ROOT=${SPLIT_ROOT:-split_data}
OUT=${OUT:-datasets/warehouse_fetch}

done_cnt=$(find "$OUT" -name "*.parquet" 2>/dev/null | wc -l)
procs=$(pgrep -fc replay_trajectory 2>/dev/null || echo 0)
inflight=$(find "$SPLIT_ROOT" -name "*.replay.log" 2>/dev/null | wc -l)

echo "===== build 进度  $(date '+%H:%M:%S') ====="
echo "已完成轨迹(parquet): $done_cnt   在跑进程: $procs   进行中: $inflight"
echo "-----------------------------------------------------------"
find "$SPLIT_ROOT" -name "*.replay.log" 2>/dev/null | sort | while read -r log; do
  name=$(basename "$log" .replay.log)
  line=$(tr '\r' '\n' < "$log" 2>/dev/null | grep -E '[0-9]+%|step/s' | tail -1)
  [ -z "$line" ] && line="(starting...)"
  printf "  %-22s %s\n" "$name" "$line"
done
