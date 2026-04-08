#!/usr/bin/env bash
set -euo pipefail

interval_seconds=300
threshold=10

if ! command -v squeue >/dev/null 2>&1; then
  echo "squeue not found in PATH" >&2
  exit 1
fi

if ! command -v scancel >/dev/null 2>&1; then
  echo "scancel not found in PATH" >&2
  exit 1
fi

while true; do
  array_job_ids=()
  while IFS= read -r job_id; do
    [[ "$job_id" == *"_"* ]] || continue
    base_id="${job_id%%_*}"
    array_job_ids+=("$base_id")
  done < <(squeue -u "${USER}" -h -o "%i")

  if ((${#array_job_ids[@]} > 0)); then
    mapfile -t unique_base_ids < <(printf "%s\n" "${array_job_ids[@]}" | sort -u)
    for base_id in "${unique_base_ids[@]}"; do
      task_count=$(squeue -u "${USER}" -h -o "%i" | awk -v base="${base_id}_" '$1 ~ "^"base {count++} END {print count+0}')
      echo "Array job ${base_id}: remaining tasks = ${task_count}"
      if ((task_count > 0 && task_count < threshold)); then
        echo "Canceling array job ${base_id} (remaining tasks: ${task_count})"
        scancel "${base_id}"
      fi
    done
  else
    echo "No array jobs found."
  fi

  sleep "${interval_seconds}"
done
