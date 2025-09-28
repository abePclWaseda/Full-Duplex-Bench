#!/bin/bash
#PBS -P gcg51557
#PBS -q R9920251000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ngpus=1
#PBS -l walltime=10:00:00
#PBS -j oe
#PBS -N 0162_full_duplex_bench

ROOT_DIR="dataset"
TASK="full"  # full | user_interruption

set -eu

echo "JOB_ID : $PBS_JOBID"
echo "WORKDIR: $PBS_O_WORKDIR"
cd   "$PBS_O_WORKDIR"

module list

source ~/miniforge3/etc/profile.d/conda.sh
conda activate full-duplex-bench

echo "==== which python ===="
which python
python --version

# ログ出力先を用意
mkdir -p logs

# 引数を明示して実行
exec python -m get_transcript.asr \
  --root_dir "$ROOT_DIR" \
  --task "$TASK" \
  > logs/0162_asr.log 2>&1
