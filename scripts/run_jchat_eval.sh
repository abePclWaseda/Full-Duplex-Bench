#!/bin/bash
#
# Full pipeline: preprocess j-chat data + run all evaluation tasks
#
# Usage:
#   bash scripts/run_jchat_eval.sh [--jchat_dir DIR] [--output_dir DIR] [--skip_preprocess]
#

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
JCHAT_DIR="${PROJECT_DIR}/dataset/j-chat"
OUTPUT_DIR="${PROJECT_DIR}/dataset/jchat_eval"
SKIP_PREPROCESS=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --jchat_dir)
            JCHAT_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip_preprocess)
            SKIP_PREPROCESS=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo " j-chat Full-Duplex-Bench Evaluation"
echo "========================================"
echo "j-chat dir:  $JCHAT_DIR"
echo "Output dir:  $OUTPUT_DIR"
echo ""

# ----------------------------------------
# Step 1: Preprocessing
# ----------------------------------------
if [ "$SKIP_PREPROCESS" -eq 0 ]; then
    echo "==== Step 1: Preprocessing ===="
    uv run python "${SCRIPT_DIR}/jchat_preprocess.py" \
        --jchat_dir "$JCHAT_DIR" \
        --output_dir "$OUTPUT_DIR"
    echo ""
else
    echo "==== Step 1: Preprocessing (skipped) ===="
    echo ""
fi

# ----------------------------------------
# Step 2: v1.0 Evaluation Tasks
# ----------------------------------------
echo "==== Step 2: v1.0 Evaluation ===="

# Map: eval_task_name -> output_dir_suffix
# "smooth_turn_taking" task uses "jchat_turn_taking" directory
declare -A TASK_MAP=(
    [pause_handling]=jchat_pause_handling
    [backchannel]=jchat_backchannel
    [smooth_turn_taking]=jchat_turn_taking
    [user_interruption]=jchat_user_interruption
)

for TASK in pause_handling backchannel smooth_turn_taking user_interruption; do
    DIR_NAME="${TASK_MAP[$TASK]}"
    TASK_DIR="${OUTPUT_DIR}/${DIR_NAME}"
    if [ -d "$TASK_DIR" ]; then
        echo "---- Evaluating: $TASK ----"
        uv run python "${PROJECT_DIR}/evaluation/evaluate_ja.py" \
            --task "$TASK" \
            --root_dir "$TASK_DIR"
        echo ""
    else
        echo "---- Skipping $TASK (no samples found at $TASK_DIR) ----"
        echo ""
    fi
done

# ----------------------------------------
# Step 3: v1.5 Evaluation Tasks
# ----------------------------------------
V15_DIR="${OUTPUT_DIR}/jchat_v15"
if [ -d "$V15_DIR" ]; then
    echo "==== Step 3: v1.5 Evaluation ===="

    echo "---- general_before_after ----"
    uv run python "${PROJECT_DIR}/evaluation/evaluate_ja.py" \
        --task general_before_after \
        --root_dir "$V15_DIR"
    echo ""

    echo "---- get_timing ----"
    uv run python "${PROJECT_DIR}/evaluation/get_timing.py" \
        --root_dir "$V15_DIR"
    echo ""

    echo "---- behavior ----"
    uv run python "${PROJECT_DIR}/evaluation/evaluate_ja.py" \
        --task behavior \
        --root_dir "$V15_DIR"
    echo ""
else
    echo "==== Step 3: v1.5 Evaluation (skipped - no v1.5 samples) ===="
fi

echo "========================================"
echo " Done!"
echo "========================================"
