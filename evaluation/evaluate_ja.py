#!/usr/bin/env python3
"""
Japanese evaluation dispatcher for Full-Duplex-Bench.

Routes evaluation tasks to the appropriate _ja.py scripts.
Language-independent tasks (general_before_after, get_timing) use original scripts.
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Run Japanese evaluation tasks.")

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[
            "backchannel",
            "pause_handling",
            "smooth_turn_taking",
            "user_interruption",
            "behavior",
            "general_before_after",
        ],
        help="Evaluation task to perform.",
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory containing data for evaluation.",
    )

    args = parser.parse_args()

    # Add evaluation directory to path for imports
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)

    if args.task == "backchannel":
        from eval_backchannel_ja import eval_backchannel

        eval_backchannel(args.root_dir)

    elif args.task == "pause_handling":
        from eval_pause_handling_ja import eval_pause_handling

        eval_pause_handling(args.root_dir)

    elif args.task == "smooth_turn_taking":
        from eval_smooth_turn_taking_ja import eval_smooth_turn_taking

        eval_smooth_turn_taking(args.root_dir)

    elif args.task == "user_interruption":
        from openai import OpenAI
        from eval_user_interruption_ja import eval_user_interruption

        api_key = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY")
        client = OpenAI(
            api_key=api_key,
        )
        eval_user_interruption(args.root_dir, client)

    elif args.task == "general_before_after":
        from eval_general_before_after import eval_general_all_split

        config = {
            "trim_silence": True,
            "squim": False,
            "utmosv2": True,
            "speaking_rate": True,
            "trim_mode": "silero",
            "agg": {
                "mode": "trim",
                "trim_prop": 0.05,
            },
            "pitch": True,
            "intensity": True,
        }

        aggregate = True
        output = eval_general_all_split(config, args.root_dir, aggregate=aggregate)
        print("General Evaluation Output:", output)
        if aggregate:
            dir_name = args.root_dir.rstrip("/").split("/")[-1]
            log_path = f"{dir_name}_general_ja.log"
            with open(log_path, "w", encoding="utf-8") as f:
                for key, value in output.items():
                    line = f"{key}: {value}\n"
                    print(line, end="")
                    f.write(line)

            # Also report cpm (characters per minute) instead of wpm
            if "post" in output and "wpm" in output["post"]:
                print(f"\n[Note] For Japanese, 'wpm' metric represents "
                      f"cpm (characters per minute): {output['post']['wpm']:.1f}")

    elif args.task == "behavior":
        from openai import OpenAI
        from eval_behavior_ja import eval_behavior_all

        api_key = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY")
        client = OpenAI(
            api_key=api_key,
        )
        output = eval_behavior_all(
            args.root_dir, client, task=args.task, aggregate=True
        )

        dir_name = args.root_dir.rstrip("/").split("/")[-1]
        log_path = f"{dir_name}_{args.task}_ja.log"

        with open(log_path, "w", encoding="utf-8") as f:
            for ax in ["C"]:
                line = f"Ratios ({ax}-axis): {output[ax]}\n"
                print(line, end="")
                f.write(line)


if __name__ == "__main__":
    main()
