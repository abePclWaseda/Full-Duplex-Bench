import os
import json
import numpy as np

from silero_vad import load_silero_vad, get_speech_timestamps
import torchaudio
import argparse

# Define the hyperparameters
window_size = 0.2
epsilon = 1e-10
time_threshold = 3


def eval_backchannel(data_dir):

    # JSD is skipped for Japanese (no GT distribution available)
    gt_distribution = None
    gt_distribution_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "icc_gt_distribution.json"
    )
    if os.path.exists(gt_distribution_path):
        try:
            with open(gt_distribution_path, "r") as f:
                gt_distribution = json.load(f)
        except Exception:
            gt_distribution = None

    vad_model = load_silero_vad()
    jsd_list = []
    TOR_list = []
    freq_list = []
    for spk in os.listdir(data_dir):
        if spk.startswith("."):
            continue
        spk_path = os.path.join(data_dir, spk)
        if not os.path.isdir(spk_path):
            continue

        out_wav_path = f"{data_dir}/{spk}/output.wav"
        if not os.path.exists(out_wav_path):
            raise FileNotFoundError("Required file 'output.wav' not found.")

        wav, sr = torchaudio.load(out_wav_path)

        segments = get_speech_timestamps(
            wav,
            vad_model,
            return_seconds=True,
        )

        backchannel_prediction = []

        json_file_name = f"{data_dir}/{spk}/output.json"

        if not os.path.exists(json_file_name):
            print("The output.json file does not exist")
            backchannel_prediction = []

        else:
            TOR = 0
            for segment in segments:
                start_time = segment["start"]
                end_time = segment["end"]
                duration = end_time - start_time
                print(f"duration is {duration}")

                if duration > time_threshold:  # means the model is formally speaking
                    TOR = 1
                    break
                else:
                    curr_text = []
                    with open(json_file_name, "r") as f:
                        data = json.load(f)
                        segments_cw = data["chunks"]

                        for segment_cw in segments_cw:
                            t_start = segment_cw["timestamp"][0]
                            t_end = segment_cw["timestamp"][1]

                            if t_start is None or (
                                t_end is None and t_start < end_time
                            ):
                                continue

                            if t_end is None:
                                t_end = t_start

                            if t_start >= start_time and t_end <= end_time:
                                curr_text.append(segment_cw["text"])
                            elif t_start <= end_time and t_end > end_time:
                                curr_text.append(segment_cw["text"])
                            elif t_start <= start_time and t_end > start_time:
                                curr_text.append(segment_cw["text"])

                    print(f"curr_text is {curr_text}")

                    # Japanese-adapted thresholds
                    num_words = len(curr_text)
                    if num_words > 8:  # was 3 for English words
                        TOR = 1
                    else:
                        if duration < 1:
                            if len(curr_text) <= 4:  # was 2 for English
                                TOR = 0
                            else:
                                TOR = 1
                        else:
                            TOR = 1

                    if len(backchannel_prediction) > 0:
                        print(backchannel_prediction)

                    if TOR == 1:
                        backchannel_prediction = []
                    else:
                        backchannel_prediction.append([start_time, end_time])

        print(f"backchannel prediction is {backchannel_prediction}")

        TOR_list.append(TOR)
        max_end_time = wav.shape[-1] / sr
        freq_list.append(len(backchannel_prediction) / max_end_time)

        # Skip JSD for Japanese (no ground-truth distribution)
        # Only compute if GT distribution exists and has this speaker key
        if gt_distribution is not None and spk in gt_distribution:
            if len(backchannel_prediction) == 0:
                js_divergence = 1
            else:
                from scipy.spatial.distance import jensenshannon
                from scipy.interpolate import interp1d

                time_intervals = [0 for i in range(int(max_end_time / window_size) + 1)]

                for interval in backchannel_prediction:
                    start = int(interval[0] / window_size)
                    end = int(interval[1] / window_size)
                    for i in range(start, end + 1):
                        if i < len(time_intervals):
                            time_intervals[i] += 1

                time_intervals = np.array(time_intervals)
                time_intervals = time_intervals + epsilon
                time_intervals = time_intervals / np.sum(time_intervals)
                time_intervals = list(time_intervals)

                gt_dist = gt_distribution[spk]

                x_gt = np.linspace(0, 1, len(gt_dist))
                x_pred = np.linspace(0, 1, len(time_intervals))

                interp_func = interp1d(
                    x_gt, gt_dist, kind="linear", fill_value="extrapolate"
                )
                gt_dist_resized = interp_func(x_pred)

                hist1 = np.array(time_intervals)
                hist2 = np.array(gt_dist_resized)

                js_divergence = jensenshannon(hist1, hist2)

            print(f"JSD: {js_divergence}")
            jsd_list.append(js_divergence)

    # Calculate statistics
    tor_mean, tor_std = np.mean(TOR_list), np.std(TOR_list)
    freq_mean, freq_std = np.mean(freq_list), np.std(freq_list)
    print("---------------------------------------------------")
    print("[Result]")
    if jsd_list:
        jsd_mean, jsd_std = np.mean(jsd_list), np.std(jsd_list)
        print(f"JSD - Mean: {jsd_mean:.4f} +/- {jsd_std:.4f}")
    else:
        print("JSD - Skipped (no ground-truth distribution for Japanese)")
    print(f"TOR - Mean: {tor_mean:.4f} +/- {tor_std:.4f}")
    print(f"Frequency - Mean: {freq_mean:.4f} +/- {freq_std:.4f}")

    print("\n[Raw Counts]")
    print(f"Number of samples: {len(TOR_list)}")
    print("---------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple argument parser")
    parser.add_argument("--root_dir", type=str)
    args = parser.parse_args()

    eval_backchannel(args.root_dir)
