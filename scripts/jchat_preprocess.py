#!/usr/bin/env python3
"""
j-chat data preprocessor for Full-Duplex-Bench.

Converts j-chat stereo WAV + character-level JSON into the benchmark format:
  - Splits stereo WAV into mono channels (ch0=speaker A, ch1=speaker B)
  - Groups character-level timestamps into utterance units
  - Generates output.json in benchmark format
  - Detects dialog events and produces per-task evaluation samples
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torchaudio

from jchat_event_detector import (
    detect_backchannels,
    detect_pauses,
    detect_smooth_turn_takings,
    detect_user_interruptions,
    detect_v15_overlaps,
)


# ----------------------------------------------------------------
# Audio helpers
# ----------------------------------------------------------------

def split_stereo_wav(stereo_path: str, out_dir: str):
    """Split stereo WAV into two mono WAV files.

    Returns (input_wav_path, output_wav_path) where:
      - input.wav  = ch0 (speaker A / user)
      - output.wav = ch1 (speaker B / evaluated model)
    """
    wav, sr = torchaudio.load(stereo_path)
    assert wav.shape[0] == 2, f"Expected stereo WAV, got {wav.shape[0]} channels"

    os.makedirs(out_dir, exist_ok=True)
    input_path = os.path.join(out_dir, "input.wav")
    output_path = os.path.join(out_dir, "output.wav")

    torchaudio.save(input_path, wav[0:1], sr)
    torchaudio.save(output_path, wav[1:2], sr)

    return input_path, output_path, sr


# ----------------------------------------------------------------
# JSON conversion helpers
# ----------------------------------------------------------------

def group_chars_to_utterances(jchat_data, speaker, gap_threshold=0.3):
    """Group consecutive characters from the same speaker into utterances.

    Splits at gaps > gap_threshold seconds between consecutive characters
    of the same speaker.
    """
    chars = [c for c in jchat_data if c["speaker"] == speaker]
    if not chars:
        return []

    utterances = []
    current_chars = [chars[0]]

    for i in range(1, len(chars)):
        prev_end = current_chars[-1]["end"]
        curr_start = chars[i]["start"]

        if curr_start - prev_end > gap_threshold:
            utterances.append(current_chars)
            current_chars = [chars[i]]
        else:
            current_chars.append(chars[i])

    if current_chars:
        utterances.append(current_chars)

    return utterances


def make_transcript_json(jchat_data, speaker):
    """Convert j-chat character data to benchmark output.json format.

    Each character becomes one chunk:
    {
      "text": "full text...",
      "chunks": [
        {"text": "X", "timestamp": [start, end]},
        ...
      ]
    }
    """
    chars = [c for c in jchat_data if c["speaker"] == speaker]
    if not chars:
        return {"text": "", "chunks": []}

    full_text = "".join(c["word"] for c in chars)
    chunks = [
        {"text": c["word"], "timestamp": [c["start"], c["end"]]}
        for c in chars
    ]

    return {"text": full_text, "chunks": chunks}


def make_transcript_json_for_slice(jchat_data, speaker, time_start, time_end):
    """Make transcript JSON for a time window, adjusting timestamps to be relative."""
    chars = [
        c for c in jchat_data
        if c["speaker"] == speaker
        and c["start"] >= time_start - 0.05  # small tolerance
        and c["end"] <= time_end + 0.05
    ]
    if not chars:
        return {"text": "", "chunks": []}

    full_text = "".join(c["word"] for c in chars)
    chunks = [
        {
            "text": c["word"],
            "timestamp": [
                round(max(0.0, c["start"] - time_start), 3),
                round(max(0.0, c["end"] - time_start), 3),
            ],
        }
        for c in chars
    ]

    return {"text": full_text, "chunks": chunks}


def slice_and_save_wav(stereo_path, channel, time_start, time_end, out_path, sr=16000):
    """Extract a time window from one channel of a stereo WAV and save."""
    wav, orig_sr = torchaudio.load(stereo_path)
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)

    start_sample = max(0, int(time_start * sr))
    end_sample = min(wav.shape[1], int(time_end * sr))

    mono = wav[channel:channel + 1, start_sample:end_sample]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torchaudio.save(out_path, mono, sr)


def slice_and_save_mix(stereo_path, time_start, time_end, out_path, sr=16000):
    """Extract a time window, mix both channels, and save as mono."""
    wav, orig_sr = torchaudio.load(stereo_path)
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)

    start_sample = max(0, int(time_start * sr))
    end_sample = min(wav.shape[1], int(time_end * sr))

    mixed = wav[:, start_sample:end_sample].mean(dim=0, keepdim=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torchaudio.save(out_path, mixed, sr)


# ----------------------------------------------------------------
# Event sample builders
# ----------------------------------------------------------------

CONTEXT_BEFORE = 3.0  # seconds of context before event
CONTEXT_AFTER = 5.0   # seconds of context after event


def _time_window(event_start, event_end, total_duration):
    """Compute a time window around an event."""
    win_start = max(0.0, event_start - CONTEXT_BEFORE)
    win_end = min(total_duration, event_end + CONTEXT_AFTER)
    return win_start, win_end


def build_pause_samples(events, stereo_path, jchat_data, file_id, out_root, total_dur):
    """Build pause handling evaluation samples."""
    task_dir = os.path.join(out_root, "jchat_pause_handling")
    for i, ev in enumerate(events):
        sample_id = f"{file_id}_pause_{i}"
        sample_dir = os.path.join(task_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        win_start, win_end = _time_window(ev["start"], ev["end"], total_dur)

        # Save output.wav (speaker B channel)
        slice_and_save_wav(stereo_path, 1, win_start, win_end,
                           os.path.join(sample_dir, "output.wav"))

        # Save output.json (speaker B transcript in this window)
        output_json = make_transcript_json_for_slice(jchat_data, "B", win_start, win_end)
        with open(os.path.join(sample_dir, "output.json"), "w", encoding="utf-8") as f:
            json.dump(output_json, f, ensure_ascii=False, indent=2)

        # Save pause.json
        pause_json = [{
            "text": "[PAUSE]",
            "timestamp": [
                round(ev["start"] - win_start, 3),
                round(ev["end"] - win_start, 3),
            ]
        }]
        with open(os.path.join(sample_dir, "pause.json"), "w", encoding="utf-8") as f:
            json.dump(pause_json, f, ensure_ascii=False, indent=2)

    return len(events)


def build_turn_taking_samples(events, stereo_path, jchat_data, file_id, out_root, total_dur):
    """Build smooth turn-taking evaluation samples."""
    task_dir = os.path.join(out_root, "jchat_turn_taking")
    for i, ev in enumerate(events):
        sample_id = f"{file_id}_tt_{i}"
        sample_dir = os.path.join(task_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        win_start, win_end = _time_window(ev["a_end"], ev["b_start"], total_dur)

        # Save output.wav (speaker B)
        slice_and_save_wav(stereo_path, 1, win_start, win_end,
                           os.path.join(sample_dir, "output.wav"))

        # Save output.json
        output_json = make_transcript_json_for_slice(jchat_data, "B", win_start, win_end)
        with open(os.path.join(sample_dir, "output.json"), "w", encoding="utf-8") as f:
            json.dump(output_json, f, ensure_ascii=False, indent=2)

        # Save turn_taking.json
        tt_json = [{
            "text": "[TURN-TAKING]",
            "timestamp": [
                round(ev["a_end"] - win_start, 3),
                round(ev["b_start"] - win_start, 3),
            ]
        }]
        with open(os.path.join(sample_dir, "turn_taking.json"), "w", encoding="utf-8") as f:
            json.dump(tt_json, f, ensure_ascii=False, indent=2)

    return len(events)


def build_backchannel_samples(events, stereo_path, jchat_data, file_id, out_root, total_dur):
    """Build backchannel evaluation samples."""
    task_dir = os.path.join(out_root, "jchat_backchannel")
    for i, ev in enumerate(events):
        sample_id = f"{file_id}_bc_{i}"
        sample_dir = os.path.join(task_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        # Use a wider window for backchannel - the whole A utterance + some context
        win_start = max(0.0, ev["a_utt_start"] - 1.0)
        win_end = min(total_dur, ev["a_utt_end"] + 2.0)

        # Save output.wav (speaker B)
        slice_and_save_wav(stereo_path, 1, win_start, win_end,
                           os.path.join(sample_dir, "output.wav"))

        # Save output.json
        output_json = make_transcript_json_for_slice(jchat_data, "B", win_start, win_end)
        with open(os.path.join(sample_dir, "output.json"), "w", encoding="utf-8") as f:
            json.dump(output_json, f, ensure_ascii=False, indent=2)

    return len(events)


def build_interruption_samples(events, stereo_path, jchat_data, file_id, out_root, total_dur):
    """Build user interruption evaluation samples."""
    task_dir = os.path.join(out_root, "jchat_user_interruption")
    for i, ev in enumerate(events):
        sample_id = f"{file_id}_int_{i}"
        sample_dir = os.path.join(task_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        # Window: from some context before interruption to after
        win_start = max(0.0, ev["a_interrupt_start"] - CONTEXT_BEFORE)
        win_end = min(total_dur, ev["a_interrupt_end"] + CONTEXT_AFTER)

        # Save output.wav (speaker B after interruption)
        slice_and_save_wav(stereo_path, 1, win_start, win_end,
                           os.path.join(sample_dir, "output.wav"))

        # Save output.json (B's response after interruption)
        output_json = make_transcript_json_for_slice(jchat_data, "B", win_start, win_end)
        with open(os.path.join(sample_dir, "output.json"), "w", encoding="utf-8") as f:
            json.dump(output_json, f, ensure_ascii=False, indent=2)

        # Save interrupt.json
        interrupt_json = [{
            "context": ev["b_context_text"],
            "interrupt": ev["a_interrupt_text"],
            "timestamp": [
                round(ev["a_interrupt_start"] - win_start, 3),
                round(ev["a_interrupt_end"] - win_start, 3),
            ]
        }]
        with open(os.path.join(sample_dir, "interrupt.json"), "w", encoding="utf-8") as f:
            json.dump(interrupt_json, f, ensure_ascii=False, indent=2)

    return len(events)


def build_v15_samples(events, stereo_path, jchat_data, file_id, out_root, total_dur):
    """Build v1.5 overlap evaluation samples."""
    task_dir = os.path.join(out_root, "jchat_v15")
    for i, ev in enumerate(events):
        sample_id = f"{file_id}_v15_{i}"
        sample_dir = os.path.join(task_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        win_start = max(0.0, ev["overlap_start"] - CONTEXT_BEFORE)
        win_end = min(total_dur, ev["overlap_end"] + CONTEXT_AFTER)

        # input.wav = ch0+ch1 mixed (noisy)
        slice_and_save_mix(stereo_path, win_start, win_end,
                           os.path.join(sample_dir, "input.wav"))

        # clean_input.wav = ch0 only (clean user)
        slice_and_save_wav(stereo_path, 0, win_start, win_end,
                           os.path.join(sample_dir, "clean_input.wav"))

        # output.wav = ch1 (model output, noisy context)
        slice_and_save_wav(stereo_path, 1, win_start, win_end,
                           os.path.join(sample_dir, "output.wav"))

        # clean_output.wav = ch1 (same as output since stereo-separated)
        slice_and_save_wav(stereo_path, 1, win_start, win_end,
                           os.path.join(sample_dir, "clean_output.wav"))

        # Transcript JSONs
        input_json = make_transcript_json_for_slice(jchat_data, "A", win_start, win_end)
        output_json = make_transcript_json_for_slice(jchat_data, "B", win_start, win_end)

        with open(os.path.join(sample_dir, "input.json"), "w", encoding="utf-8") as f:
            json.dump(input_json, f, ensure_ascii=False, indent=2)
        with open(os.path.join(sample_dir, "clean_input.json"), "w", encoding="utf-8") as f:
            json.dump(input_json, f, ensure_ascii=False, indent=2)
        with open(os.path.join(sample_dir, "output.json"), "w", encoding="utf-8") as f:
            json.dump(output_json, f, ensure_ascii=False, indent=2)
        with open(os.path.join(sample_dir, "clean_output.json"), "w", encoding="utf-8") as f:
            json.dump(output_json, f, ensure_ascii=False, indent=2)

        # metadata.json
        metadata = {
            "context_text": ev.get("context_text", ""),
            "current_turn_text": ev.get("current_turn_text", ""),
            "timestamps": [
                round(ev["overlap_start"] - win_start, 3),
                round(ev["overlap_end"] - win_start, 3),
            ]
        }
        with open(os.path.join(sample_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    return len(events)


# ----------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------

def get_total_duration(stereo_path):
    """Get total duration in seconds."""
    info = torchaudio.info(stereo_path)
    return info.num_frames / info.sample_rate


def process_one_file(wav_path, json_path, output_dir):
    """Process a single j-chat file pair."""
    file_id = Path(wav_path).stem

    # Load j-chat JSON
    with open(json_path, "r", encoding="utf-8") as f:
        jchat_data = json.load(f)

    total_dur = get_total_duration(wav_path)
    print(f"Processing {file_id}: {total_dur:.1f}s, {len(jchat_data)} characters")

    # Detect events
    pauses = detect_pauses(jchat_data)
    turn_takings = detect_smooth_turn_takings(jchat_data)
    backchannels = detect_backchannels(jchat_data)
    interruptions = detect_user_interruptions(jchat_data)
    v15_overlaps = detect_v15_overlaps(jchat_data)

    print(f"  Detected: {len(pauses)} pauses, {len(turn_takings)} turn-takings, "
          f"{len(backchannels)} backchannels, {len(interruptions)} interruptions, "
          f"{len(v15_overlaps)} v1.5 overlaps")

    # Build samples
    n_pause = build_pause_samples(pauses, wav_path, jchat_data, file_id, output_dir, total_dur)
    n_tt = build_turn_taking_samples(turn_takings, wav_path, jchat_data, file_id, output_dir, total_dur)
    n_bc = build_backchannel_samples(backchannels, wav_path, jchat_data, file_id, output_dir, total_dur)
    n_int = build_interruption_samples(interruptions, wav_path, jchat_data, file_id, output_dir, total_dur)
    n_v15 = build_v15_samples(v15_overlaps, wav_path, jchat_data, file_id, output_dir, total_dur)

    print(f"  Built samples: pause={n_pause}, tt={n_tt}, bc={n_bc}, int={n_int}, v15={n_v15}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess j-chat data for Full-Duplex-Bench")
    parser.add_argument("--jchat_dir", type=str, required=True,
                        help="Directory containing j-chat WAV+JSON files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for evaluation samples")
    args = parser.parse_args()

    jchat_dir = Path(args.jchat_dir)
    wav_files = sorted(jchat_dir.glob("*.wav"))

    if not wav_files:
        print(f"No WAV files found in {jchat_dir}")
        return

    for wav_path in wav_files:
        json_path = wav_path.with_suffix(".json")
        if not json_path.exists():
            print(f"Warning: No JSON file found for {wav_path}, skipping")
            continue

        process_one_file(str(wav_path), str(json_path), args.output_dir)

    print("\nDone! Output written to:", args.output_dir)


if __name__ == "__main__":
    main()
