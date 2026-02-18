#!/usr/bin/env python3
"""
Dialog event auto-detector for j-chat data.

Detects the following events from character-level timestamped j-chat JSON:
  - Pause Handling: gaps in speaker A's speech where B stays silent
  - Smooth Turn-Taking: A ends → B starts speaking
  - Backchannel: short B utterances during A's speech
  - User Interruption: A interrupts during B's long utterance
  - v1.5 Overlap: simultaneous speech from both speakers
"""


# ----------------------------------------------------------------
# Common: build speaker timeline
# ----------------------------------------------------------------

def group_chars_to_utterances(jchat_data, speaker, gap_threshold=0.3):
    """Group consecutive characters from the same speaker into utterances.

    Returns list of dicts: [{"start", "end", "text", "chars": [...]}]
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
            utt = _chars_to_utterance(current_chars)
            utterances.append(utt)
            current_chars = [chars[i]]
        else:
            current_chars.append(chars[i])

    if current_chars:
        utterances.append(_chars_to_utterance(current_chars))

    return utterances


def _chars_to_utterance(chars):
    """Convert a list of character dicts to an utterance dict."""
    text = "".join(c["word"] for c in chars)
    return {
        "start": chars[0]["start"],
        "end": chars[-1]["end"],
        "text": text,
        "num_chars": len(chars),
        "chars": chars,
    }


def build_speaker_timeline(jchat_data):
    """Build timelines for both speakers.

    Returns:
        a_utts: list of speaker A utterances
        b_utts: list of speaker B utterances
    """
    a_utts = group_chars_to_utterances(jchat_data, "A", gap_threshold=0.3)
    b_utts = group_chars_to_utterances(jchat_data, "B", gap_threshold=0.3)
    return a_utts, b_utts


def _intervals_overlap(s1, e1, s2, e2):
    """Check if two time intervals overlap."""
    return s1 < e2 and s2 < e1


def _get_overlap(s1, e1, s2, e2):
    """Get the overlap interval, or None if no overlap."""
    s = max(s1, s2)
    e = min(e1, e2)
    if s < e:
        return s, e
    return None


def _b_silent_during(b_utts, start, end, tolerance=0.1):
    """Check if speaker B has no substantial speech during [start, end]."""
    for b in b_utts:
        if _intervals_overlap(b["start"], b["end"], start, end):
            # B has speech overlapping this interval
            overlap = _get_overlap(b["start"], b["end"], start, end)
            if overlap and (overlap[1] - overlap[0]) > tolerance:
                return False
    return True


# ----------------------------------------------------------------
# 2a. Pause Handling detection
# ----------------------------------------------------------------

def detect_pauses(jchat_data, min_gap=0.5, max_gap=3.0):
    """Detect pauses in speaker A's speech where B stays silent.

    A pause is a gap between consecutive A utterances (0.5s~3.0s)
    during which B has no substantial speech.
    """
    a_utts, b_utts = build_speaker_timeline(jchat_data)

    pauses = []
    for i in range(len(a_utts) - 1):
        gap_start = a_utts[i]["end"]
        gap_end = a_utts[i + 1]["start"]
        gap_duration = gap_end - gap_start

        if min_gap <= gap_duration <= max_gap:
            if _b_silent_during(b_utts, gap_start, gap_end):
                pauses.append({
                    "start": gap_start,
                    "end": gap_end,
                    "duration": gap_duration,
                    "a_before_text": a_utts[i]["text"],
                    "a_after_text": a_utts[i + 1]["text"],
                })

    return pauses


# ----------------------------------------------------------------
# 2b. Smooth Turn-Taking detection
# ----------------------------------------------------------------

def detect_smooth_turn_takings(jchat_data, min_gap=0.3, max_gap=3.0,
                                min_chars=5):
    """Detect smooth turn-taking points: A ends, then B starts.

    Conditions:
    - Gap between A end and B start is 0.3s~3.0s
    - Both A and B utterances have >= min_chars characters
    """
    a_utts, b_utts = build_speaker_timeline(jchat_data)

    events = []
    for a_utt in a_utts:
        if a_utt["num_chars"] < min_chars:
            continue

        a_end = a_utt["end"]

        # Find the first B utterance that starts after A ends
        for b_utt in b_utts:
            if b_utt["start"] > a_end:
                gap = b_utt["start"] - a_end
                if min_gap <= gap <= max_gap and b_utt["num_chars"] >= min_chars:
                    events.append({
                        "a_end": a_end,
                        "b_start": b_utt["start"],
                        "gap": gap,
                        "a_text": a_utt["text"],
                        "b_text": b_utt["text"],
                    })
                break  # only consider the first B utterance after A ends

    return events


# ----------------------------------------------------------------
# 2c. Backchannel detection
# ----------------------------------------------------------------

# Japanese backchannel patterns
BACKCHANNEL_PATTERNS = [
    "うん", "ううん", "うーん", "そう", "そうそう", "ああ", "あー",
    "はい", "ええ", "なるほど", "ふーん", "へー", "ほー",
    "そうだね", "そうですね", "そっか", "うんうん", "はいはい",
    "ん", "んー", "あ", "え",
]


def detect_backchannels(jchat_data, max_chars=6, max_duration=1.5):
    """Detect backchannels: short B utterances during A's speech.

    Conditions:
    - B utterance is <= max_chars characters
    - B utterance duration is <= max_duration seconds
    - B utterance occurs during (overlaps with) an A utterance
    """
    a_utts, b_utts = build_speaker_timeline(jchat_data)

    events = []
    for b_utt in b_utts:
        if b_utt["num_chars"] > max_chars:
            continue
        b_dur = b_utt["end"] - b_utt["start"]
        if b_dur > max_duration:
            continue

        # Check if this B utterance overlaps with any A utterance
        for a_utt in a_utts:
            if _intervals_overlap(a_utt["start"], a_utt["end"],
                                   b_utt["start"], b_utt["end"]):
                # Check if text matches backchannel patterns (optional, but informative)
                b_text_clean = b_utt["text"].replace("。", "").replace("、", "").strip()
                is_pattern = any(b_text_clean == p for p in BACKCHANNEL_PATTERNS)

                events.append({
                    "b_start": b_utt["start"],
                    "b_end": b_utt["end"],
                    "b_text": b_utt["text"],
                    "b_duration": b_dur,
                    "a_utt_start": a_utt["start"],
                    "a_utt_end": a_utt["end"],
                    "a_text": a_utt["text"],
                    "is_known_pattern": is_pattern,
                })
                break  # count once per B utterance

    return events


# ----------------------------------------------------------------
# 2d. User Interruption detection
# ----------------------------------------------------------------

def detect_user_interruptions(jchat_data, min_b_duration=2.0):
    """Detect user interruptions: A starts speaking while B is in a long utterance.

    Conditions:
    - B is in an utterance of >= min_b_duration seconds
    - A starts a new utterance during B's speech
    """
    a_utts, b_utts = build_speaker_timeline(jchat_data)

    events = []
    for b_utt in b_utts:
        b_dur = b_utt["end"] - b_utt["start"]
        if b_dur < min_b_duration:
            continue

        # Find A utterances that start during B's long utterance
        for a_utt in a_utts:
            if (b_utt["start"] < a_utt["start"] < b_utt["end"]
                    and a_utt["num_chars"] >= 3):
                events.append({
                    "b_context_text": b_utt["text"],
                    "b_utt_start": b_utt["start"],
                    "b_utt_end": b_utt["end"],
                    "a_interrupt_text": a_utt["text"],
                    "a_interrupt_start": a_utt["start"],
                    "a_interrupt_end": a_utt["end"],
                })
                break  # one interruption per B utterance

    return events


# ----------------------------------------------------------------
# 2e. v1.5 Overlap event detection
# ----------------------------------------------------------------

def detect_v15_overlaps(jchat_data, min_overlap_duration=0.3):
    """Detect overlap regions where both speakers talk simultaneously.

    Returns events with overlap intervals and surrounding context.
    """
    a_utts, b_utts = build_speaker_timeline(jchat_data)

    events = []
    for a_utt in a_utts:
        for b_utt in b_utts:
            overlap = _get_overlap(a_utt["start"], a_utt["end"],
                                    b_utt["start"], b_utt["end"])
            if overlap is None:
                continue

            overlap_start, overlap_end = overlap
            overlap_dur = overlap_end - overlap_start

            if overlap_dur < min_overlap_duration:
                continue

            events.append({
                "overlap_start": overlap_start,
                "overlap_end": overlap_end,
                "overlap_duration": overlap_dur,
                "context_text": a_utt["text"],
                "current_turn_text": b_utt["text"],
                "a_utt_start": a_utt["start"],
                "a_utt_end": a_utt["end"],
                "b_utt_start": b_utt["start"],
                "b_utt_end": b_utt["end"],
            })

    # Deduplicate overlapping events (keep unique overlap intervals)
    events.sort(key=lambda e: e["overlap_start"])
    deduped = []
    for ev in events:
        if deduped and ev["overlap_start"] < deduped[-1]["overlap_end"]:
            # Merge with previous if overlapping
            if ev["overlap_end"] > deduped[-1]["overlap_end"]:
                deduped[-1]["overlap_end"] = ev["overlap_end"]
                deduped[-1]["overlap_duration"] = (
                    deduped[-1]["overlap_end"] - deduped[-1]["overlap_start"]
                )
        else:
            deduped.append(ev)

    return deduped


# ----------------------------------------------------------------
# CLI for standalone testing
# ----------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Detect dialog events in j-chat data")
    parser.add_argument("--json_path", type=str, required=True,
                        help="Path to j-chat JSON file")
    args = parser.parse_args()

    with open(args.json_path, "r", encoding="utf-8") as f:
        jchat_data = json.load(f)

    print("=== Speaker Timeline ===")
    a_utts, b_utts = build_speaker_timeline(jchat_data)
    print(f"Speaker A: {len(a_utts)} utterances")
    for u in a_utts[:5]:
        print(f"  [{u['start']:.2f}-{u['end']:.2f}] ({u['num_chars']} chars) {u['text'][:40]}")
    print(f"Speaker B: {len(b_utts)} utterances")
    for u in b_utts[:5]:
        print(f"  [{u['start']:.2f}-{u['end']:.2f}] ({u['num_chars']} chars) {u['text'][:40]}")

    print("\n=== Pauses ===")
    pauses = detect_pauses(jchat_data)
    print(f"Found {len(pauses)} pauses")
    for p in pauses[:5]:
        print(f"  [{p['start']:.2f}-{p['end']:.2f}] gap={p['duration']:.2f}s")

    print("\n=== Turn-Takings ===")
    tts = detect_smooth_turn_takings(jchat_data)
    print(f"Found {len(tts)} turn-takings")
    for t in tts[:5]:
        print(f"  A_end={t['a_end']:.2f} B_start={t['b_start']:.2f} gap={t['gap']:.2f}s")

    print("\n=== Backchannels ===")
    bcs = detect_backchannels(jchat_data)
    print(f"Found {len(bcs)} backchannels")
    for b in bcs[:5]:
        print(f"  [{b['b_start']:.2f}-{b['b_end']:.2f}] '{b['b_text']}' "
              f"(pattern={b['is_known_pattern']})")

    print("\n=== User Interruptions ===")
    ints = detect_user_interruptions(jchat_data)
    print(f"Found {len(ints)} interruptions")
    for i in ints[:5]:
        print(f"  A interrupts at {i['a_interrupt_start']:.2f} "
              f"during B [{i['b_utt_start']:.2f}-{i['b_utt_end']:.2f}]")
        print(f"    B context: {i['b_context_text'][:40]}")
        print(f"    A interrupt: {i['a_interrupt_text'][:40]}")

    print("\n=== v1.5 Overlaps ===")
    ovs = detect_v15_overlaps(jchat_data)
    print(f"Found {len(ovs)} overlap events")
    for o in ovs[:5]:
        print(f"  [{o['overlap_start']:.2f}-{o['overlap_end']:.2f}] "
              f"dur={o['overlap_duration']:.2f}s")
