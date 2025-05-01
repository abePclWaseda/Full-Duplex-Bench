import torchaudio
import numpy as np


# Define your adjustment function
def adjust_pauses_for_hf_pipeline_output(pipeline_output, split_threshold=0.12):
    adjusted_chunks = pipeline_output["chunks"].copy()
    for i in range(len(adjusted_chunks) - 1):
        current_chunk = adjusted_chunks[i]
        next_chunk = adjusted_chunks[i + 1]
        current_start, current_end = current_chunk["timestamp"]
        next_start, next_end = next_chunk["timestamp"]
        pause_duration = next_start - current_end
        if pause_duration > 0:
            distribute = (
                split_threshold / 2
                if pause_duration > split_threshold
                else pause_duration / 2
            )
            adjusted_chunks[i]["timestamp"] = (current_start, current_end + distribute)
            adjusted_chunks[i + 1]["timestamp"] = (next_start - distribute, next_end)
    pipeline_output["chunks"] = adjusted_chunks
    return pipeline_output


# Define a function to load and preprocess each audio file
def load_audio(example):
    audio_path = example["audio_path"]
    audio_tensor, sr = torchaudio.load(audio_path)
    # Resample if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        audio_tensor = resampler(audio_tensor)
    # Use the first channel if multi-channel
    if audio_tensor.size(0) == 2:
        audio_tensor = audio_tensor[1, :]
    elif audio_tensor.size(0) == 1:
        audio_tensor = audio_tensor[0, :]
    # Convert tensor to numpy array
    audio_array = audio_tensor.numpy()
    example["array"] = audio_array
    example["sampling_rate"] = 16000
    return example


# Modify the collate function to separate pipeline inputs from metadata
def collate_fn(batch):
    # Convert the stored list back into a numpy array for each sample.
    pipeline_inputs = [
        {"array": np.array(sample["array"]), "sampling_rate": sample["sampling_rate"]}
        for sample in batch
    ]
    metadata = [sample["audio_path"] for sample in batch]
    return pipeline_inputs, metadata
