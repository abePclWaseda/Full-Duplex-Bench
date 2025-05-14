import os
import json
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
from glob import glob
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from tqdm import tqdm
import argparse
from utils import adjust_pauses_for_hf_pipeline_output, load_audio, collate_fn


### REPLACE WITH YOUR OWN SETUP ###
your_huggingface_token = "YOUR_HUGGINGFACE_TOKEN"
###################################


def get_time_aligned_transcription(data_path, task):
    # List of audio file paths
    audio_paths = sorted(glob(f"{data_path}/*/output.wav"))

    # Set device and torch_dtype based on availability
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "nyrahealth/CrisperWhisper"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        token=your_huggingface_token,
        attn_implementation="eager",
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    # Create the pipeline with an increased batch size
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=1,  # Increase batch size if memory allows
        return_timestamps="word",
        torch_dtype=torch_dtype,
        device=device,
    )

    # Create a dataset from the list of audio paths
    data = [{"audio_path": path} for path in audio_paths]
    print(len(data))
    dataset = Dataset.from_list(data)

    # Apply preprocessing in parallel
    dataset = dataset.map(load_audio, num_proc=4)

    # Build the DataLoader for batching
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, collate_fn=collate_fn, shuffle=False)

    # Process the dataset in batches
    for pipeline_inputs, metadata in tqdm(dataloader):
        try:
            if task == "user_interruption":
                # read the interrupt.json file
                metadata_path = metadata[0].replace(
                    "output.wav", "interrupt.json"
                )
                # read the json file
                with open(metadata_path, "r") as f:
                    interrupt_metadata = json.load(f)
                timestamps = interrupt_metadata[0]["timestamp"]
                end_interrupt = timestamps[1]
                end_interrupt_idx = int(end_interrupt * pipeline_inputs[0]["sampling_rate"])

                # truncate the audio to the start_interrupt till the end
                pipeline_inputs[0]["array"] = pipeline_inputs[0]["array"][end_interrupt_idx:]

            # The pipeline now receives a list of dictionaries with only the required keys
            hf_pipeline_outputs = pipe(pipeline_inputs)

            # If we did a user interruption, shift all returned timestamps by the offset
            if task == "user_interruption":
                for out in hf_pipeline_outputs:
                    # shift each word‐level timestamp back to the original timeline
                    for chunk in out.get("chunks", []):
                        start, end = chunk["timestamp"]
                        chunk["timestamp"] = (start + end_interrupt, end + end_interrupt)

            # Iterate over outputs and corresponding metadata to save results
            for audio_path, output in zip(metadata, hf_pipeline_outputs):
                crisper_whisper_result = adjust_pauses_for_hf_pipeline_output(output)
                print(crisper_whisper_result)
                result_path = audio_path.replace("output.wav", "output.json")
                print(result_path)
                os.makedirs(os.path.dirname(result_path), exist_ok=True)
                with open(result_path, "w") as f:
                    json.dump(crisper_whisper_result, f, indent=4)
        except Exception as e:
            print(f"Error processing batch: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple argument parser")
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--task", type=str)
    args = parser.parse_args()

    get_time_aligned_transcription(args.root_dir, args.task)
