## Run ASR to Prepare for Evaluation
An ASR script is provided to process the audio outputs. Ensure your data follows the structure:
```
{OUTPUT_WAV_ROOT_PATH}/{ID}/output.wav
```

### Note
Due to the potential Whisper version mismatch in [transformers](https://github.com/huggingface/transformers), we recommend reinstalling it as follows:
``` bash
chmod +x run_patch.sh
bash run_patch.sh
```

### Running ASR

Use the following command to run the ASR script:

```bash
python asr.py --root_dir {OUTPUT_WAV_ROOT_PATH}
```

Replace `{OUTPUT_WAV_ROOT_PATH}` with the actual root path containing your output WAV files.

Note that in the `asr.py` file, you have to specify your huggingface API key in the beginning of the python file.
