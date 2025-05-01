# Full-Duplex-Bench Evaluation Script

This folder contains evaluation scripts for Full-Duplex-Bench tasks. Each evaluation script corresponds to a specific task.

## Supported Tasks

- ``backchannel``
- ``pause_handling``
- ``smooth_turn_taking``
- ``user_interruption``

## Requirements

Ensure you have run the `prepare_for_eval/asr.py` to obtain the time-aligned transcription `output.json`.

## Running the Evaluation

Use the following command format to run an evaluation script for a specific task. Run the script via the command line as follows:

```bash
python evaluate.py --task {TASK_NAME} --root_dir {DIRECTORY_PATH}
```

NOTE: if you are evaluating the `user_interruption`, you need to specify the `organization` and `api_key` for the `openai` API in the beginning of the python file. 

### Arguments

- `--task`: Specify the evaluation task. Must be one of:
  - `backchannel`
  - `pause_handling`
  - `smooth_turn_taking`
  - `user_interruption`

- `--root_dir`: Path to the root directory containing evaluation data and the time-aligned transcription `output.json`.

### Example of `root_dir` 👀
The example evaluation data is in `example_data` folder. You can see the data in `example_data/{TASK}_example/{ID}` to know the structure of the evaluation data. 


