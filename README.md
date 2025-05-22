# Full-Duplex-Bench: A Benchmark to Evaluate Full-duplex Spoken Dialogue Models on Turn-taking Capabilities
> Authors: [Guan-Ting Lin](https://daniellin94144.github.io/), [Jiachen Lian*](https://jlian2.github.io/), [Tingle Li*](https://tinglok.netlify.app/), [Qirui Wang*](https://www.linkedin.com/in/qrw-160509207/), [Gopala Anumanchipalli](https://www2.eecs.berkeley.edu/Faculty/Homepages/gopala.html), [Alexander H. Liu](https://alexander-h-liu.github.io/), [Hung-yi Lee](https://speech.ee.ntu.edu.tw/~hylee/index.html)

## TL;DR
A benchmark to evaluate full-duplex spoken dialogue models on pause handling, backchanneling, turn-taking, and user interruptions.

[![arXiv](https://img.shields.io/badge/arXiv-2409.06666-b31b1b.svg?logo=arXiv)]([https://arxiv.org/abs/2409.06666](https://arxiv.org/abs/2503.04721))
[![code](https://img.shields.io/badge/Github-Code-keygen.svg?logo=github)](https://github.com/DanielLin94144/Full-Duplex-Bench)

<div align="center"><img src="https://github.com/user-attachments/assets/70b6525c-61ee-4c48-a1fb-59dc6dfe85cc" width="85%"/></div>

## Highlight 💡
Spoken dialogue modeling presents unique challenges beyond text-based language modeling, requiring real-time interaction capabilities such as turn-taking, backchanneling, and pause handling. Existing evaluation methods primarily focus on half-duplex processing, leaving the full-duplex capabilities of modern models underexplored. 

Full-Duplex-Bench provides an open and standardized benchmark to assess these interactive behaviors systematically. Our framework leverages automatic metrics for consistent and reproducible evaluations across key dimensions:

- **Pause Handling:** Evaluates whether models correctly identify natural pauses without interrupting the speaker.
- **Backchanneling:** Measures how well models generate short acknowledgments at appropriate times.
- **Smooth Turn-Taking:** Assesses response timing to ensure fluid dialogue transitions.
- **User Interruption Management:** Tests a model's ability to adapt to user interruptions and shift focus appropriately.

<div align="center"><img src="https://github.com/user-attachments/assets/e936d330-1105-42fc-b5c6-d7ee8f40d27c" width="65%"/></div>

We will provide a set of curated datasets, automatic evaluation scripts, and baseline results on multiple full-duplex models in upcoming releases. Our benchmark aims to drive progress in spoken dialogue systems by encouraging fair and open evaluation practices. 

The audio demo samples can be found in [[Demo]](https://full-duplex-bench.github.io/).

## Timeline ⏱
- **(2025/4/30) Dataset Release:** ✅ see under the `dataset` folder
- **(2025/4/30) Evaluation Codebase:** ✅ see under the `evaluation` folder
- **(Ongoing) Expanded Model Evaluations & Community Contributions**

Stay tuned for upcoming releases!

## Getting Started 🏁
### Installation
```
conda create -n full-duplex-bench python=3.8
conda activate full-duplex-bench
pip install -r requirements.txt
```

### Step-by-step Instruction
#### 1. Model Inference
The goal of model inference is to let the model generate the time-synchronous `output.wav` given the audio stream of user speech (`input.wav`). You can use you own model to generate the output speech for evaluation.

We will provide the example inference code of Freeze-omni under `model_inference/freeze-omni` for different tasks. 

#### 2. Prepare for Evaluation with time-aligned transcription
Under `get_transcript` folder, you can find `asr.py` to obtain the time-aligned transcription for the model generated audio. For more details please see the readme in the folder.

#### 3. Running Evaluations
Under `evaluation` folder, please see the readme file in the folder for detailed instruction to run the evaluation for each tasks.

## Citation 📖
If you have any questions, please feel free to submit an issue or contact Guan-Ting Lin (daniel094144@gmail.com)

If you found this research helpful, please consider citing our work:

```
@misc{lin2025fullduplexbenchbenchmarkevaluatefullduplex,
      title={Full-Duplex-Bench: A Benchmark to Evaluate Full-duplex Spoken Dialogue Models on Turn-taking Capabilities}, 
      author={Guan-Ting Lin and Jiachen Lian and Tingle Li and Qirui Wang and Gopala Anumanchipalli and Alexander H. Liu and Hung-yi Lee},
      year={2025},
      eprint={2503.04721},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.04721}, 
}
```

