# Evaluation Data of Full-Duplex-Bench 

## Data Access

You can download the dataset from the following Google Drive link: [Dataset Download Link](https://drive.google.com/drive/folders/1DtoxMVO9_Y_nDs2peZtx3pw-U2qYgpd3?usp=sharing)

## Overview of data 

| Dataset     | Task                  | # of Samples |
|-------------|------------------------|--------------|
| Candor      | Pause Handling         | 216          |
| Candor      | Smooth Turn-Taking     | 119          |
| ICC         | Backchannel            | 55           |
| Synthetic   | User Interruption      | 200          |
| Synthetic   | Pause Handling         | 137          |


## File Descriptions

### The neccessary files for evaluation 🚨
- **input.wav**: Audio stream of user speech, intended as model input.
- **{TASK}.json**: Task-specific annotation files detailing "interrupt", "pause", or "turn_taking" labels.

### supplementary files
- **transcription.json**: Contains the transcribed text of the input audio and word-level timing alignments.

### Task-Specific Files ONLY for Synthetic User Interruption
- **context.wav** *(only in synthetic_user_interruption)*: Audio providing context before the interruption event.
- **interrupt.wav** *(only in synthetic_user_interruption)*: Audio segment representing the interruption.

## Dataset Structure
The dataset includes five primary interaction tasks from Candor, ICC, and synthetic data:
### Candor Pause Handling
```
candor_pause_handling/{ID}/
├── input.wav                 # User speech input stream
├── pause.json                # Annotations of pause events
└── transcription.json        # Text and time-aligned transcription
```

### Candor Turn Taking
```
candor_turn_taking/{ID}/
├── input.wav                 # User speech input stream
├── turn_taking.json          # Annotations of turn-taking events
└── transcription.json        # Text and time-aligned transcription
```

### ICC Backchannel
```
icc_backchannel/{ID}/
├── input.wav                 # User speech input stream
└── transcription.json        # Text and time-aligned transcription
```

### Synthetic Pause Handling
```
synthetic_pause_handling/{ID}/
├── input.wav                 # User speech input stream
├── pause.json                # Annotations of synthetic pause events
└── transcription.json        # Text and time-aligned transcription
```

### Synthetic User Interruption
```
synthetic_user_interruption/{ID}/
├── input.wav                 # User speech input stream
├── context.wav               # Contextual audio preceding interruption
├── interrupt.wav             # Interruption audio segment
└── interrupt.json            # Annotations of interruption events
```


## License 
The datasets are selected from [Candor](https://www.science.org/doi/full/10.1126/sciadv.adf3197) and [ICC](https://aclanthology.org/2024.findings-emnlp.909/). We continue to release this dataset under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license.

## Citation 
If you use this dataset, please cite it accordingly.

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

@article{reece2023candor,
  title={The CANDOR corpus: Insights from a large multimodal dataset of naturalistic conversation},
  author={Reece, Andrew and Cooney, Gus and Bull, Peter and Chung, Christine and Dawson, Bryn and Fitzpatrick, Casey and Glazer, Tamara and Knox, Dean and Liebscher, Alex and Marin, Sebastian},
  journal={Science Advances},
  volume={9},
  number={13},
  pages={eadf3197},
  year={2023},
  publisher={American Association for the Advancement of Science}
}

@inproceedings{umair-etal-2024-large,
    title = "Large Language Models Know What To Say But Not When To Speak",
    author = "Umair, Muhammad  and
      Sarathy, Vasanth  and
      Ruiter, Jan",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.909/",
    doi = "10.18653/v1/2024.findings-emnlp.909",
    pages = "15503--15514",
}

```

