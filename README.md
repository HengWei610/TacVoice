# 🏅 Sports Speech Recognition for Athlete Performance Analysis

A modular deep learning framework for multimodal speech-based performance analysis in sports, built upon the paper:

> **"Advanced Speech Recognition for Athlete Performance Analysis in Sports Science"**  

---

## 📌 Features

- 🎙️ Robust speech recognition in noisy, dynamic environments
- ⚽ Multimodal performance modeling using spatial, tactical, and physiological data
- 🧠 DGIN (Dynamic Game Intelligence Network): real-time player behavior modeling
- 🧩 ATIS (Adaptive Tactical Intelligence Strategy): opponent-aware, reward-driven decision making
- 📊 Integrated support for benchmarking, visualization, and ablation analysis

---

## 🗂️ Project Structure

```
sports-speech-recognition/
├── models/                  # Core model components
│   ├── dgin/                # Encoder + policy logic
│   └── atis/                # Reward model + opponent strategy planner
├── training/                # Training logic & config
├── evaluation/              # Evaluation metrics & plotting
└── utils/                   # Data loading, preprocessing, augmentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/HengWei610/TacVoice.git
cd sports-speech-recognition
```

### 2. Install Dependencies

Requires Python 3.8+ and PyTorch ≥ 2.0.

```bash
pip install -r requirements.txt
```

---

## 🧪 Training & Evaluation

### Train DGIN (speech encoder + action policy)

```bash
python training/train_dgin.py
```

### Train ATIS (reward-based tactical planner)

```bash
python training/train_atis.py
```

### Evaluate ASR Performance

```bash
python evaluation/evaluate_asr.py
```

To visualize confusion matrix or training trends:

```python
from evaluation.visualize_results import plot_confusion_matrix, plot_metric_trends
```

---

## 🧠 Model Overview

### ✅ DGIN (Dynamic Game Intelligence Network)
- Encodes player state, ball dynamics, and game context
- Graph Neural Network + Temporal Convolution + Attention Fusion
- Outputs player-specific latent embeddings and policy actions

### ✅ ATIS (Adaptive Tactical Intelligence Strategy)
- Models team tactics and opponent behavior using reinforcement learning
- Optimizes strategy via game-theoretic reasoning and reward shaping
- Supports trajectory-aware planning and real-time recalibration

---

## 📦 Datasets

This framework uses both domain-specific and benchmark datasets:

### ⚽ SoccerNet  
Multi-modal dataset for soccer with events, video, audio, and player metadata.  
🔗 [https://www.soccer-net.org](https://www.soccer-net.org)

```bibtex
@inproceedings{giancola2018soccernet,
  title={SoccerNet: A Scalable Dataset for Action Spotting in Soccer Videos},
  author={Giancola, Silvio et al.},
  booktitle={CVPR}, year={2018}
}
```

---

### 🔊 LibriSpeech  
Read English audiobook speech corpus. Used for ASR pretraining and fine-tuning.  
🔗 [http://www.openslr.org/12](http://www.openslr.org/12)

```bibtex
@inproceedings{panayotov2015librispeech,
  title={Librispeech: An ASR corpus based on public domain audio books},
  author={Panayotov, V. et al.},
  booktitle={ICASSP}, year={2015}
}
```

---

### 🏠 CHiME-5  
Multi-speaker conversational speech in real home settings.  
🔗 [https://chimechallenge.github.io/chime5](https://chimechallenge.github.io/chime5)

```bibtex
@inproceedings{barker2018chime5,
  title={The CHiME-5 corpus: Multi-microphone conversational speech},
  author={Barker, J. et al.},
  booktitle={ICASSP}, year={2018}
}
```

---

### 🚉 Aurora-4  
Noise-augmented version of WSJ0 for robust ASR benchmarking.  
🔗 [https://catalog.ldc.upenn.edu/LDC2000S87](https://catalog.ldc.upenn.edu/LDC2000S87)

```bibtex
@inproceedings{parihar2002aurora,
  title={The Aurora experimental framework for evaluating speech recognition in noise},
  author={Parihar, N. and Picone, J.},
  booktitle={ISCA ITRW ASR}, year={2000}
}
```

---

## ⚙️ Configuration

All hyperparameters are defined in:

```yaml
training/config.yaml
```

You can modify:
- Learning rate
- Epochs
- Batch size
- Model dimensions
- Reward shaping parameters

---

## 📊 Metrics & Visualization

- **Accuracy**, **F1-score**, **AUC**: Evaluated via `evaluation/metrics.py`
- **Confusion Matrix**: `plot_confusion_matrix()`
- **Trend Curves**: `plot_metric_trends()`

---

## 📚 Citation

If you use this work in your research:

```bibtex
@article{wei2025sportsasr,
  title={Advanced Speech Recognition for Athlete Performance Analysis in Sports Science},
  author={Wei, Heng and Zhang, Yuze},
  journal={Preprint}, year={2025}
}
```

---

## 🚧 Future Development

We plan to extend this framework with the following enhancements:

- **Multilingual Speech Support**: Extend speech recognition models to support multiple languages and dialects.
- **Sport-Specific Adaptation Modules**: Introduce specialized modules for other sports (e.g., basketball, rugby, tennis).
- **Edge Deployment**: Optimize models using pruning and quantization for edge devices.
- **Real-Time Tactical Demos**: Link speech input to visual tactical visualizations.
- **Online Learning**: Enable live adaptation to athlete performance and new speech patterns.

Contributions and forks are welcome!

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 Legend Co.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions.

See the [LICENSE](./LICENSE) file for the full license text.

---

## 🙏 Acknowledgements

This work was supported by:

- **Shanxi University**, School of Economics and Management  
- **College of Physical Education**, Shanxi University  
- **Statistical Science Research Project of Shanxi Province**  
  *(Grant No. 2024D009 – Statistical Monitoring of New Quality Productivity and Its Influence on High-Quality Economic Development)*

Special thanks to open-source contributors and frameworks including:
- PyTorch, Torch-Geometric, Transformers, librosa
- SoccerNet and the broader speech recognition community

---
