# Acoustic-Guitar-Source-Separation
Acoustic Guitar Source Separation fine-tuning Open-Unmix model

## Project Overview
This project fine-tunes the **Open-Unmix** model for **Music Source Separation (MSS)**, specifically to isolate **acoustic guitar** tracks from mixed audio. The goal is to enhance flexibility in music production by enabling the extraction of individual instrument stems.

---

## Key Features
- **Model**: [Open-Unmix](https://github.com/sigsep/open-unmix), a deep neural network based on a **bidirectional LSTM**, originally designed for vocals, drums, bass, and "other" stems.
- **Fine-Tuning**: Adapted the model to recognize **acoustic guitar** using the [BabySlakh](https://zenodo.org/record/4599666) dataset (synthetic audio derived from MIDI).
- **Dataset**: Subset of [Slakh2100](https://slakh.com), preprocessed and augmented for training.

---

## Methodology
### 1. Preprocessing
- Filtered tracks to retain **acoustic guitar + bass** stems.
- Segmented audio into **20-second clips** (10-second overlap).
- Applied augmentation:  
  - Gaussian noise  
  - Pitch shifting  
  - Time stretching  
- Split data: **Train (80%)**, Validation (10%), Test (10%).

### 2. Training
- **Hyperparameters**:
  - Learning Rate: `0.001` (adaptive decay).
  - Batch Size: `32`.
  - Epochs: `60`.
  - Hidden Size: `512` units/layer.
- **Best Performance**: Achieved at epoch 57 (`loss = 0.65`).

### 3. Evaluation Metrics
| Metric       | Acoustic Guitar (Avg) | Bass (Avg) |
|-------------|----------------------|------------|
| **SDR**     | 1.38                 | 1.42       |
| **SAR**     | -6.06                | -5.14      |

*(Higher = better for SDR; SAR measures artifacts)*  

---

## Results
- Successfully adapted Open-Unmix for **acoustic guitar separation**.
- Performance comparable to baseline (bass stems), but **artifacts remain a challenge** (low SAR).

---

## Future Work
1. **Expand Dataset**: Use full [Slakh2100](https://slakh.com) (~100 GB) or real recordings.
2. **Advanced Augmentation**: Simulate room acoustics/microphone effects.
3. **Real-Time Processing**: Explore lightweight/causal architectures.
4. **Extend to Other Instruments**: Strings, percussion, etc.

---

## References
- Open-Unmix Paper: [Stöter et al., 2019](https://joss.theoj.org/papers/10.21105/joss.01667)
- MUSDB18 Dataset: [sigsep.github.io](https://sigsep.github.io/datasets/musdb.html)
- BabySlakh/Slakh2100: [IEEE WASPAA 2019](https://ieeexplore.ieee.org/document/8937164)

---

## How to Use
For inference with the fine-tuned model:  
```python
from openunmix import predict  
predict.separate(mix="your_audio.wav", targets=["acoustic_guitar"])
