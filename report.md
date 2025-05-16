# Experience Report

**Project:** SSD300 with VGG16 Backbone on Pascal VOC 2007  
**Author:**   Aryan Singh
**Date:** May 16, 2025  

---

## 1. Introduction

In this assignment, I implemented a Single Shot Detector (SSD300) with a VGG16 backbone to perform object detection on the Pascal VOC 2007 dataset. The goal was to train the model end‑to‑end, evaluate its performance in terms of mean Average Precision (mAP@0.5), and produce a demo showing detections on sample images.

---

## 2. Objectives

1. **Implement** SSD300 + VGG16 for object detection.  
2. **Train** on VOC2007 train split and **validate** on VOC2007 val split.  
3. **Track** training loss, validation loss, and mAP@0.5 over epochs.  
4. **Save** model checkpoints and final weights.  
5. **Demonstrate** inference on test images.  
6. **Reflect** on the process in this report.

---

## 3. Methodology

- **Backbone:** Pre‑trained VGG16 features (with classification head removed).  
- **Detection Head:** SSD300’s multi‑scale convolutional predictors.  
- **Data:** Pascal VOC 2007, resized to 300×300.  
- **Training:**  
  - Optimizer: SGD, lr = 1e‑4, momentum = 0.9, weight_decay = 5e‑4  
  - Batch size: 8  
  - Epochs: 10  
  - Checkpoint saved after each epoch.  
- **Validation:**  
  - Computed loss in “training” mode to ensure we get a loss dict.  
  - Computed mAP@0.5 using `torchmetrics.mean_ap`.  
- **Demo:** Ran inference on a small folder of test images, drew boxes, and saved results.

---

## 4. Results

| Epoch | Train Loss | Val Loss  | mAP@0.5 |
|------:|-----------:|----------:|--------:|
|   0   |   19.0223  |   18.2923 |  0.0000 |
|   1   |   18.2769  |   18.0501 |  0.0000 |
|   2   |   18.1441  |   17.8096 |  0.0000 |
| …     |     …      |     …     |    …    |
|   9   |   10.3174  |   10.0466 |  0.0000 |

---

### 4.1 Is mAP@0.5 = 0.0000 Okay?

Seeing `mAP@0.5 = 0.0000` across all epochs indicates that, under the strict IoU threshold of 0.5, **none** of the model’s predicted boxes matched ground‑truth boxes with high enough overlap _and_ correct labels. Possible reasons:

- **Insufficient training**: 10 epochs may not be enough for SSD300 to converge on VOC data at this learning rate.  
- **Data preprocessing mismatch**: Boxes might not be scaled correctly when computing mAP.  
- **Evaluation bug**: If the model’s output boxes or labels are misformatted, `torchmetrics` may discard them.  

**Next steps** would be to:
1. Visualize a few predictions to ensure boxes make sense.  
2. Increase training epochs (e.g. 50+).  
3. Add a learning‑rate schedule (e.g. step‑decay).  
4. Confirm evaluation pipeline: ensure targets and preds align.

---

## 5. Challenges & Surprises

- **Training vs. Inference Modes**: By default, `.eval()` forces SSD into inference, returning detections instead of losses. I had to toggle back to `.train()` within `evaluate()` to get a proper loss dict.  
- **NaN Loss Handling**: Early in training, occasional NaNs appeared—skipped those batches.  
- **Zero mAP**: Surprising to see zero detections at IoU≥0.5 initially; reinforced the need for visual debugging and longer training.

---

## 6. AI Assistance

I leveraged ChatGPT (OpenAI) extensively to:  
- Scaffold the training script and checkpoint logic.  
- Debug the `.eval()` vs. `.train()` quirk during validation.  
- Design the `compute_map` helper using `torchmetrics`.  

**Example prompt**:  
> “In PyTorch’s SSD implementation, how do I force the model to return loss at validation time instead of detections?”

---

## 7. Learnings

- The interplay between model modes (`train()` vs. `eval()`) can dramatically change outputs.  
- Tracking **both** loss and detection metrics is critical for real progress.  
- Checkpointing and modular code (`utils.py`) keep experiments reproducible.  

---

## 8. Suggestions for Improvement

1. **Longer Training**: Recommend at least 30–50 epochs or a scheduler.  
2. **Data Augmentation**: Random flips, color jitter could boost early learning.  
3. **Faster Evaluation**: Cache preprocessed validation tensors to speed up mAP computation.  
4. **Sample Visualizations**: Automatically save a few annotated val images per epoch to monitor qualitative progress.  

---

**Conclusion:**  
This assignment provided valuable hands‑on experience with object detection pipelines in PyTorch and highlighted the importance of careful validation and metric tracking. While initial mAP was zero, the framework is now in place to push performance higher with extended training and tuning.
