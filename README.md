# Identify objective lens power of microscopic images for documentation recovery

**Course:** COMP 432 Machine Learning - Winter 2026

**Student Name:** Ronnie Chan

<br>

## Repository Overview

This repository provides the utility script and the custom PyTorch classes for the BreaKHis dataset.
The submitted Colab notebook fetches this `magnification_utils.py` script **automatically** using the command below:

```
!wget https://raw.githubusercontent.com/chanronnie/Identify-objective-lens-power-of-microscopic-images-for-documentation-recovery/main/magnification_utils.py
```

<br>

## Project Assets

Due to their large size, additional project assets (such as the raw BreaKHis dataset from Hugging Face and saved model weights as `.pth`) are shared via my **Google Drive**. The submitted Colab notebook will load these files **automatically** during execution.

| File Name              | Category       | Purpose                                                   |
| :--------------------- | :------------- | :-------------------------------------------------------- |
| **conv2d_study.db**    | Optuna Study   | Complete 15-trial hyperparameter search history           |
| **conv2d_log.json**    | Training Logs  | Train/Valid loss and accuracy for the custom CNN          |
| **resnet18_log.json**  | Training Logs  | Train/Valid loss and accuracy for the ResNet-18           |
| **conv2d_model.pth**   | Custom Weights | Final trained state of the custom CNN                     |
| **resnet18_model.pth** | Resnet Weights | Final trained state of the ResNet-18 model for comparison |
