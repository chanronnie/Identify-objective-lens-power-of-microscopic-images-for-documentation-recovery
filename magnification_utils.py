'''
File: magnification_utils.py
--------------------------------------
Student:Ronnie Chan (27206003)


This file contains the classes MedicalImagesDataset, MedicalImages and Viz.
'''

# ===========================
# IMPORT LIBRARIES
# ===========================
# General Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import pandas as pd
import polars as pl
import io
import gc
import os
import json
import time
from pathlib import Path

# For training CNN models
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models      # for Resnet-18
from torch.utils.data import Dataset, DataLoader


# ===========================
# class: MedicalImagesDataset
# ===========================

class MedicalImagesDataset(Dataset):
  """
  This class creates a custom PyTorch Dataset class for the BreaKHis histopathology dataset.
  
  ACKNOWLEDGMENT
  Code adapted from https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html with changes.
  Modifications were made to support the BreaKHis dataset.
  """


  def __init__(self, df, transform=None):
    """
    Initializes the dataset instance with specific data and transformations.

    Arguments:
    ----------
    df: polars.DataFrame
      The preprocessed BreaKHis dataset

    transform : callable, optional
        PyTorch transforms for data augmentation/scaling.
    """

    self.df = df
    self.transform = transform


  def __len__(self):
    """
    Returns the number of records in the dataset.

    Returns:
    --------
    int
      The number of records in the dataset.
    """

    return len(self.df)


  def __getitem__(self, idx):
    """
    Retrieves and preprocesses a single image-label pair.

    Arguments:
    ----------
    idx: int
      The index of the sample to be retrieved.

    Returns:
    --------
    image: torch.Tensor or PIL.Image
      The processed image.
    label: int
      The numerical class label corresponding to the magnification level.
    """

    # Extract the X, y
    image_bytes  = self.df[idx, 'bytes']
    label = self.df[idx, "label"]

    # Convert bytes to RGB
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Transform the image
    if self.transform:
      image = self.transform(image)

    return image, label
  
# end of MedicalImagesDataset
