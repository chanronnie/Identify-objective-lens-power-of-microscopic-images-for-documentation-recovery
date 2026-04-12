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


# ===========================
# class: MedicalImagesDataset
# ===========================

class MedicalImages():
  """This class handles BreaKHis dataset preprocessing"""

  def __init__(self):
    """Initializes the dataset instance with specific data."""

    self.df = None   # Initiate Polars DataFrame
    self.train_set = None
    self.valid_set = None
    self.test_set = None


  def load_data(self, data_path):
    """
    Loads BreaKHis data from given path and converts to polars.DataFrame.

    Arguments:
    ----------
    data_path: String
      The path where the data is stored.
    """

    # 1. Load datat from disk
    raw_data = load_from_disk(data_path)

    # 2. Convert to a Polars DataFrame via the Arrow Table
    raw_df = pl.from_arrow(raw_data.data.table)

    # 3. Unnest the polar DataFrame
    self.df = (
        raw_df.unnest("image")                                  # Unnest the original Polar DataFrame
              .select(["bytes", "label"])                       # select only columns 'bytes' (x) and 'label' (y)
              .with_columns(pl.col("label").cast(pl.Int8))      # downcast dtype int64 to int8
              .with_row_index("index")                          # add index columns
    )
    print("Polars DataFrame Sucessfully Loaded.")


  def get_dataframe(self):
    """Returns the BreaKHis data as polars.DataFrame

    Returns:
    -------
    polars.DataFrame
      The BreaKHis dataset as a Polars DataFrame.
    """

    return self.df


  def split_data(self, valid_size=0.15, test_size=0.15):
    """"
    Splits BreaKHis data into training, validation and test sets.

    Arguments:
    ----------
    valid_size: float
      The proportion of the dataset to include in the validation set.
    test_size: float
      The proportion of the dataset to include in the test set.

    Returns:
    --------
    None
    """

    # 1. Get all indices
    indices = np.arange(self.df.height)

    # 2. Split testing set
    train_valid_idx, test_idx = train_test_split(indices,
                                              test_size=test_size,
                                              random_state=42,
                                              stratify=self.df['label'])    # to prevent class imbalance

    # 3. Split training/validation
    valid_ratio = valid_size / (1.0 - test_size)
    train_valid_labels = self.df[train_valid_idx, 'label']
    train_idx, valid_idx = train_test_split(train_valid_idx,
                                          test_size=valid_ratio,
                                          random_state=42,
                                          stratify=train_valid_labels)   # to prevent class imbalance - consistent split accross all categories

    # 4. Split the DataFrame into sets
    self.train_set = self.df[train_idx]
    self.valid_set = self.df[valid_idx]
    self.test_set =  self.df[test_idx]


  def get_split_data(self):
    """"
    Retrieves and returns the training, validation and test sets.

    Returns:
    --------
    train_set: polars.DataFrame
      The training
    valid_set: polars.DataFrame
      The validation set
    test_set: polars.DataFrame
      The test set
    """

    return self.train_set, self.valid_set, self.test_set


  def build_loader(self, split_data, batch_size, transformer, shuffle=False):
    """
    Builds a PyTorch DataLoader for the given dataset by using MedicalImagesDataset.

    Arguments:
    ---------
    split_data: polars.DataFrame
      The dataset to be loaded.
    batch_size: int
      The number of samples per batch to load.
    transformer: torchvision.transforms.Compose
      PyTorch transforms for data augmentation/scaling.

    Returns:
    --------
    torch.utils.data.DataLoader:
      The DataLoader for the given dataset.
    """
    return DataLoader(
          MedicalImagesDataset(split_data, transformer),
          batch_size=batch_size,
          pin_memory=True,
          prefetch_factor=2,
          num_workers = 2,
          shuffle=shuffle)


  def get_dataloaders(self, batch_size):
    """
    Build data loaders for the training, validation and test sets.

    Arguments:
    ----------
    batch_size: int
      The number of samples per batch to load.

    Returns:
    --------
    train_loader: torch.utils.data.DataLoader
      The DataLoader for the training
    valid_loader: torch.utils.data.DataLoader
      The DataLoader for the validation set
    test_loader: torch.utils.data.DataLoader
      The DataLoader for the test set
    """

    # Get the transformers
    train_transformer, standard_transformer = self.__get_transformers__()

    # Build and return the data loaders
    train_loader = self.build_loader(self.train_set, batch_size, train_transformer, shuffle=True)
    valid_loader = self.build_loader(self.valid_set, batch_size, standard_transformer, shuffle=False)
    test_loader = self.build_loader(self.test_set, batch_size, standard_transformer, shuffle=False)
    return train_loader, valid_loader, test_loader


  def __get_transformers__(self):
    """
    Defines the preprocessing transformation.

    Returns:
    --------
    train_transformer: torchvision.transforms.Compose
      The preprocessing transformation for train set
    standard_transformer: torchvision.transforms.Compose
      The preprocessing transformation for validation and test sets
    """

    # Defines the default values for mean and std
    default_mean = [0.485, 0.456, 0.406]
    default_std = [0.229, 0.224, 0.225]

    # Define the preprocessing transformation for train set
    train_transformer = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=default_mean, std=default_std)])

    # Define the preprocessing transformation for validation and test sets
    standard_transformer = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=default_mean, std=default_std)])

    return train_transformer, standard_transformer
# end of MedicalImages
