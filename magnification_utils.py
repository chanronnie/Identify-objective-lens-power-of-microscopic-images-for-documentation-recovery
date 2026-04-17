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

# For training CNN models
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader


# ===========================
# class: MedicalImagesDataset
# ===========================

class MedicalImagesDataset(Dataset):
  """
  This class creates a custom PyTorch Dataset class for the BreaKHis histopathology dataset.

  Example
  -------
  >>> transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
  >>> dataset = MedicalImagesDataset(df=my_df, transform=transform)
  >>> img, label = dataset[0]
  >>> print(img.shape)
  torch.Size([3, 224, 224])
  
  ACKNOWLEDGMENT
  --------------
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
  """
  Utility class for preprocessing BreaKHis dataset
  
  Example
  -------
  >>> in_path = 'data_name'
  >>> mi = MedicalImages()
  >>> df = mi.load_data(in_path)
  >>> print(df.head())
  """

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
    raw_df = pl.from_arrow(raw_data["train"].data.table)

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



# ===========================
# class: Viz
# ===========================

class Viz:
  """
  Utility class for generating visualizations.
  
  Example:
  -------
  >>> import matplotlib.pyplot as plt
  >>> fig, ax = plt.subplots(1, 2, figsize=(10, 3))
  >>> Viz.plot_class_balance(train_df, "Training Set", ax=ax[0])
  >>> Viz.plot_class_balance(val_df, "Validation Set", ax=ax[1])
  >>> plt.show()
  """

  @staticmethod
  def plot_class_balance(df, title, ax=None):
    """
    Plots the class distribution for the given set.

    Arguments:
    ----------
    df: polars.DataFrame
      The dataset to be plotted.
    title: String
      The title of the plot.

    Returns:
    -------
    None
    """

    if ax is None:
      ax = plt.gca()
    label_map = {0: '100x', 1: '200x', 2: '400x', 3: '40x'}
    magnification_classes = df.group_by("label").len(name="count").sort("count", descending=True).to_pandas()
    magnification_classes['label'] = magnification_classes['label'].map(label_map)

    sns.barplot(data=magnification_classes, x="label", y="count", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Objective Lense")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)


  @staticmethod
  def get_trial_records(db, trial):
    """
    Retrieves the attributes from the Optuna study database for a given trial.

    Arguments:
    ----------
    db: pandas.DataFrame
      The Optuna study database
    trial: int
      Trial to retrieve

    Returns:
    -------- 
    train_losses: list
      The training loss history for the given trial
    valid_losses: list
      The validation loss history for the given trial 
    lr: float
      The learning rate for the given trial
    layers: int
      The number of layers for the given trial
    accuracy: float
      The validation accuracy for the given trial
    """
    trial_record = db[db['number'] == trial]
    train_losses = trial_record['user_attrs_train_loss_history'].iloc[0]
    valid_losses = trial_record['user_attrs_valid_loss_history'].iloc[0]
    lr = trial_record['params_lr'].iloc[0]
    layers = trial_record['params_n_layers'].iloc[0]
    accuracy = trial_record['user_attrs_valid_accuracy'].iloc[0]

    return train_losses, valid_losses, lr, layers, accuracy


  @staticmethod
  def plot_learning_curves(train_losses, valid_losses, title, ax=None, show_epochs=True):
    """
    Plots the training and validation losses (learning curves) with a given title.

    Arguments
    ---------
    train_losses: list
      The training loss history
    valid_losses: list
      The validation loss history
    title: str
      The title of the plot
    ax: matplotlib.axes.Axes
      The axis to plot on
    
    Returns
    ------- 
    None
    """

    # Plot the training and validation losses
    epochs = np.arange(len(train_losses))
    ax.plot(epochs, train_losses, '-o', markersize=2, label='Train loss')
    ax.plot(epochs, valid_losses, '-o', markersize=2, label='Validation loss')

    if show_epochs == True:
      ax.set_xticks(epochs)

    # Customize labels and title
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Losses')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
  

  @staticmethod
  def plot_confusion_matrix(conf_matrix, color, title, ax):
    """
    Plots a confusion matrix.

    Arguments:
    ----------
    conf_matrix: numpy.ndarray
      The confusion matrix to be plotted.
    color: str
      The color of the confusion matrix.
    title: str
      The title of the confusion matrix.
    
    Returns:
    --------
    None
    """

    class_names = ['100x', '200x', '400x', '40x']
    sns.heatmap(conf_matrix, 
                annot=True, 
                fmt='d', 
                cmap=color, ax=ax,
                xticklabels=class_names, 
                yticklabels=class_names)
    ax.set_title(title)
    ax.set_ylabel("Actual Magnification")
    ax.set_xlabel("Predicted Magnification")
  
  
  @staticmethod
  def plot_predictions(images, true_labels, pred_labels_conv2d, pred_labels_resnet18):
    """
    Plots 10 images with their true and predicted labels for both Conv2d and ResNet-18 on a grid (2x5)

    Arguments:
    ----------
    images: torch.Tensor
      A batch of images.
    true_labels: torch.Tensor
      The true labels for the images.
    pred_labels_conv2d: torch.Tensor
      The predicted labels for the images using the Conv2d model.
    pred_labels_resnet18: torch.Tensor
      The predicted labels for the images using the ResNet-18 model.
    
    Returns:
    --------
    None
    """

    # Define the grid
    fig, ax = plt.subplots(2, 5, figsize=(12, 5))
    ax_flatten = ax.flatten() 

    # Set title and the class names
    fig.suptitle(f"Magnification Classification: Conv2d vs ResNet-18", 
                  fontsize=16, fontweight='bold')  
    class_names = ["100x", "200x", "400x", "40x"]

    for i in range(len(images)):

        # Get the true and predicted labels
        true_label = class_names[true_labels[i]]
        pred_label_conv2d = class_names[pred_labels_conv2d[i]]
        pred_label_resnet18 = class_names[pred_labels_resnet18[i]]
              
        # Prepare image for plotting
        pixels = images[i].cpu().numpy()                        # Convert to Numpy array for plotting
        img = pixels.transpose((1, 2, 0))                       # Convert (C, H, W) Tensor to (H, W, C) for Matplotlib
        norm_img = (img - img.min()) / (img.max() - img.min())  # Normalize image to [0, 1] for correct image display
        
        # Plot image in the axis with its true and predicted labels
        ax_flatten[i].imshow(norm_img)
        ax_flatten[i].set_title(f"True Label: {true_label}\n"
                                f"Predicted (Conv2D): {pred_label_conv2d}\n"
                                f"Predicted (Resnet18): {pred_label_resnet18}", 
                                fontsize=10)
        ax_flatten[i].axis('off')
              
    plt.tight_layout()
    plt.show()

# end of Viz class
