import os
import cv2  # type: ignore
import torch  # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from PIL import Image # type: ignore


def calculate_histogram(image, num_channels):
    """
    Calculate the histogram of an image.
    
    Parameters:
        image (numpy.ndarray): The image array.
        num_channels (int): Number of image channels (1=Grayscale, 3=RGB).
    
    Returns:
        numpy.ndarray: Computed histogram (2D array for each channel).
    """
    histograms = []
    for i in range(num_channels):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256]).flatten()
        histograms.append(hist)
    return np.array(histograms)

def plot_histogram(histograms, num_channels):
    """
    Visualize the histogram.
    
    Parameters:
        histograms (numpy.ndarray): Histogram data (2D array for each channel).
        num_channels (int): Number of channels (1=Grayscale, 3=RGB).
    
    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    if num_channels == 1:
        plt.plot(histograms[0], color='gray', label="Grayscale Channel")
        plt.title("Average Histogram for Grayscale Images")
    else:
        colors = ('r', 'g', 'b')
        for i, color in enumerate(colors):
            plt.plot(histograms[i], color=color, label=f"Average {color.upper()} Channel")
        plt.title("Average Histogram for RGB Channels")
    
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_average_histogram(image_paths):
    """
    Compute the average histogram across multiple images.
    
    Parameters:
        image_paths (list of str): List of image file paths.
    
    Returns:
        numpy.ndarray, int: The average histogram data and number of channels.
    """
    num_images = len(image_paths)
    channel_histograms = None
    num_channels = 0
    
    for image_path in image_paths:
        # Read the image
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue
        
        # Determine number of channels
        if len(image.shape) == 2:  # Grayscale
            num_channels = 1
            image_channels = image
        elif len(image.shape) == 3 and image.shape[2] == 3:  # RGB
            num_channels = 3
            image_channels = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            print(f"Unsupported image format: {image_path}")
            continue
        
        # Compute histogram
        histograms = calculate_histogram(image_channels, num_channels)
        
        # Accumulate histogram
        if channel_histograms is None:
            channel_histograms = histograms
        else:
            channel_histograms += histograms
    
    if channel_histograms is not None:
        # Calculate average histogram
        channel_histograms /= num_images
    return channel_histograms, num_channels

def plot_average_histogram(image_paths):
    """
    Compute and plot the average histogram for multiple images.
    
    Parameters:
        image_paths (list of str): List of image file paths.
    
    Returns:
        None
    """
    histograms, num_channels = calculate_average_histogram(image_paths)
    if histograms is not None:
        plot_histogram(histograms, num_channels)
    else:
        print("No valid images found.")

