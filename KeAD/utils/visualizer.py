"""Anomaly Visualization."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator

import os
import cv2
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries


class ImageGrid:
    """Helper class that compiles multiple images into a grid using subplots.

    Individual images can be added with the `add_image` method. When all images have been added, the `generate` method
    must be called to compile the image grid and obtain the final visualization.
    """

    def __init__(self) -> None:
        self.images: list[dict] = []
        self.figure: matplotlib.figure.Figure
        self.axis: np.ndarray

    def add_image(self, image: np.ndarray, title: str | None = None, color_map: str | None = None) -> None:
        """Add an image to the grid.

        Args:
          image (np.ndarray): Image which should be added to the figure.
          title (str): Image title shown on the plot.
          color_map (str | None): Name of matplotlib color map used to map scalar data to colours. Defaults to None.
        """
        image_data = dict(image=image, title=title, color_map=color_map)
        self.images.append(image_data)

    def generate(self) -> np.ndarray:
        """Generate the image.

        Returns:
            Image consisting of a grid of added images and their title.
        """
        num_cols = len(self.images)
        figure_size = (num_cols * 5, 5)
        self.figure, self.axis = plt.subplots(1, num_cols, figsize=figure_size)
        self.figure.subplots_adjust(right=0.9)

        axes = self.axis if isinstance(self.axis, np.ndarray) else np.array([self.axis])
        for axis, image_dict in zip(axes, self.images):
            axis.axes.xaxis.set_visible(False)
            axis.axes.yaxis.set_visible(False)
            axis.imshow(image_dict["image"], image_dict["color_map"], vmin=0, vmax=255)
            if image_dict["title"] is not None:
                axis.title.set_text(image_dict["title"])
        self.figure.canvas.draw()
        # convert canvas to numpy array to prepare for visualization with opencv
        img = np.frombuffer(self.figure.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(self.figure.canvas.get_width_height()[::-1] + (3,))
        plt.close(self.figure)
        return img


def normalize(pred, max_value=None, min_value=None):    # 归一化[0, 1]
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def apply_ad_scoremap(image, scoremap, alpha=0.5):   # 将score map映射到 image上
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def anomaly_map_to_color_map(anomaly_map: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Compute anomaly color heatmap.

    Args:
        anomaly_map (np.ndarray): Final anomaly map computed by the distance metric.
        normalize (bool, optional): Bool to normalize the anomaly map prior to applying
            the color map. Defaults to True.

    Returns:
        np.ndarray: [description]
    """
    if normalize:
        anomaly_map = (anomaly_map - anomaly_map.min()) / np.ptp(anomaly_map)
    anomaly_map = anomaly_map * 255
    anomaly_map = anomaly_map.astype(np.uint8)

    anomaly_map = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
    #anomaly_map = cv2.cvtColor(anomaly_map, cv2.COLOR_BGR2RGB)
    return anomaly_map


def superimpose_anomaly_map(
    anomaly_map: np.ndarray, image: np.ndarray, alpha: float = 0.4, gamma: int = 0, normalize: bool = False
) -> np.ndarray:
    """Superimpose anomaly map on top of in the input image.

    Args:
        anomaly_map (np.ndarray): Anomaly map
        image (np.ndarray): Input image
        alpha (float, optional): Weight to overlay anomaly map
            on the input image. Defaults to 0.4.
        gamma (int, optional): Value to add to the blended image
            to smooth the processing. Defaults to 0. Overall,
            the formula to compute the blended image is
            I' = (alpha*I1 + (1-alpha)*I2) + gamma
        normalize: whether or not the anomaly maps should
            be normalized to image min-max


    Returns:
        np.ndarray: Image with anomaly map superimposed on top of it.
    """

    anomaly_map = anomaly_map_to_color_map(anomaly_map.squeeze(), normalize=normalize)
    superimposed_map = cv2.addWeighted(anomaly_map, alpha, image, (1 - alpha), gamma)
    return superimposed_map


def vis(pathes, anomaly_map, score, img_size, save_path, cls_name, gt_masks=None, suffix=''):   # 可视化 anomaly 图
    #pred_masks = anomaly_map >= 0.5
    
    for idx, path in enumerate(pathes):
        visualization = ImageGrid()
        cls = path.split('/')[-2]
        filename = path.split('/')[-1]
  
        vis = cv2.cvtColor(cv2.resize(cv2.imread(path), (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
        mask_show = normalize(anomaly_map[idx]).cpu().numpy()
        mask = anomaly_map[idx].cpu().numpy()
        pred_mask = mask_show >= 0.5
        vis_map = superimpose_anomaly_map(mask_show, vis, normalize=False)  # apply_ad_scoremap(vis, mask_show)

        visualization.add_image(vis, "Image")
        if gt_masks is not None:
            visualization.add_image(image=gt_masks[idx].cpu().numpy()*255, color_map="gray", title="Ground Truth")
        visualization.add_image(vis_map, "Predicted Heat Map")
        visualization.add_image(image=pred_mask*255, color_map="gray", title="Predicted Mask")
        image = visualization.generate()

        save_vis = os.path.join(save_path, cls_name[idx])  # cls
        if not os.path.exists(save_vis):
            os.makedirs(save_vis)
        cv2.imwrite(os.path.join(save_vis, filename[:-4]+'.png'), image)
 