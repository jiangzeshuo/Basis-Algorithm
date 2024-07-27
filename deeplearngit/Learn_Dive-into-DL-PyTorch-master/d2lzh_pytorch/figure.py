# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:50:33 2020

The image module contains functions for plotting

@author: as
"""
from matplotlib import pyplot as plt


def set_figsize(figsize=(3.5, 2.5)):
    """Set matplotlib figure size."""
    fig = plt.figure()
    plt.rcParams['figure.figsize'] = figsize


def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format.
    Convert the bounding box (top-left x, top-left y, bottom-right x, bottom-right y) 
    format to matplotlib format: ((upper-left x, upper-left y), width, height)
    """
    return plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1], 
                        fill=False, edgecolor=color, linewidth=2)


def show_bboxes(axes, bboxes, labels=None, colors=None):
    """Show bounding boxes.
    bboxes: 待绘制的bbox， need be format as [[x1,y1,x2,y2],[...], ..., [...]]
    labels: 与要绘制的bbox一一对应的标注信息，将会绘制在bbox的左上角
    colors: 标注框显示的颜色，不设置会自动使用几个默认颜色进行轮换
    """
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox, color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=6, color=text_color,
                      bbox=dict(facecolor=color, lw=0))


def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes


