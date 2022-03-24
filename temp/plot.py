# Result Visualization Helper
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu
import torchvision
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

M, N = 4, 6
RESULT_IMG_PATH = '/tmp/test_result.jpg'
CIFAR10_LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']

def plot_results(images, labels, preds):
  images, labels, preds = images[:M*N], labels[:M*N], preds[:M*N]
  inv_norm = transforms.Normalize(
      mean=(-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010),
      std=(1/0.2023, 1/0.1994, 1/0.2010))

  num_images = images.shape[0]
  fig, axes = plt.subplots(M, N, figsize=(16, 9))
  fig.suptitle('Correct / Predicted Labels (Red text for incorrect ones)')

  for i, ax in enumerate(fig.axes):
    ax.axis('off')
    if i >= num_images:
      continue
    img, label, prediction = images[i], labels[i], preds[i]
    img = inv_norm(img)
    img = img.permute(1, 2, 0) # (C, M, N) -> (M, N, C)
    label, prediction = label.item(), prediction.item()
    if label == prediction:
      ax.set_title(u'\u2713', color='blue', fontsize=22)
    else:
      ax.set_title(
          'X {}/{}'.format(CIFAR10_LABELS[label],
                          CIFAR10_LABELS[prediction]), color='red')
    ax.imshow(img)
  plt.savefig(RESULT_IMG_PATH, transparent=True)