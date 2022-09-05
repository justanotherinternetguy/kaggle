import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import timm
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from pathlib import Path

from fastai.vision import *
from fastai.metrics import error_rate

from PIL import Image

inputs = Path("./data")
print(os.listdir(inputs))