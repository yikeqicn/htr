import cv2
import torch.utils.data as data
import numpy as np
from glob import glob
import os
from os.path import join, basename, dirname

class EyDigitStrings(data.Dataset):

  def __init__(self, root):

    self.root = root
    allFiles = glob(join(root, '**/*.jpg'))
    gtText = [basename(f)[:-4] for f in allFiles]
    self.samples = list(zip(allFiles, gtText))

  def __len__(self):
    return len(self.samples)


  def __getitem__(self, idx):

    imgPath = self.samples[idx][0]
    gtText = self.samples[idx][1]
    img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    return img, gtText


root = '/Users/dl367ny/datasets/htr_assets/crowdsource/processed'
dataset = EyDigitStrings(root)
dataloader = data.DataLoader(dataset, batch_size=5, shuffle=True, num_workers=10)
