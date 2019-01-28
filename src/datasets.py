import random
import numpy as np
import cv2
from SamplePreprocessor import preprocess
from glob import glob

import gzip
import pickle
import torch.utils.data as data
import os
from utils import maybe_download
from os.path import join, basename, dirname, exists
home = os.environ['HOME']

class IAM(data.Dataset):
  '''iam dataset'''

  def __init__(self, root='/root/datasets', transform=None):

    self.transform=transform
    self.root = join(root,'iam_handwriting')

    # download and put dataset in correct directory
    maybe_download('https://www.dropbox.com/sh/tdd0784neuv9ysh/AABm3gxtjQIZ2R9WZ-XR9Kpra?dl=0',
                   'iam_handwriting', root, 'folder')
    if exists(join(self.root,'words.tgz')):
      os.makedirs(join(self.root, 'words'))
      os.system('tar xvzf '+join(self.root, 'words.tgz')+' --directory '+join(self.root, 'words'))
      os.system('rm '+join(self.root,'words.tgz'))

    # begin collecting all words in IAM dataset frm the words.txt summary file at the root of IAM directiory
    labelsFile = open(join(self.root,'words.txt'))
    chars = set()
    self.samples = []
    for line in labelsFile:

      # ignore comment line
      if not line or line[0] == '#':
        continue

      lineSplit = line.strip().split(' ')
      assert len(lineSplit) >= 9

      # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
      fileNameSplit = lineSplit[0].split('-')
      fileName = join(self.root, 'words/') + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' + \
                 lineSplit[0] + '.png'

      # GT text are columns starting at 9
      label = ' '.join(lineSplit[8:])
      self.chars = chars.union(set(list(label)))

      # put sample into list
      self.samples.append( (fileName, label) )

  def __str__(self):
    return 'IAM words dataset. Data location: '+self.root+', Length: '+str(len(self))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):

    label = self.samples[idx][1]
    # img = preprocess(cv2.imread(self.samples[i][0], cv2.IMREAD_GRAYSCALE),
    #                  args.imgsize, self.args, False, is_testing)
    img = cv2.imread(self.samples[idx][0], cv2.IMREAD_GRAYSCALE)
    if self.transform:
      img = self.transform(img)

    return img, label

class EyDigitStrings(data.Dataset):

  def __init__(self, root='/root/datasets', transform=None):

    self.transform = transform
    self.root = join(root, 'htr_assets/crowdsource/processed')

    # custom dataset loader
    allfiles = glob(join(self.root, '**/*.jpg'), recursive=True)
    labels = [basename(f)[:-4] if basename(f).find('empty-')==-1 else '_' for f in allfiles] # if filename has 'empty-', then the ground truth is nothing
    chars = set.union(*[set(l) for l in labels])
    self.samples = list(zip(allfiles,labels))

  def __str__(self):
    return 'EY Digit Strings dataset. Data location: '+self.root+', Length: '+str(len(self))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):

    label = self.samples[idx][1]
    img = cv2.imread(self.samples[idx][0], cv2.IMREAD_GRAYSCALE)
    if self.transform:
      img = self.transform(img)

    return img, label


iam = IAM(join(home,'datasets'))
ey = EyDigitStrings(join(home,'datasets'))
