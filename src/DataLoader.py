import random
import numpy as np
import cv2
from SamplePreprocessor import preprocess
from glob import glob
from os.path import join
import os
from os.path import join, basename, dirname

import gzip
import pickle
import torch.utils.data as data
import os
from utils import maybe_download
from os.path import join, basename, dirname
home = os.environ['HOME']

class IAM(data.Dataset):
  '''iam dataset'''

  def __init__(self, root, transform=None):

    maybe_download('https://www.dropbox.com/sh/tdd0784neuv9ysh/AABm3gxtjQIZ2R9WZ-XR9Kpra?dl=0',
                   'iam_handwriting2', join(root, 'iam_handwriting2'), 'tar')

    self.root = join(root, 'MNISTpoly'+str(size), 'train')
    self.transform=transform

    # custom dataset loader
    fileName = []
    for f in filePath: fileName = fileName + glob(join(f, '**/*.jpg'), recursive=True)
    gtText = [basename(f)[:-4] if basename(f).find('empty-')==-1 else '_' for f in fileName] # if filename has 'empty-', then the gt is '_'
    chars = set.union(*[set(t) for t in gtText])
    self.samples = [Sample(g,f) for g,f in zip(gtText, fileName)]

    # begin collecting all words in IAM dataset frm the words.txt summary file at the root of IAM directiory
    f = open(filePath + 'words.txt')
    chars = set()
    for line in f:

      # ignore comment line
      if not line or line[0] == '#':
        continue

      lineSplit = line.strip().split(' ')
      assert len(lineSplit) >= 9

      # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
      fileNameSplit = lineSplit[0].split('-')
      fileName = filePath + 'words/' + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' + \
                 lineSplit[0] + '.png'

      # GT text are columns starting at 9
      gtText = ' '.join(lineSplit[8:])[:maxTextLen]
      chars = chars.union(set(list(gtText)))

      # put sample into list
      self.samples.append( (fileName, gtText) )

  def __str__(self):
    return 'IAM words dataset. Data location: '+self.root

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):

    gtText = self.samples[i].gtText
    # img = preprocess(cv2.imread(self.samples[i][0], cv2.IMREAD_GRAYSCALE),
    #                  args.imgsize, self.args, False, is_testing)
    img = cv2.imread(self.samples[i][0])

    return img, gtText

class EyDigitStrings(data.Dataset):

  def __init__(self, root):

    self.root = join(root, 'MNISTpoly'+str(size), 'train')

    # custom dataset loader
    fileName = []
    for f in filePath: fileName = fileName + glob(join(f, '**/*.jpg'), recursive=True)
    gtText = [basename(f)[:-4] if basename(f).find('empty-')==-1 else '_' for f in fileName] # if filename has 'empty-', then the ground truth is nothing
    self.chars = set.union(*[set(t) for t in gtText])
    self.samples = [(f,g) for f,g in zip(fileName, gtText)]

  def __str__(self):
    return 'EY Digit Strings dataset. Data location: '+self.root



class Sample:
  "sample from the dataset"
  def __init__(self, gtText, filePath):
    self.gtText = gtText
    self.filePath = filePath

class Batch:
  "batch containing images and ground truth texts"
  def __init__(self, gtTexts, imgs):
    self.imgs = np.stack(imgs, axis=0)
    self.gtTexts = gtTexts

class DataLoader:
  "loads data which corresponds to IAM format, see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database"

  def __init__(self, filePath, batchSize, imgSize, maxTextLen, args, is_test=False):
    "loader for dataset at given location, preprocess images and text according to parameters"

    # assert filePath[-1] == '/'

    self.args = args
    self.dataAugmentation = False
    self.currIdx = 0
    self.batchSize = batchSize
    self.imgSize = imgSize
    self.samples = []

    if args.iam:

      f = open(filePath + 'words.txt')
      chars = set()
      for line in f:
        # ignore comment line
        if not line or line[0] == '#':
          continue

        lineSplit = line.strip().split(' ')
        assert len(lineSplit) >= 9

        # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
        fileNameSplit = lineSplit[0].split('-')
        fileName = filePath + 'words/' + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' + \
                   lineSplit[0] + '.png'

        # GT text are columns starting at 9
        gtText = ' '.join(lineSplit[8:])[:maxTextLen]
        chars = chars.union(set(list(gtText)))

        # put sample into list
        self.samples.append(Sample(gtText, fileName))

    else:
      # custom dataset loader
      fileName = []
      for f in filePath: fileName = fileName + glob(join(f, '**/*.jpg'), recursive=True)
      gtText = [basename(f)[:-4] if basename(f).find('empty-')==-1 else '_' for f in fileName] # if filename has 'empty-', then the gt is '_'
      chars = set.union(*[set(t) for t in gtText])
      self.samples = [Sample(g,f) for g,f in zip(gtText, fileName)]

    # split into training and validation set: 95% - 5%
    if not is_test: # 95% split for train and valid
      random.shuffle(self.samples)
      splitIdx = int(0.95 * len(self.samples))
      self.trainSamples = self.samples[:splitIdx]
      self.validationSamples = self.samples[splitIdx:]
      print("Number of train/valid samples: ", len(self.trainSamples), ",", len(self.validationSamples))
    else: # no split (all in valid) for test
      self.trainSamples = []
      self.validationSamples = [sample for sample in self.samples if sample.gtText.find('-')==-1] # remove samples with hyphens since model isnt trained on that
      print("Number of test samples: ", len(self.validationSamples))

    # put words into lists
    self.trainWords = [x.gtText for x in self.trainSamples]
    self.validationWords = [x.gtText for x in self.validationSamples]

    # number of randomly chosen samples per epoch for training
    self.numTrainSamplesPerEpoch = len(self.trainSamples)

    # start with train set
    self.trainSet()

    # list of all chars in dataset
    self.charList = sorted(list(chars))

  def trainSet(self):
    "switch to randomly chosen subset of training set"
    self.dataAugmentation = True
    self.currIdx = 0
    random.shuffle(self.trainSamples)
    self.samples = self.trainSamples[:self.numTrainSamplesPerEpoch]

  def validationSet(self):
    "switch to validation set"
    self.dataAugmentation = False
    self.currIdx = 0
    self.samples = self.validationSamples

  def getIteratorInfo(self):
    "current batch index and overall number of batches"
    return (self.currIdx // self.batchSize + 1, len(self.samples) // self.batchSize)

  def hasNext(self):
    "iterator"
    return self.currIdx + self.batchSize <= len(self.samples)

  def getNext(self, is_testing=False):
    "iterator"
    batchRange = range(self.currIdx, self.currIdx + self.batchSize)
    gtTexts = [self.samples[i].gtText for i in batchRange]
    imgs = [preprocess(cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE),
                       self.imgSize, self.args, self.dataAugmentation, is_testing) for i in batchRange]
    self.currIdx += self.batchSize
    return Batch(gtTexts, imgs)

  iam = IAM(join(home, 'datasets'))
