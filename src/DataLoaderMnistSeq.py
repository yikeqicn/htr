import random
import numpy as np
import cv2
from SamplePreprocessor import preprocess
from glob import glob
from os.path import join
import os

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

  def __init__(self, filePath, batchSize, imgSize, maxTextLen):
    "loader for dataset at given location, preprocess images and text according to parameters"

    assert filePath[-1] == '/'

    self.dataAugmentation = False
    self.currIdx = 0
    self.batchSize = batchSize
    self.imgSize = imgSize
    self.samples = []

    # f = open(filePath + 'words.txt')
    # chars = set()
    # for line in f:
    #   # ignore comment line
    #   if not line or line[0] == '#':
    #     continue
    #
    #   lineSplit = line.strip().split(' ')
    #   assert len(lineSplit) >= 9
    #
    #   # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
    #   fileNameSplit = lineSplit[0].split('-')
    #   fileName = filePath + 'words/' + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' + \
    #              lineSplit[0] + '.png'
    #
    #   # GT text are columns starting at 9
    #   gtText = ' '.join(lineSplit[8:])[:maxTextLen]
    #   chars = chars.union(set(list(gtText)))
    #
    #   # put sample into list
    #   self.samples.append(Sample(gtText, fileName))

    # MODIFIED HERE FOR OUR CUSTOM DATASET
    fileName = glob(join(filePath, '*.jpg'))
    gtText = [os.path.basename(f)[:-4] for f in fileName]
    chars = set.union(*[set(t) for t in gtText])
    self.samples = [Sample(g,f) for g,f in zip(gtText, fileName)]

    # split into training and validation set: 95% - 5%
    random.shuffle(self.samples)
    splitIdx = int(0.95 * len(self.samples))
    self.trainSamples = self.samples[:splitIdx]
    self.validationSamples = self.samples[splitIdx:]
    print("Number of train/valid samples: ", len(self.trainSamples), ",", len(self.validationSamples))
    print('Batch size: '+str(self.batchSize))

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

  def getNext(self):
    "iterator"
    batchRange = range(self.currIdx, self.currIdx + self.batchSize)
    gtTexts = [self.samples[i].gtText for i in batchRange]
    imgs = [preprocess(cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE),
                       self.imgSize, self.dataAugmentation) for i in batchRange]
    self.currIdx += self.batchSize
    return Batch(gtTexts, imgs)