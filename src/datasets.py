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
      if not exists(join(self.root, 'words')):
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

      # put sample into list
      # qyk exclude empty images
      img_test=cv2.imread(fileName, cv2.IMREAD_GRAYSCALE) #qyk
      if not (img_test is None or np.min(img_test.shape) <= 1): #qyk
        self.samples.append( (fileName, label) ) #qyk

      # makes list of characters
      chars = chars.union(set(list(label)))
      self.charList = sorted(list(chars))

  def __str__(self):
    return 'IAM words dataset. Data location: '+self.root+', Length: '+str(len(self))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):

    label = self.samples[idx][1]
    # img = preprocess(cv2.imread(self.samples[i][0], cv2.IMREAD_GRAYSCALE),
    #                  args.imgsize, self.args, False, is_testing)
    img = cv2.imread(self.samples[idx][0], cv2.IMREAD_GRAYSCALE)
    #print(self.samples[idx][0])#
    #print(self.samples[idx][1]) #for debug purpose
    if self.transform:
      img = self.transform(img)

    return img, label

class EyDigitStrings(data.Dataset):

  def __init__(self, root='/root/datasets', transform=None):

    self.transform = transform
    self.root = join(root, 'htr_assets/crowdsource/processed')
    maybe_download(source_url='https://www.dropbox.com/s/dsg41kaajrfvfvj/htr_assets.zip?dl=0',
                   filename='htr_assets', target_directory=root, filetype='zip') # qyk added, the source is yq's dropbox
    # custom dataset loader
    allfiles = glob(join(self.root, '**/*.jpg'), recursive=True)
    labels = [basename(f)[:-4] if basename(f).find('empty-')==-1 else '_' for f in allfiles] # if filename has 'empty-', then the ground truth is nothing
    self.samples = list(zip(allfiles,labels))

    # makes list of characters
    chars = set.union(*[set(l) for l in labels])
    self.charList = sorted(list(chars))

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

class IRS(data.Dataset):

  def __init__(self, root='/root/datasets', transform=None):

    self.transform = transform
    self.root = join(root, 'irs_handwriting')
    maybe_download(source_url='https://www.dropbox.com/s/54jarzcb0mju32d/img_cropped_irs.zip?dl=0', filename='irs_handwriting', target_directory=root, filetype='zip')
    if exists(join(root, 'img_cropped_irs')): os.system('mv '+join(root, 'img_cropped_irs')+' '+self.root)

    folder_depth = 3
    allfiles = glob(join(self.root, '**/'*folder_depth+'*.jpg'))
    labels = [basename(f)[:-4] for f in allfiles]
    self.samples = list(zip(allfiles, labels))
    # makes list of characters
    chars = set.union(*[set(l) for l in labels])
    self.charList = sorted(list(chars))

  def __len__(self):
    return len(self.samples)

  def __str__(self):
    return 'IRS dataset. Data location: '+self.root+', Length: '+str(len(self))

  def __getitem__(self, idx):

    label = self.samples[idx][1]
    img = cv2.imread(self.samples[idx][0], cv2.IMREAD_GRAYSCALE)
    if self.transform:
      img = self.transform(img)

    return img, label

class PRT(data.Dataset):

  def __init__(self, root='/root/datasets', transform=None):

    self.transform = transform
    self.root = join(root, 'img_print_100000')
    maybe_download(source_url='https://www.dropbox.com/s/cbhpy6clfi9a5lz/img_print_100000_clean.zip?dl=0',filename='img_print_100000_clean', target_directory=root, filetype='zip')
    #yq patch delete unrecognized non-english samples in linux
    #os.system('find '+ root+' -maxdepth 1 -name "*.jpg" -type f -delete') find ./logs/examples -maxdepth 1 -name "*.log"
    if exists(join(root, 'img_print_100000_clean')): os.system('mv ' + join(root, 'img_print_100000_clean') + ' ' + self.root)

    folder_depth = 1
    allfiles = glob(join(self.root, '**/' * folder_depth + '*.jpg'))
    allfiles = [f for f in allfiles if len(basename(f))-4<=25 and (not '#U' in f) and (not '---' in f)] # screen out non-recognized characters qyk
    labels = [basename(f)[:-4] for f in allfiles]
    self.samples = list(zip(allfiles, labels))
    # makes list of characters
    chars = set.union(*[set(l) for l in labels])
    self.charList = sorted(list(chars))

  def __len__(self):
    return len(self.samples)

  def __str__(self):
    return 'Printing dataset. Data location: ' + self.root + ', Length: ' + str(len(self))

  def __getitem__(self, idx):

    label = self.samples[idx][1]
    img = cv2.imread(self.samples[idx][0], cv2.IMREAD_GRAYSCALE)
    if self.transform:
      img = self.transform(img)

    return img, label


# dataroot = join(home,'datasets')
# iam = IAM(dataroot)
# ey = EyDigitStrings(dataroot)
# irs = IRS(dataroot)
