import numpy as np
import cv2
from os.path import basename, join
import os
import sys
import matplotlib.pyplot as plt

def read_uzn(uznFile):
  '''Read uzn file as a list of box coordinates coords'''
  coords = open(uznFile,'r').readlines()
  coords = [c.split() for c in coords]
  coords = coords[1:]
  return coords

def coord2crop(imRef, coord):
  '''Extract cropped image/name out of reference image using coordinates given in coord'''
  xywh = [int(c) for c in coord[:4]]
  imCrop = imRef[xywh[1]:xywh[1] + xywh[3], xywh[0]:xywh[0] + xywh[2]]
  nameCrop = coord[4]
  return imCrop, nameCrop

def coords2crops(imRef, coords):
  '''Extract all crops from imRef given list of coord in coords'''
  crops = [coord2crop(imRef, coord) for coord in coords]
  return crops

def crop_from_file(files, uznFile, saveDir):
  '''crop all patches from list of image files using the coordinates given in uznFile. Save patches in saveDir'''
  os.makedirs(saveDir, exist_ok=True)
  coords = read_uzn(uznFile)
  for file in files:
    im = cv2.imread(file, cv2.IMREAD_COLOR)
    crops = coords2crops(im, coords)
    for imcrop, namecrop in crops:
      saveFile = join(saveDir, basename(file[:-4]) +'-' + namecrop + '.jpg')
      cv2.imwrite(saveFile, imcrop)
