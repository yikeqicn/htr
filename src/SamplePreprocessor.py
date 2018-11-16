import random
import numpy as np
import cv2
from matplotlib.pyplot import plot, imshow, colorbar, show, axis
from numpy.random import randint
import os


def horizontal_stretch(img, minFactor, maxFactor):
  '''randomly stretch image horizontally by amount uniformly between minFactor and maxFactor'''
  return cv2.resize(img, (int(img.shape[1] * np.random.uniform(minFactor, maxFactor)), img.shape[0]))


def target_aspect_pad(img, targetRatio=32 / 128):
  '''change aspect ratio of image to targetRatio by padding one of the dimensions. original image will be placed in
  random location within the expanded canvas'''
  nr, nc = img.shape
  currentRatio = nr / nc
  if currentRatio > targetRatio:
    dc = int(nr * (1 / targetRatio - 1 / currentRatio))
    dc2 = randint(max(dc, 1))
    dc1 = dc - dc2
    padding = ((0, 0), (dc1, dc2))
  else:
    dr = int(nc * (targetRatio - currentRatio))
    dr2 = randint(max(dr, 1))
    dr1 = dr - dr2
    padding = ((dr1, dr2), (0, 0))
  img = np.pad(img, padding, 'maximum')
  return img


def keep_aspect_pad(img, maxFactor):
  '''pad image by such that it expands by rand(maxFactor) while keeping its aspect ratio fixed. original image will be
  placed in random location within tthe expanded canvas. maxFactor must be greater than 1'''
  nr, nc = img.shape
  ratio = nr / nc
  dc = randint(max(int((maxFactor - 1) * nc), 1))
  dr = int(ratio * dc)
  dc2 = randint(max(dc, 1))
  dc1 = dc - dc2
  dr2 = randint(max(dr, 1))
  dr1 = dr - dr2
  padding = ((dr1, dr2), (dc1, dc2))
  img = np.pad(img, padding, 'maximum')
  return img


def clean_lines(img, threshold=.23):
  '''use hough transnform to remove lines from the ey dataset'''
  img_copy = img.copy()
  if len(img.shape) > 2:
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
  else:
    gray = img_copy.copy()
  largerDim = np.max(gray.shape)
  origShape = gray.shape
  sqrShape = [largerDim, largerDim]

  # image preprocessing for the hough transform
  gray = cv2.resize(gray,
                    (largerDim, largerDim))  # resize to be square so that votes for both horz and vert lines are equal
  gray = cv2.GaussianBlur(gray, (5, 5), 0)
  edges = cv2.Canny(gray, 50, 150, apertureSize=3)  # edge detection
  # Image.fromarray(edges).show() # debug

  # apply hough transform
  width = edges.shape[0]
  thresholdPix = int(threshold * width)  # threshold is percentage of full image width expected to get votes
  lines = cv2.HoughLines(edges, 1, 1 * np.pi / 180, threshold=thresholdPix)

  # loop over detected lines in hough space and convert to euclidean
  for rho, theta in np.squeeze(lines):
    # leverage the fact that we know the lines occur at the borders of the image and are horz or vert
    conditionTheta = (abs(180 / np.pi * theta - 0) < 3) | \
                     (abs(180 / np.pi * theta - 90) < 3) | \
                     (abs(180 / np.pi * theta - 180) < 3) | \
                     (abs(180 / np.pi * theta - 270) < 3) | \
                     (abs(180 / np.pi * theta - 360) < 3)
    conditionRho = (abs(180 / np.pi * theta - 0) < 3) & (abs(rho - 0) < .07 * width) | \
                   (abs(180 / np.pi * theta - 0) < 3) & (abs(rho - width) < .07 * width) | \
                   (abs(180 / np.pi * theta - 0) < 3) & (abs(rho + width) < .07 * width) | \
                   (abs(180 / np.pi * theta - 180) < 3) & (abs(rho - 0) < .07 * width) | \
                   (abs(180 / np.pi * theta - 180) < 3) & (abs(rho - width) < .07 * width) | \
                   (abs(180 / np.pi * theta - 180) < 3) & (abs(rho + width) < .07 * width) | \
                   (abs(180 / np.pi * theta - 90) < 3) & (abs(rho - 0) < .2 * width) | \
                   (abs(180 / np.pi * theta - 90) < 3) & (abs(rho - width) < .2 * width) | \
                   (abs(180 / np.pi * theta - 90) < 3) & (abs(rho + width) < .2 * width)
    # draw the lines
    if conditionTheta & conditionRho:
      # plot( rho, theta, 'or' , markersize=4) # debug
      a = np.cos(theta)
      b = np.sin(theta)
      x0 = a * rho
      y0 = b * rho
      x1 = int(x0 + 1000 * (-b))
      y1 = int(y0 + 1000 * (a))
      x2 = int(x0 - 1000 * (-b))
      y2 = int(y0 - 1000 * (a))
      # scale back to original image size
      x1 = int(x1 * origShape[1] / sqrShape[1])
      x2 = int(x2 * origShape[1] / sqrShape[1])
      y1 = int(y1 * origShape[0] / sqrShape[0])
      y2 = int(y2 * origShape[0] / sqrShape[0])
      cv2.line(img_copy, (x1, y1), (x2, y2), (255, 255, 255), thickness=14)
    else:
      # plot( rho, theta, '.b' , markersize=4) # debug
      pass
  return img_copy


def tight_crop(img, threshold=1 - 1.5e-2):
  '''tightly crop an image, removing whitespace'''
  img_copy = 255 - img
  if len(img_copy.shape) > 2: img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
  img_copy[img_copy > 20] = 255  # binarize
  img_copy[img_copy <= 20] = 0
  # Image.fromarray(img_copy).show() # debug
  img_copy = cv2.erode(img_copy, np.ones((3, 3)))

  # function: whiten the border given the crop coordinates
  def clean_border(img, r1, r2, c1, c2, debug=False):
    img_copy = img.copy()
    if debug:
      img_copy[:r1, :] = 125
      img_copy[-r2:, :] = 125
      img_copy[:, :c1] = 125
      img_copy[:, -c2:] = 125
    else:
      img_copy[:r1, :] = 0
      img_copy[-r2:, :] = 0
      img_copy[:, :c1] = 0
      img_copy[:, -c2:] = 0
    return img_copy

  # function: calculate ratio of preserved black pixels after the border cleaning
  ratio_preserved = lambda crop: np.sum(clean_border(img_copy, crop[0], crop[1], crop[2], crop[3])) / np.sum(img_copy)

  # iteratively crop more and more on each side alternatingly till preservedRatio hits threshold
  crop = [0, 1, 0, 1]
  edgeId = -1
  subThreshold = 1
  increment = .5e-3
  while subThreshold >= threshold:
    edgeId += 1
    subThreshold -= increment
    nextCrop = crop.copy()
    while ratio_preserved(nextCrop) >= subThreshold:
      crop = nextCrop.copy()
      nextCrop[np.mod(edgeId, 4)] += 1
    # Image.fromarray(clean_border(img_copy, crop[0], crop[1], crop[2], crop[3], debug=True)).show()
    # print(crop, np.mod(edgeId,4), ratio_preserved(crop), subThreshold)

  return img[crop[0]:-crop[1], crop[2]:-crop[3]]  # crop the image and return


def center_pad(img, pad):
  '''center crop the image by defining the amoiunt of negative padding to shrink'''
  return img[pad:-pad, pad:-pad]


def preprocess(img, imgSize, args, dataAugmentation=False):
  "put img into target img of size imgSize, transpose for TF and normalize gray-values"

  # there are damaged files in IAM dataset - just use black image instead
  if img is None or np.min(img.shape) <= 1:
    img = np.zeros([imgSize[1], imgSize[0]])

  # increase dataset size by applying random stretches to the images
  if args.noncustom:

    if dataAugmentation:
      stretch = (random.random() - 0.5)  # -0.5 .. +0.5
      wStretched = max(int(img.shape[1] * (1 + stretch)), 1)  # random width, but at least 1
      img = cv2.resize(img, (wStretched, img.shape[0]))  # stretch horizontally by factor 0.5 .. 1.5

    # create target image and copy sample image into it
    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1),
               max(min(ht, int(h / f)), 1))  # scale according to f (result at least 1 and at most wt or ht)
    img = cv2.resize(img, newSize)
    target = np.ones([ht, wt]) * 255
    target[0:newSize[1], 0:newSize[0]] = img

  else:

    if True:  # dataAugmentation
      # ADDED by ronny
      img = clean_lines(img)
      img = tight_crop(img)
      img = standardize(img)
      img = horizontal_stretch(img, minFactor=.7, maxFactor=1.5)
      img = target_aspect_pad(img, targetRatio=imgSize[1] / imgSize[0])
      img = keep_aspect_pad(img, maxFactor=1.5)

    target = cv2.resize(img, imgSize, interpolation=cv2.INTER_CUBIC)

  # transpose for TF
  img = cv2.transpose(target)

  # normalize
  (m, s) = cv2.meanStdDev(img)
  m = m[0][0]
  s = s[0][0]
  img = img - m
  img = img / s if s > 0 else img

  return img
