import numpy as np
import cv2
from matplotlib.pyplot import plot, imshow, colorbar, show, axis, hist
from PIL import Image
import os
from glob import glob

def clean_lines(img, threshold=.23):
  '''use hough transnform to remove lines from the ey dataset'''
  img_copy = img.copy()
  if len(img.shape)>2: gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
  else: gray = img_copy.copy()
  largerDim = np.max(gray.shape)
  origShape = gray.shape
  sqrShape = [largerDim, largerDim]

  # image preprocessing for the hough transform
  gray = cv2.resize(gray, (largerDim, largerDim)) # resize to be square so that votes for both horz and vert lines are equal
  gray = cv2.GaussianBlur(gray,(5,5),0)
  edges = cv2.Canny(gray, 50, 150, apertureSize=3) # edge detection
  # Image.fromarray(edges).show() # debug

  # apply hough transform
  width = edges.shape[0]
  thresholdPix = int( threshold * width ) # threshold is percentage of full image width expected to get votes
  lines = cv2.HoughLines(edges, 1, 1 * np.pi / 180, threshold=thresholdPix)

  # loop over detected lines in hough space and convert to euclidean
  for rho, theta in np.squeeze(lines):
    # leverage the fact that we know the lines occur at the borders of the image and are horz or vert
    conditionTheta = (abs(180/np.pi * theta - 0  ) < 3) | \
                     (abs(180/np.pi * theta - 90 ) < 3) | \
                     (abs(180/np.pi * theta - 180) < 3) | \
                     (abs(180/np.pi * theta - 270) < 3) | \
                     (abs(180/np.pi * theta - 360) < 3)
    conditionRho = (abs(180/np.pi * theta - 0  ) < 3) & (abs(rho - 0    ) < .07*width) | \
                   (abs(180/np.pi * theta - 0  ) < 3) & (abs(rho - width) < .07*width) | \
                   (abs(180/np.pi * theta - 0  ) < 3) & (abs(rho + width) < .07*width) | \
                   (abs(180/np.pi * theta - 180) < 3) & (abs(rho - 0    ) < .07*width) | \
                   (abs(180/np.pi * theta - 180) < 3) & (abs(rho - width) < .07*width) | \
                   (abs(180/np.pi * theta - 180) < 3) & (abs(rho + width) < .07*width) | \
                   (abs(180/np.pi * theta - 90) < 3) & (abs(rho - 0    ) < .2*width) | \
                   (abs(180/np.pi * theta - 90) < 3) & (abs(rho - width) < .2*width) | \
                   (abs(180/np.pi * theta - 90) < 3) & (abs(rho + width) < .2*width)
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
      x1 = int( x1 * origShape[1]/sqrShape[1] )
      x2 = int( x2 * origShape[1]/sqrShape[1] )
      y1 = int( y1 * origShape[0]/sqrShape[0] )
      y2 = int( y2 * origShape[0]/sqrShape[0] )
      cv2.line(img_copy, (x1, y1), (x2, y2), (255, 255, 255), thickness=14)
    else:
      # plot( rho, theta, '.b' , markersize=4) # debug
      pass
  return img_copy

def tight_crop(img, threshold=1-1.5e-2):
  '''tightly crop an image, removing whitespace'''
  img_copy = 255 - img
  if len(img_copy.shape)>2: img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
  img_copy[img_copy>20] = 255 # binarize
  img_copy[img_copy<=20] = 0
  # Image.fromarray(img_copy).show() # debug
  img_copy = cv2.erode(img_copy, np.ones((3,3)))

  # function: whiten the border given the crop coordinates
  def clean_border(img, r1, r2, c1, c2, debug=False):
    img_copy = img.copy()
    if debug:
      img_copy[:r1, :]  = 125
      img_copy[-r2:, :] = 125
      img_copy[:, :c1]  = 125
      img_copy[:, -c2:] = 125
    else:
      img_copy[:r1, :]  = 0
      img_copy[-r2:, :] = 0
      img_copy[:, :c1]  = 0
      img_copy[:, -c2:] = 0
    return img_copy

  # function: calculate ratio of preserved black pixels after the border cleaning
  ratio_preserved = lambda crop: np.sum(clean_border(img_copy, crop[0], crop[1], crop[2], crop[3])) / np.sum(img_copy)

  # iteratively crop more and more on each side alternatingly till preservedRatio hits threshold
  nr, nc = img_copy.shape
  crop = [nr//2, nr//2, nc//2, nc//2]
  edgeId = -1
  subThreshold = 1e-2
  while subThreshold <= threshold:
    edgeId += 1
    increment = 1e-2 if subThreshold < .9*threshold else 1e-3
    subThreshold += increment
    nextCrop = crop.copy()
    while ratio_preserved(nextCrop) <= subThreshold:
      crop = nextCrop.copy()
      nextCrop[np.mod(edgeId,4)] -= 1
      print(ratio_preserved(nextCrop), nextCrop)
    Image.fromarray(clean_border(img_copy, crop[0], crop[1], crop[2], crop[3], debug=True)).show()
    print(crop, np.mod(edgeId,4), ratio_preserved(crop), subThreshold)

  return img[crop[0]:-crop[1], crop[2]:-crop[3]] # crop the image and return

def center_pad(img, pad):
  '''center crop the image by defining the amoiunt of negative padding to shrink'''
  return img[pad:-pad, pad:-pad]


files = glob('/Users/dl367ny/datasets/htr_assets/crowdsource/extracted/*/*.jpg')
# files = ['/Users/dl367ny/datasets/htr_assets/crowdsource/extracted/112301/42544.jpg']
# files = ['/Users/dl367ny/datasets/htr_assets/crowdsource/extracted/112116/719,000.jpg']
# files = ['/Users/dl367ny/datasets/htr_assets/crowdsource/extracted/112133/$341,510.jpg']
# files = ['/Users/dl367ny/datasets/htr_assets/crowdsource/extracted/112042/2,504,650.jpg']
for file in np.array(files)[np.random.permutation(len(files))[:10]]:

  img = Image.open(file)
  img = np.array(img)
  # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # img = center_pad(img, 10)
  # img = np.pad(img, 10, 'maximum')
  # Img = Image.fromarray(img); Img.show()
  img = clean_lines(img)
  img = tight_crop(img)
  Img = Image.fromarray(img); Img.show()

