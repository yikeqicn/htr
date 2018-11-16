from PIL import Image
import numpy as np
import cv2
from functools import reduce
from os.path import basename, join
import os
import sys
from matplotlib.pyplot import plot, imshow, show, colorbar

def remove_background(im, threshold):
  mask = im < threshold
  imMasked = im.copy()
  imMasked[mask] = 0
  return imMasked

def merge_patch(imBase, imPatch, centroid, threshold=100):
  '''Takes imPatch and superimpose on imBase at centroid. Returns modified image'''

  imBase, imPatch = 255-imBase, 255-imPatch # invert images fro processing
  nrb, ncb = imBase.shape
  nrp, ncp = imPatch.shape

  # make white areas of imPatch transparent
  imPatchMasked = remove_background(imPatch, threshold)

  # get difference of centroids between base and patch
  centroidPatch = np.array([int(dim/2) for dim in imPatchMasked.shape])
  delta = np.array(centroid) - centroidPatch

  # add difference of centroids to the x,y position of patch
  cc, rr = np.meshgrid(np.arange(ncp), np.arange(nrp))
  rr = rr + delta[0]
  cc = cc + delta[1]

  # remove all parts of patch image that would expand base image
  keep = reduce(np.logical_and, [rr>=0, rr<nrb, cc>=0, cc<ncb])
  nrk, nck = np.max(rr[keep])-np.min(rr[keep])+1, np.max(cc[keep])-np.min(cc[keep])+1
  imPatchKeep = imPatchMasked[keep]


  # merge base and patch by taking maximum pixel at each position
  imMerge = imBase.copy()
  imBaseCrop = imBase.copy()
  imBaseCrop = imBaseCrop[rr[keep], cc[keep]]
  imMerge[rr[keep], cc[keep]] = np.maximum(imBaseCrop, imPatchKeep)

  return 255-imMerge # invert back

file = '/Users/dl367ny/htrdata/crowdsource/extracted/111003/$9,900,000.jpg'
patchFile = '/Users/dl367ny/htrdata/cropped_patches/nw_horizontal-2/Declined - Handwritten (1)_Redacted-2-aligned-Unnamed2.jpg'
imBase = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
im = cv2.imread(patchFile, cv2.IMREAD_GRAYSCALE)
im = cv2.resize(im, None, fx=3, fy=1)+50

nrb, ncb = imBase.shape
centroid = int(.4*nrb), int(ncb/2)
imMerge = merge_patch(imBase, im, centroid, 100)
Image.fromarray(imMerge).show()
