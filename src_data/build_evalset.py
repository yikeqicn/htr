from multiprocessing import Pool
from glob import glob
from os.path import join, basename
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import subprocess
from utils_uzn import coords2crops, read_uzn

pagenumber = 1
imgAlignDir = join('nw_im_reg', str(pagenumber))
uznDir = 'uzn'
outDir = 'nw_im_crop'
allFiles = glob(join(imgAlignDir, '*.jpg'))
os.makedirs(outDir, exist_ok=True)

uznFile = join(uznDir, 'page'+str(pagenumber)+'.uzn')
coords = read_uzn(uznFile)

for file in allFiles:

  img = cv2.imread(file, cv2.IMREAD_COLOR)
  crops = coords2crops(img, coords)
  for imcrop, namecrop in crops:
    outFile = join(outDir, basename(file[:-4]) +'-' + namecrop + '.jpg')
    cv2.imwrite(outFile, imcrop)
