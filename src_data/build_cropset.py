from multiprocessing import Pool
from glob import glob
from os.path import join, basename
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import subprocess
from utils_uzn import crop_from_file

pagenumber = 2
formDir = './registered_forms/nw_im_reg'
uznDir = './uzn'
patchDir = './cropped_patches'

# list of image files
formDir = join(formDir, str(pagenumber))
files = glob(join(formDir, '*.jpg'))

# uzn file
uznName = 'nw_horizontal-'+str(pagenumber)+'.uzn'
uznFile = join('uzn', uznName)

# save directory
saveDir = join(patchDir, uznName[:-4])

# crop all
crop_from_file(files, uznFile, saveDir)

