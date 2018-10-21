import cv2
import numpy as np
from os.path import join
import matplotlib.pyplot as plt

##
dataroot = '../data/words'
img = cv2.imread(join(dataroot, 'a01/a01-003u/a01-003u-00-00.png'))
img = img.mean(axis=2)
plt.imshow(img)
plt.colorbar()
plt.show()

power = 4
ul = np.exp(-(xx**power+yy**power)/sigma**power)

