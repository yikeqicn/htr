import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pdf2image
import os
from os.path import join
import pyzbar.pyzbar as pyzbar
import qrcode

seed = 112800
nPageToGen = 100
probComma = .8
probDollar = .2
probDecimal = .3
HOME = os.environ['HOME']
crowdRoot = join(HOME, 'datasets', 'htr_assets', 'crowdsource')
worksheetRoot = 'worksheets'
os.makedirs(join(crowdRoot, 'generated'), exist_ok=True)


def insert_commas(txt):
  '''Insert commas separating every 3 order of magnitudes into the string of numbers'''
  numberListMod = []
  counter = 0
  for i in range(len(txt)-1, -1, -1):
    numberListMod.append(txt[i])
    if i<=len(txt)-4: counter += 1
    if counter==3 and i!=0:
      numberListMod.append(',')
      counter = 0
  return ''.join(numberListMod)[::-1]

def insert_dollarsign(txt):
  '''insert dollarsign, must be after inserting commas'''
  return '$'+txt

def strip_decimal(txt):
  '''strip decimal and all numbers to the left, must be after inserting comma'''
  return txt[:-3]

def generate_numberstr(probComma, probDollar, probDecimal):
  '''generate random number and place into string with artifacts'''
  log = np.random.uniform(3,8)
  number = 10**log
  number = np.around(number, -np.random.randint(-2,int(np.log10(number))))
  txt = '%.2f'%number
  txt = insert_commas(txt) if np.random.rand()<probComma else txt
  txt = insert_dollarsign(txt) if np.random.rand()<probDollar else txt
  txt = strip_decimal(txt) if np.random.rand()>probDecimal else txt
  return txt

def generate_page(imBlank, seed):
  '''fill imBlank with random text'''

  xAnchor = 168
  xAnchor2 = 450
  yAnchor = 266
  xCnt = 3
  yCnt = 26
  xSpacing = 770
  ySpacing = 107

  labels = {}
  im = imBlank.copy()
  im.paste(qrcode.make(seed), box=(870, 2920))
  im = np.array(im)

  for i in range(xCnt):
    for j in range(yCnt):

      # generate the string
      txt = generate_numberstr(probComma, probDollar, probDecimal)

      # special case of first entry
      if i==0 and j==0:
        txt = insert_dollarsign(insert_commas('%.2f'%(seed/100)))
        textorg = (xAnchor2 + i * xSpacing, yAnchor + j * ySpacing)
        font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
        cv2.putText(im, txt, textorg, font, fontScale=1, color=[0,0,0], thickness=2)

      # general case
      textorg = (xAnchor + i * xSpacing, yAnchor + j * ySpacing)
      font = cv2.FONT_HERSHEY_SIMPLEX
      cv2.putText(im, txt, textorg, font, fontScale=1, color=[0,0,0], thickness=2)
      labels[(i,j)] = txt

  # np.save(str(seed)+'.npy', labels)
  pickle.dump(labels, open(join(crowdRoot, 'generated', 'label-'+str(seed) + '.pkl'), 'wb'))
  im = Image.fromarray(np.uint8(im))
  return im

np.random.seed(seed)
pagesBlank = pdf2image.convert_from_path(join(crowdRoot, 'worksheet.pdf'), dpi=300, thread_count=6)
imBlank = pagesBlank[0]
pages = [generate_page(imBlank, seed+i) for i in range(nPageToGen)]
basename = str(seed)+'-'+str(seed+nPageToGen-1)+'.pdf'
pages[0].save(join(crowdRoot, 'generated', basename), format='PDF', resolution=100, save_all=True, append_images=pages[1:])
