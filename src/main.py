
from comet_ml import Experiment
experiment = Experiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", parse_args=False,
                        project_name='htr-restart')

import sys
import argparse
import cv2
import editdistance
import numpy as np
from DataLoader import DataLoader, Batch
# from DataLoaderMnistSeq import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess
import os
from os.path import join, basename, dirname
import matplotlib.pyplot as plt
import shutil
import utils

# optional command line args
parser = argparse.ArgumentParser()
parser.add_argument("--train", help="train the NN", action="store_true")
parser.add_argument("--validate", help="validate the NN", action="store_true")
parser.add_argument("--beamsearch", help="use beam search instead of best path decoding", action="store_true")
parser.add_argument("--wordbeamsearch", help="use word beam search instead of best path decoding", action="store_true")
parser.add_argument("--name", default='debug', type=str, help="name of the log")
parser.add_argument("--gpu", default='0', type=str, help="gpu numbers")
parser.add_argument("--batchsize", default=50, type=int, help='batch size')
parser.add_argument("--custom", help="custom augmentation", action="store_true")
parser.add_argument("--dataset", default='iam', type=str, help='[iam, mnistseq]')
args = parser.parse_args()
experiment.set_name(args.name)
experiment.log_multiple_params(vars(args))
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

home = os.environ['HOME']
reporoot = join(home, 'repo')
ckptroot = join(home, 'ckpt')
ckptpath = join(ckptroot, args.name)
if args.name=='debug': shutil.rmtree(ckptpath, ignore_errors=True)
os.makedirs(ckptpath, exist_ok=True)

# chublet
class FilePaths:
  "filenames and paths to data"

  fnCkptpath = ckptpath
  fnCharList = join(ckptpath, 'charList.txt')
  fnCorpus = join(ckptpath, 'corpus.txt')
  fnAccuracy = join(ckptpath, 'accuracy.txt')
  if args.dataset=='mnistseq': fnTrain = '/data/home/jdegange/vision/digitsdataset2/'
  if args.dataset=='iam': fnTrain = join(home, 'datasets/iam_handwriting/')
  fnInfer = join(home, 'datasets', 'htr_debug', 'trainbold.png')

def train(model, loader):
  "train NN"
  epoch = 0  # number of training epochs since start
  bestCharErrorRate = float('inf')  # best valdiation character error rate
  noImprovementSince = 0  # number of epochs no improvement of character error rate occured
  earlyStopping = 5  # stop training after this number of epochs without improvement
  while True:
    epoch += 1
    print('Epoch:', epoch, ' Training...')

    # train
    loader.trainSet()
    counter = 0
    while loader.hasNext():
      iterInfo = loader.getIteratorInfo()
      batch = loader.getNext()
      loss = model.trainBatch(batch)
      if np.mod(iterInfo[0],200)==0: print('TRAIN: Batch:', iterInfo[0], '/', iterInfo[1], 'Loss:', loss)
      step = iterInfo[0]+(epoch-1)*iterInfo[1]
      experiment.log_metric('train/loss', loss, step)
      if counter<5: # log images
        text = batch.gtTexts[counter]
        utils.log_image(experiment, batch, text, 'train', ckptpath, counter, epoch)
        counter += 1

    # validate
    charErrorRate, wordAccuracy= validate(model, loader, epoch)
    experiment.log_metric('valid/cer', charErrorRate, step)
    experiment.log_metric('valid/wer', 1-wordAccuracy, step)

    # if best validation accuracy so far, save model parameters
    if charErrorRate < bestCharErrorRate:
      print('Character error rate improved, save model')
      bestCharErrorRate = charErrorRate
      noImprovementSince = 0
      model.save()
      open(FilePaths.fnAccuracy, 'w').write(
        'Validation character error rate of saved model: %f%%' % (charErrorRate * 100.0))
    else:
      print('Character error rate not improved')
      noImprovementSince += 1

    # stop training if no more improvement in the last x epochs
    if noImprovementSince >= earlyStopping:
      print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
      break


def validate(model, loader, epoch):
  "validate NN"
  print('Validate NN')
  loader.validationSet()
  numCharErr, numCharTotal, numWordOK, numWordTotal = 0, 0, 0, 0
  plt.figure(figsize=(6,2))
  counter = 0
  while loader.hasNext():
    iterInfo = loader.getIteratorInfo()
    batch = loader.getNext()
    recognized = model.inferBatch(batch)
    for i in range(len(recognized)):
      numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
      numWordTotal += 1
      dist = editdistance.eval(recognized[i], batch.gtTexts[i])
      numCharErr += dist
      numCharTotal += len(batch.gtTexts[i])
      if counter<10: # log images
        text = ' '.join(['[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"'])
        utils.log_image(experiment, batch, text, 'valid', ckptpath, counter, epoch)
        counter += 1

  # print validation result
  charErrorRate = numCharErr / numCharTotal
  wordAccuracy = numWordOK / numWordTotal
  print('VALID: Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate * 100.0, wordAccuracy * 100.0))
  return charErrorRate, wordAccuracy


def infer(model, fnImg):
  "recognize text in image provided by file path"
  img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize, args)
  batch = Batch(None, [img] * args.batchsize)  # fill all batch elements with same input image
  recognized = model.inferBatch(batch)  # recognize text
  print('Recognized:', '"' + recognized[0] + '"')  # all batch elements hold same result


def main():
  "main function"

  decoderType = DecoderType.BestPath
  if args.beamsearch:
    decoderType = DecoderType.BeamSearch
  elif args.wordbeamsearch:
    decoderType = DecoderType.WordBeamSearch

  # train or validate on IAM dataset
  if args.train or args.validate:
    # load training data, create TF model
    loader = DataLoader(FilePaths.fnTrain, args.batchsize, Model.imgSize, Model.maxTextLen, args)

    # save characters of model for inference mode
    open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))

    # save words contained in dataset into file
    open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))

    # execute training or validation
    if args.train:
      model = Model(args, loader.charList, decoderType, FilePaths=FilePaths)
      train(model, loader)
    elif args.validate:
      model = Model(args, loader.charList, decoderType, mustRestore=True, FilePaths=FilePaths)
      validate(model, loader)

  # infer text on test image
  else:
    print(open(FilePaths.fnAccuracy).read())
    model = Model(args, open(FilePaths.fnCharList).read(), decoderType, mustRestore=True, FilePaths=FilePaths)
    print(FilePaths.fnCkptpath)
    model = Model(args, open(FilePaths.fnCharList).read(), mustRestore=True, FilePaths=FilePaths)
    infer(model, FilePaths.fnInfer)


if __name__ == '__main__':
  main()
