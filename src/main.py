from comet_ml import Experiment
experiment = Experiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", parse_args=False,
                        project_name='iam-hpsearch')

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
import sys
home = os.environ['HOME']

# basic operations
parser = argparse.ArgumentParser()
parser.add_argument("--name", default='debug', type=str, help="name of the log")
parser.add_argument("--gpu", default='0', type=str, help="gpu numbers")
parser.add_argument("--train", help="train the NN", action="store_true")
parser.add_argument("--validate", help="validate the NN", action="store_true")
parser.add_argument("--transfer_from", default='/root/ckpt/adam-lr1e-2', type=str, help="name of pretrained model for transfer learning")
parser.add_argument("--batchesTrained", default=0, type=int, help='number of batches already trained (for lr schedule)')
# beam search
parser.add_argument("--beamsearch", help="use beam search instead of best path decoding", action="store_true")
parser.add_argument("--wordbeamsearch", help="use word beam search instead of best path decoding", action="store_true")
# training hyperparams
parser.add_argument("--batchsize", default=50, type=int, help='batch size')
parser.add_argument("--lrInit", default=1e-2, type=float, help='initial learning rate')
parser.add_argument("--optimizer", default='rmsprop', help="adam, rmsprop, momentum")
parser.add_argument("--wdec", default=1e-4, type=float, help='weight decay')
# trainset hyperparams
parser.add_argument("--noncustom", help="noncustom (original) augmentation technique", action="store_true")
parser.add_argument("--noartifact", help="dont insert artifcats", action="store_true")
parser.add_argument("--iam", help='use iam dataset', action='store_true')
parser.add_argument("--datapath", default='/root/datasets/htr_assets/crowdsource/processed/', type=str, help="train/valid path if not using iam")
parser.add_argument("--testpath", default='/root/datasets/htr_assets/nw_im_crop_curated/', type=str, help="test path ")
# densenet hyperparams
parser.add_argument("--nondensenet", help="noncustom (original) vanilla cnn", action="store_true")
parser.add_argument("--growth_rate", default=12, type=int, help='growth rate (k)')
parser.add_argument("--layers_per_block", default=18, type=int, help='number of layers per block')
parser.add_argument("--total_blocks", default=5, type=int, help='nuber of densenet blocks')
parser.add_argument("--keep_prob", default=1, type=float, help='keep probability in dropout')
parser.add_argument("--reduction", default=0.4, type=float, help='reduction factor in 1x1 conv in transition layers')
parser.add_argument("--bc_mode", default=True, type=bool, help="bottleneck and compresssion mode")
# rnn hyperparams
parser.add_argument("--rnndim", default=256, type=int, help='rnn dimenstionality')
parser.add_argument("--rnnsteps", default=32, type=int, help='number of desired time steps (image slices) to feed rnn')
# img size
parser.add_argument("--imgsize", default=[128,32], type=int, nargs='+')
# testset crop
parser.add_argument("--crop_r1", default=3, type=int)
parser.add_argument("--crop_r2", default=28, type=int)
parser.add_argument("--crop_c1", default=10, type=int)
parser.add_argument("--crop_c2", default=115, type=int)
args = parser.parse_args()

open('commands.log','a').write('nohup python '+' '.join(sys.argv)+' &\n') # write command to the log

experiment.set_name(args.name)
experiment.log_multiple_params(vars(args))
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

reporoot = join(home, 'repo')
ckptroot = join(home, 'ckpt')
ckptpath = join(ckptroot, args.name)
if args.name=='debug': shutil.rmtree(ckptpath, ignore_errors=True)
os.makedirs(ckptpath, exist_ok=True)

# chublet
class FilePaths:
  "filenames and paths to data"
  fnCkptpath = ckptpath
  fnTransferFrom = join(ckptroot, args.transfer_from) if args.transfer_from!='' else None
  fnCharList = join(ckptpath, 'charList.txt')
  fnCorpus = join(ckptpath, 'corpus.txt')
  fnAccuracy = join(ckptpath, 'accuracy.txt')
  # fnTrain = '/data/home/jdegange/vision/digitsdataset2/' # mnist digit sequences
  # fnTrain = '/root/datasets/htr_assets/crowdsource/extracted/' # ey handwritten digit strings
  fnTrain = args.datapath
  if args.iam: fnTrain = join(home, 'datasets/iam_handwriting/')
  fnInfer = join(home, 'datasets', 'htr_debug', 'trainbold.png')

def train(model, loader, testloader=None):
  "train NN"
  epoch = 0  # number of training epochs since start
  bestCharErrorRate = bestWordErrorRate = float('inf')  # best valdiation character error rate
  noImprovementSince = 0  # number of epochs no improvement of character error rate occured
  earlyStopping = 12  # stop training after this number of epochs without improvement
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
      step = iterInfo[0]+(epoch-1)*iterInfo[1]
      if np.mod(iterInfo[0],200)==0:
        print('TRAIN: Batch:', iterInfo[0], '/', iterInfo[1], 'Loss:', loss)
        experiment.log_metric('train/loss', loss, step)
      if counter<5: # log images
        text = batch.gtTexts[counter]
        utils.log_image(experiment, batch, text, 'train', ckptpath, counter, epoch)
        counter += 1

    # validate
    charErrorRate, wordAccuracy= validate(model, loader, epoch)
    experiment.log_metric('valid/cer', charErrorRate, step)
    experiment.log_metric('valid/wer', 1-wordAccuracy, step)

    # test
    if testloader!=None:
      charErrorRate, wordAccuracy= validate(model, testloader, epoch, is_testing=True)
      experiment.log_metric('test/cer', charErrorRate, step)
      experiment.log_metric('test/wer', 1-wordAccuracy, step)

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
    if 1-wordAccuracy < bestWordErrorRate:
      bestWordErrorRate = 1-wordAccuracy
    experiment.log_metric('best/cer', bestCharErrorRate, step)
    experiment.log_metric('best/wer', bestWordErrorRate, step)

    # # stop training if no more improvement in the last x epochs
    # if noImprovementSince >= earlyStopping:
    #   print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
    #   break
    if epoch>=50: print('Done with training at epoch', epoch, 'bestCharErrorRate='+str(bestCharErrorRate)); break


def validate(model, loader, epoch, is_testing=False):
  "validate NN"
  if not is_testing: print('Validating NN')
  else: print('Testing NN')
  loader.validationSet()
  numCharErr, numCharTotal, numWordOK, numWordTotal = 0, 0, 0, 0
  plt.figure(figsize=(6,2))
  counter = 0
  n_log = 20 if is_testing else 10
  while loader.hasNext():
    iterInfo = loader.getIteratorInfo()
    batch = loader.getNext(is_testing)
    recognized = model.inferBatch(batch)
    for i in range(len(recognized)):
      numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
      numWordTotal += 1
      dist = editdistance.eval(recognized[i], batch.gtTexts[i])
      numCharErr += dist
      numCharTotal += len(batch.gtTexts[i])
      if counter<n_log: # log images
        text = ' '.join(['[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"'])
        utils.log_image(experiment, batch, text, 'test' if is_testing else 'valid', ckptpath, counter, epoch)
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
    loader = DataLoader(FilePaths.fnTrain, args.batchsize, args.imgsize, Model.maxTextLen, args)
    testloader = DataLoader(args.testpath, args.batchsize, args.imgsize, Model.maxTextLen, args, is_test=True)

    # save characters of model for inference mode
    open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))

    # save words contained in dataset into file
    open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))

    # execute training or validation
    if args.train:
      model = Model(args, loader.charList, decoderType, FilePaths=FilePaths)
      train(model, loader, testloader)
    elif args.validate:
      model = Model(args, loader.charList, decoderType, mustRestore=True, FilePaths=FilePaths)
      validate(model, loader)

  # infer text on test image
  else:
    print(open(FilePaths.fnAccuracy).read())
    model = Model(args, open(FilePaths.fnCharList).read(), decoderType, mustRestore=True, FilePaths=FilePaths)
    infer(model, FilePaths.fnInfer)


if __name__ == '__main__':
  main()
