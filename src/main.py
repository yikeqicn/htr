from comet_ml import Experiment
experiment = Experiment(api_key="YkPEmantOag1R1VOJmXz11hmt", parse_args=False, project_name='htr')
# yike: changed to my comet for debug
import sys
import argparse
import cv2
import editdistance
import numpy as np
from datasets import EyDigitStrings, IAM, IRS, PRT
from torch.utils.data import DataLoader, ConcatDataset, random_split
import torchvision
import torchvision.transforms as transforms
# from DataLoader import DataLoader, Batch
# from DataLoaderMnistSeq import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess
import os
from os.path import join, basename, dirname
import matplotlib.pyplot as plt
import shutil
import utils
import sys
import socket
home = os.environ['HOME']



# basic operations
parser = argparse.ArgumentParser()
parser.add_argument("-name", default='debug', type=str, help="name of the log")
parser.add_argument("-gpu", default='0', type=str, help="gpu numbers")
parser.add_argument("-train", help="train the NN", action="store_true")
parser.add_argument("-validate", help="validate the NN", action="store_true")
parser.add_argument("-transfer", action="store_true")
parser.add_argument("-batchesTrained", default=0, type=int, help='number of batches already trained (for lr schedule)')
# beam search
parser.add_argument("-beamsearch", help="use beam search instead of best path decoding", action="store_true")
parser.add_argument("-wordbeamsearch", help="use word beam search instead of best path decoding", action="store_true")
# training hyperparam
parser.add_argument("-batchsize", default=50, type=int, help='batch size')
parser.add_argument("-lrInit", default=1e-2, type=float, help='initial learning rate')
parser.add_argument("-optimizer", default='rmsprop', help="adam, rmsprop, momentum")
parser.add_argument("-wdec", default=1e-4, type=float, help='weight decay')
parser.add_argument("-lrDrop1", default=10, type=int, help='step to drop lr by 10 first time')
parser.add_argument("-lrDrop2", default=1000, type=int, help='step to drop lr by 10 sexond time')
parser.add_argument("-epochEnd", default=40, type=int, help='end after this many epochs')
# trainset hyperparam
parser.add_argument("-noncustom", help="noncustom (original) augmentation technique", action="store_true")
parser.add_argument("-noartifact", help="dont insert artifcats", action="store_true")
parser.add_argument("-iam", help='use iam dataset', action='store_true')
# densenet hyperparam
parser.add_argument("-nondensenet", help="use noncustom (original) vanilla cnn", action="store_true")
parser.add_argument("-growth_rate", default=12, type=int, help='growth rate (k)')
parser.add_argument("-layers_per_block", default=18, type=int, help='number of layers per block')
parser.add_argument("-total_blocks", default=5, type=int, help='nuber of densenet blocks')
parser.add_argument("-keep_prob", default=1, type=float, help='keep probability in dropout')
parser.add_argument("-reduction", default=0.4, type=float, help='reduction factor in 1x1 conv in transition layers')
parser.add_argument("-bc_mode", default=True, type=bool, help="bottleneck and compresssion mode")
# rnn,  hyperparams
parser.add_argument("-rnndim", default=256, type=int, help='rnn dimenstionality')
parser.add_argument("-rnnsteps", default=32, type=int, help='number of desired time steps (image slices) to feed rnn')
# img size
parser.add_argument("-imgsize", default=[128,32], type=int, nargs='+') #qyk change to 64, default 128,32
# testset crop
parser.add_argument("-crop_r1", default=3, type=int)
parser.add_argument("-crop_r2", default=28, type=int)
parser.add_argument("-crop_c1", default=10, type=int)
parser.add_argument("-crop_c2", default=115, type=int)
# filepaths
parser.add_argument("-dataroot", default='/root/datasets', type=str)
parser.add_argument("-ckptroot", default='/root/ckpt', type=str)
parser.add_argument("-urlTransferFrom", default=None, type=str)
args = parser.parse_args()

name = args.name
experiment.set_name(name)
experiment.log_parameters(vars(args))
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

reporoot = join(home, 'repo')
ckptroot = join(home, 'ckpt')
args.ckptpath = join(ckptroot, name)
if args.name=='debug': shutil.rmtree(args.ckptpath, ignore_errors=True)
os.makedirs(args.ckptpath, exist_ok=True)



# class FilePaths:
#   fnCkptpath = args.ckptpath
#   urlTransferFrom = 'https://www.dropbox.com/sh/vpgg5yah4hc0vjg/AADi2L6hDxXUn40JZPKus4ADa?dl=0'
#   fnCharList = join(args.ckptpath, 'charList.txt')
#   fnCorpus = join(args.ckptpath, 'corpus.txt')
#   fnAccuracy = join(args.ckptpath, 'accuracy.txt')
#   # fnTrain = '/data/home/jdegange/vision/digitsdataset2/' # mnist digit sequences
#   fnTrain = ['/root/datasets/htr_assets/crowdsource/processed/',
#              # '/root/datasets/htr_assets/nw_empty_patches/train/',
#              ]
#   fnTest = ['/root/datasets/htr_assets/nw_im_crop_curated/',
#             # '/root/datasets/htr_assets/nw_empty_patches/test/',
#             ]
#   if args.iam: fnTrain = join(home, 'datasets/iam_handwriting/')
#   fnInfer = join(home, 'datasets', 'htr_debug', 'trainbold.png')

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
    # loader = DataLoader(FilePaths.fnTrain, args.batchsize, args.imgsize, Model.maxTextLen, args)
    # testloader = DataLoader(FilePaths.fnTest, args.batchsize, args.imgsize, Model.maxTextLen, args, is_test=True)

    # tnansforms
    transform_train = transforms.Compose([
      #lambda img: (np.zeros([args.imgsize[1], args.imgsize[0]]) if img is None or np.min(img.shape) <= 1 else cv2.resize(img, (args.imgsize[1],args.imgsize[0]), interpolation=cv2.INTER_CUBIC)),
      #lambda img: np.zeros([args.imgsize[1], args.imgsize[0]]) if (img is None or np.min(img.shape) <= 1) else cv2.resize(img, (args.imgsize[1],args.imgsize[0]), interpolation=cv2.INTER_CUBIC)
      lambda img: cv2.resize(img, (args.imgsize[1],args.imgsize[0]), interpolation=cv2.INTER_CUBIC),#(img, (32,128), interpolation=cv2.INTER_CUBIC),
    ])

    # instantiate datasets
    iam = IAM(args.dataroot, transform=transform_train)
    eydigits = EyDigitStrings(args.dataroot, transform=transform_train)
    printed = PRT(args.dataroot,transform=transform_train) # yike todo

    irs = IRS(args.dataroot,transform=transform_train) #yike todo


    # concatenate datasets
    concat = ConcatDataset([iam, eydigits,irs,printed]) # concatenate the multiple datasets
    idxTrain = int( .9 * len(concat) )
    trainset, testset = random_split(concat, [idxTrain, len(concat)-idxTrain])
    trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=2)

    # save characters of model for inference mode
    charlist = list(set.union(set(iam.charList),set(eydigits.charList),set(irs.charList),set(printed.charList)))
    open(join(args.ckptpath, 'charList.txt'), 'w').write(str().join(charlist))

    # # save words contained in dataset into file
    # open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))

    # execute training or validation
    if args.train:
      model = Model(args, charlist, decoderType)
      train(model, trainloader, testloader)
    elif args.validate:
      model = Model(args, charlist, decoderType, mustRestore=True)
      validate(model, testloader)

  # infer text on test image
  else:
    print(open(join(args.ckptpath, 'accuracy.txt')).read())
    model = Model(args, open(join(args.ckptpath, 'charList.txt')).read(), decoderType, mustRestore=True)
    infer(model, FilePaths.fnInfer)

def train(model, loader, testloader=None):
  "train NN"
  epoch = 0  # number of training epochs since start
  bestCharErrorRate = bestWordErrorRate = float('inf')  # best valdiation character error rate
  while True:
    epoch += 1; print('Epoch:', epoch, ' Training...')

    # train
    counter = 0
    step = 0
    for idx, (images, labels) in enumerate(loader):

      # convert torchtensor to numpy
      images = images.numpy()

      # train batch
      try:
        loss = model.trainBatch(images, labels)
      except:
        print(labels)
      step += 1

      # save training status
      if np.mod(idx,110)==0:
        print('TRAIN: Batch:', idx/len(loader), 'Loss:', loss)
        experiment.log_metric('train/loss', loss, step)

      # log images
      if epoch==1 and counter<5:
        text = labels[counter]
        utils.log_image(experiment, images[counter], text, 'train', args.ckptpath, counter, epoch)
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

    # log best metrics
    if charErrorRate < bestCharErrorRate: # if best validation accuracy so far, save model parameters
      print('Character error rate improved, save model')
      bestCharErrorRate = charErrorRate
      noImprovementSince = 0
      model.save(epoch)
      open(join(args.ckptpath, 'accuracy.txt'), 'w').write(
        'Validation character error rate of saved model: %f%%' % (charErrorRate * 100.0))
    else:
      print('Character error rate not improved')
      noImprovementSince += 1
    if 1-wordAccuracy < bestWordErrorRate:
      bestWordErrorRate = 1-wordAccuracy
    experiment.log_metric('best/cer', bestCharErrorRate, step)
    experiment.log_metric('best/wer', bestWordErrorRate, step)

    # stop training
    if epoch>=args.epochEnd: print('Done with training at epoch', epoch, 'sigoptObservation='+str(bestCharErrorRate)); break


def validate(model, loader, epoch, is_testing=False):
  "validate NN"
  if not is_testing: print('Validating NN')
  else: print('Testing NN')
  loader.validationSet()
  numCharErr, numCharTotal, numWordOK, numWordTotal = 0, 0, 0, 0
  plt.figure(figsize=(6,2))
  counter = 0
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

      if is_testing and epoch==args.epochEnd:
        text = ' '.join(['[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"'])
        utils.log_image(experiment, batch.imgs[i], text, 'test-'+('ok' if dist==0 else 'err'), args.ckptpath, counter, epoch)
        counter += 1

    if epoch==1 and counter<5 and not is_testing: # log images
      text = ' '.join(['[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"'])
      utils.log_image(experiment, batch.imgs[i], text, 'valid', args.ckptpath, counter, epoch)
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


if __name__ == '__main__':
  main()
