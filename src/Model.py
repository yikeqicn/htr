import numpy as np
import sys
import tensorflow as tf
from os.path import join
from Densenet4htr import Densenet4htr
import utils

class DecoderType:
  BestPath = 0
  BeamSearch = 1
  WordBeamSearch = 2

class Model:
  "minimalistic TF model for HTR"

  # model constants
  batchSize = 50
  # imgSize = (128, 32)
  imgSize = (192, 48)
  maxTextLen = 32

  def __init__(self, args, charList, decoderType=DecoderType.BestPath, mustRestore=False, FilePaths=None):
    "init model: add CNN, RNN and CTC and initialize TF"
    self.charList = charList
    self.decoderType = decoderType
    self.mustRestore = mustRestore
    self.snapID = 0
    self.FilePaths = FilePaths
    self.batchsize = args.batchsize
    self.lrInit = args.lrInit
    self.args = args


    # Input
    self.inputImgs = tf.placeholder(tf.float32, shape=(self.batchsize, args.imgsize[0], args.imgsize[1]))

    # CNN
    if args.nondensenet:
      cnnOut4d = self.setupCNN(self.inputImgs)
    else: # use densenet by default
      cnnOut4d = self.setupCNNdensenet(self.inputImgs, args)

    # RNN
    rnnOut3d = self.setupRNN(cnnOut4d)

    # CTC
    (self.ctcloss, self.decoder) = self.setupCTC(rnnOut3d)

    # Explicit regularizers
    self.loss = self.ctcloss + args.wdec * self.setupWdec(args)

    # optimizer for NN parameters
    self.batchesTrained = args.batchesTrained
    self.learningRate = tf.placeholder(tf.float32, shape=[])
    if args.optimizer=='rmsprop':
      self.optimizer = tf.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)
    elif args.optimizer=='adam':
      self.optimizer = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss)
    elif args.optimizer=='momentum':
      self.optimizer = tf.train.MomentumOptimizer(self.learningRate, .9).minimize(self.loss)

    # initialize TF
    (self.sess, self.saver) = self.setupTF()

  def setupCNN(self, cnnIn3d):
    "vanilla cnn from original github repo"
    cnnIn4d = tf.expand_dims(input=cnnIn3d, axis=3)

    # list of parameters for the layers
    kernelVals = [5, 5, 3, 3, 3]
    featureVals = [1, 32, 64, 128, 128, 256]
    strideVals = poolVals = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2)]
    numLayers = len(strideVals)

    # create layers
    pool = cnnIn4d  # input to first CNN layer
    for i in range(numLayers):
      kernel = tf.Variable(
        tf.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i], featureVals[i + 1]], stddev=0.1))
      conv = tf.nn.conv2d(pool, kernel, padding='SAME', strides=(1, 1, 1, 1))
      relu = tf.nn.relu(conv)
      pool = tf.nn.max_pool(relu, (1, poolVals[i][0], poolVals[i][1], 1), (1, strideVals[i][0], strideVals[i][1], 1),
                            'VALID')

    self.is_training = tf.placeholder(tf.bool, shape=[]) # dummy placeholder to prevent error, no effect
    return pool

  def setupCNNdensenet(self, cnnIn3d, args):
    "ADDED BY RONNY: densenet cnn"
    cnnIn4d = tf.expand_dims(input=cnnIn3d, axis=3)
    net = Densenet4htr(cnnIn4d, **vars(args))
    self.is_training = net.is_training
    print('shape of cnn output: '+str(net.output.get_shape().as_list()))
    return net.output

  def setupRNN(self, rnnIn4d):
    "create RNN layers and return output of these layers"
    rnnIn3d = tf.squeeze(rnnIn4d, axis=[2])

    # basic cells which is used to build RNN
    numHidden = self.args.rnndim
    cells = [tf.contrib.rnn.LSTMCell(num_units=numHidden, state_is_tuple=True) for _ in range(2)]  # 2 layers

    # stack basic cells
    stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

    # bidirectional RNN
    # BxTxF -> BxTx2H
    ((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnnIn3d,
                                                    dtype=rnnIn3d.dtype)

    # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
    concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)

    # project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
    kernel = tf.Variable(tf.truncated_normal([1, 1, numHidden * 2, len(self.charList) + 1], stddev=0.1))
    logits = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])
    # with tf.variable_scope('logits'):
    #   logits = tf.squeeze(tf.layers.conv2d(concat, len(self.charList)+1, 1, use_bias=True), axis=[2]) # FIXED BY RONNY
    return logits

  def setupCTC(self, ctcIn3d):
    "create CTC loss and decoder and return them"
    # BxTxC -> TxBxC
    ctcIn3dTBC = tf.transpose(ctcIn3d, [1, 0, 2])
    # ground truth text as sparse tensor
    self.gtTexts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]),
                                   tf.placeholder(tf.int32, [None]),
                                   tf.placeholder(tf.int64, [2]))
    # calc loss for batch
    self.seqLen = tf.placeholder(tf.int32, [None])
    loss = tf.nn.ctc_loss(labels=self.gtTexts, inputs=ctcIn3dTBC, sequence_length=self.seqLen, ctc_merge_repeated=True)

    # decoder: either best path decoding or beam search decoding
    if self.decoderType == DecoderType.BestPath:
      decoder = tf.nn.ctc_greedy_decoder(inputs=ctcIn3dTBC, sequence_length=self.seqLen)
    elif self.decoderType == DecoderType.BeamSearch:
      decoder = tf.nn.ctc_beam_search_decoder(inputs=ctcIn3dTBC, sequence_length=self.seqLen, beam_width=50,
                                              merge_repeated=False)
    elif self.decoderType == DecoderType.WordBeamSearch:
      # import compiled word beam search operation (see https://github.com/githubharald/CTCWordBeamSearch)
      word_beam_search_module = tf.load_op_library('TFWordBeamSearch.so')

      # prepare information about language (dictionary, characters in dataset, characters forming words)
      chars = str().join(self.charList)
      wordChars = open('wordCharList.txt').read().splitlines()[0]
      corpus = open(self.FilePaths.fnCorpus).read()

      # decode using the "Words" mode of word beam search
      decoder = word_beam_search_module.word_beam_search(tf.nn.softmax(ctcIn3dTBC, dim=2), 50, 'Words', 0.0,
                                                         corpus.encode('utf8'), chars.encode('utf8'),
                                                         wordChars.encode('utf8'))

    # return a CTC operation to compute the loss and a CTC operation to decode the RNN output
    return (tf.reduce_mean(loss), decoder)

  def setupWdec(self, args):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables(): # all weights count toward weight decay except batchnorm and biases
      if var.op.name.find(r'BatchNorm') == -1 & var.op.name.find(r'bias:0')==-1:
        costs.append(tf.nn.l2_loss(var))
    return tf.add_n(costs)

  def setupTF(self):
    "initialize TF"
    print('Python: ' + sys.version)
    print('Tensorflow: ' + tf.__version__)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))

    saver = tf.train.Saver(max_to_keep=1)  # saver saves model to file
    modelDir = self.FilePaths.fnCkptpath
    latestSnapshot = tf.train.latest_checkpoint(modelDir)  # is there a saved model?

    # if model must be restored (for inference), there must be a snapshot
    if self.mustRestore and not latestSnapshot:
      raise Exception('No saved model found in: ' + modelDir)

    # load saved model if available
    if latestSnapshot:
      print('Init with stored values from ' + latestSnapshot)
      saver.restore(sess, latestSnapshot)
    else:
      print('Ran global_variables_initializer')
      sess.run(tf.global_variables_initializer())

    if self.FilePaths.urlTransferFrom!=None: # ADDED BY RONNY initialize params from other model (transfer learning)

      utils.maybe_download(source_url=self.FilePaths.urlTransferFrom,
                           filename=self.FilePaths.fnCkptpath,
                           target_directory=None,
                           filetype='folder',
                           force=True)
      saverTransfer = tf.train.Saver(tf.trainable_variables()[:-1])  # load all variables except from logit (classification) layer
      latestSnapshot = tf.train.latest_checkpoint(self.FilePaths.fnCkptpath)  # is there a saved model?
      if not latestSnapshot: raise Exception('No TransferFrom saved model in '+self.FilePaths.urlTransferFrom)
      print('Loaded variable values (except logit layer) from ' + latestSnapshot)
      saverTransfer.restore(sess, latestSnapshot)

    return (sess, saver)

  def toSparse(self, texts):
    "put ground truth texts into sparse tensor for ctc_loss"
    indices = []
    values = []
    shape = [len(texts), 0]  # last entry must be max(labelList[i])

    # go over all texts
    for (batchElement, text) in enumerate(texts):
      # convert to string of label (i.e. class-ids)
      labelStr = [self.charList.index(c) for c in text]
      # sparse tensor must have size of max. label-string
      if len(labelStr) > shape[1]:
        shape[1] = len(labelStr)
      # put each label into sparse tensor
      for (i, label) in enumerate(labelStr):
        indices.append([batchElement, i])
        values.append(label)

    return (indices, values, shape)

  def decoderOutputToText(self, ctcOutput):
    "extract texts from output of CTC decoder"

    # contains string of labels for each batch element
    encodedLabelStrs = [[] for i in range(self.batchsize)]

    # word beam search: label strings terminated by blank
    if self.decoderType == DecoderType.WordBeamSearch:
      blank = len(self.charList)
      for b in range(self.batchsize):
        for label in ctcOutput[b]:
          if label == blank:
            break
          encodedLabelStrs[b].append(label)

    # TF decoders: label strings are contained in sparse tensor
    else:
      # ctc returns tuple, first element is SparseTensor
      decoded = ctcOutput[0][0]

      # go over all indices and save mapping: batch -> values
      idxDict = {b: [] for b in range(self.batchsize)}
      for (idx, idx2d) in enumerate(decoded.indices):
        label = decoded.values[idx]
        batchElement = idx2d[0]  # index according to [b,t]
        encodedLabelStrs[batchElement].append(label)

    # map labels to chars for all batch elements
    return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]

  def trainBatch(self, batch):
    "feed a batch into the NN to train it"
    sparse = self.toSparse(batch.gtTexts)
    lrnrate = self.lrInit if self.batchesTrained < self.args.lrDrop1 else (
      self.lrInit*1e-1 if self.batchesTrained < self.args.lrDrop2 else self.lrInit*1e-2)  # decay learning rate
    (_, lossVal) = self.sess.run([self.optimizer, self.loss], {self.inputImgs: batch.imgs,
                                                                  self.gtTexts: sparse,
                                                                  self.seqLen: [Model.maxTextLen] * self.batchsize,
                                                                  self.learningRate: lrnrate,
                                                                  self.is_training: True})
    self.batchesTrained += 1
    return lossVal

  def inferBatch(self, batch):
    "feed a batch into the NN to recngnize the texts"
    decoded = self.sess.run(self.decoder,
                            {self.inputImgs: batch.imgs, self.seqLen: [Model.maxTextLen] * self.batchsize, self.is_training: False})
    return self.decoderOutputToText(decoded)

  def save(self):
    "save model to file"
    self.snapID += 1
    self.saver.save(self.sess, join(self.FilePaths.fnCkptpath, 'model'), global_step=self.snapID)
