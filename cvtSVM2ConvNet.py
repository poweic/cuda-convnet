#!/usr/bin/python

import os
import math
import pickle
import numpy as np
import sys

import argparse
parser = argparse.ArgumentParser()

def read_svm_data(filename):

  def svm_read_problem(filename):
	  """
	  Thanks to febeling !! (please visit https://github.com/febeling/libsvm/tree/master/python )
	  """
	  prob_y = []
	  prob_x = []
	  for line in open(filename) if filename != "-" else sys.stdin:
		  line = line.split(None, 1)
		  # In case an instance with all zero features
		  if len(line) == 1: line += ['']
		  label, features = line
		  xi = {}
		  for e in features.split():
			  ind, val = e.split(":")
			  xi[int(ind)] = int(val)
			  # xi[int(ind)] = float(val)
		  prob_y += [int(label)]
		  prob_x += [xi]
	  return (prob_y, prob_x)

  y, x = svm_read_problem(filename)

  # Change 1-based index to 0-based index
  # If label in y starts from 1 instead of 0, it will CAUSE severe CRASH when training convnet.
  if not (min(y) == 1 or min(y) == 0):
    raise AssertionError("\33[31mLabels must either be 0-based or 1-based array\33[0m")

  y = [i-1 for i in y] if min(y) == 1 else y

  def find_max_dimension(data):
    dim = max(data[0].keys())
    for i in range(len(data)):
      if len(data[i].keys()) == 0:
	continue

      d = max(data[i].keys())
      if d > dim:
	dim = d

    return dim

  N = len(x)
  dim = find_max_dimension(x)
  X = np.zeros((dim, N), dtype=np.uint8)

  for i in range(len(x)):
    for k, v in x[i].iteritems():
      X[k-1][i] = v

  return y, X


def make_batch_meta(filename, y, X, batchSize):

  '''
  batches.meta.keys() = 
    num_cases_per_batch
    label_names
    num_vis
    data_mean
  '''

  meta = {}
  meta['label_names']	      = [str(i) for i in list(set(y))]
  meta['num_cases_per_batch'] = batchSize
  meta['num_vis']	      = X.shape[0]
  meta['data_mean']	      = np.mean(X, axis=1, keepdims=1, dtype=np.float32)

  pickle.dump(meta, open(filename, 'wb'))

def make_batches(folder, y, X, batchSize):

  '''
  data_batch_1.keys() = 
  batch_label
  labels
  data
  filenames
  '''

  nBatchs = int(math.ceil(len(y) / batchSize))

  for b in range(nBatchs):
    istart = b*batchSize
    iend   = (b+1)*batchSize

    batch = {}
    batch['batch_label'] = 'batch ' + `b` + ' of ' + `nBatchs`
    batch['labels']	 = y[istart:iend]
    batch['data']	 = X[:, istart:iend]
    batch['filenames']	 = [`i` + '.png' for i in list(range(istart, iend))]

    filename = folder + '/data_batch_' + `b`
    pickle.dump(batch, open(filename, 'wb'))

def cvtSVM2CudaConvNet(filename, nBatch, output_dir):
  y, X = read_svm_data(filename)

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  batchSize = int(math.ceil(len(y) / nBatch))
  make_batch_meta(output_dir + '/batches.meta', y, X, batchSize)
  make_batches(output_dir, y, X, batchSize)



parser.add_argument("svm_in", help="the filename of input data in LibSVM format")
parser.add_argument("convnet-out-dir", help="the directory of output data in CUDA ConvNet format")
parser.add_argument("--batches", help="# of batches", type=int, default=10)
args = vars(parser.parse_args())

cvtSVM2CudaConvNet(args['svm_in'], args['batches'], args['convnet-out-dir'])
