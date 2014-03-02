#!/bin/bash

if [ ! -f data/cifar-10-py-colmajor.tar.gz ]; then
  mkdir -p data/
  cd data/ 
  wget http://www.cs.toronto.edu/~kriz/cifar-10-py-colmajor.tar.gz
  tar zxvf cifar-10-py-colmajor.tar.gz
  cd -
fi

DATA=data/cifar-10-py-colmajor/
python convnet.py --data-path=$DATA --save-path=exp/ --test-range=6 --train-range=1-5 --layer-def=example-layers/layers-19pct.cfg --layer-params=example-layers/layer-params-19pct.cfg --data-provider=cifar --test-freq=13
