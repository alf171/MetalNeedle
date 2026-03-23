#!/bin/bash

source ./scripts/setup.sh
MNIST_BATCH_SIZE=${MNIST_BATCH_SIZE:-128} \
MNIST_EPOCHS=${MNIST_EPOCHS:-1} \
MNIST_MAX_BATCHES=${MNIST_MAX_BATCHES:-50} \
python3.14 -m tst.mnist
