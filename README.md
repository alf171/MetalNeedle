# MetalNeedle
Inspired by taking the course 10-714, I am going to be porting needle to use metal. We will also bench mark with a bunch of different architectures like LSTMs, RNNs, CNNs, Transformers (vs Pytorch).

### Goals
  - [ ] recreate needle
    - just basic operations for now ig
  - [ ] start to incorporate metal stuff 
  - [ ] benchmark performance (RNNs / LTSMs, CNNs, Transformers, Alexnet?)
  - [ ] compiler stuff?
  - [ ] multi device stuff (https://colossalai.org/docs/concepts/paradigms_of_parallelism/)

## prereqs
  - setup poetry and install dependencies
  - currently use numpy, pytorch (for correctness and benchmarking), and pybind11 (for C++ support)

## Run instructions 
  - ./scripts/c.sh to setup pybind stuff
  - run ./scripts/setup.sh allows python to find Needle package
  - run ./scripts/uts.sh or /scripts/benchmark.sh

## TODO 
  - [ ] support autograd with computation graphs
  - [ ] support metal or other gpu hardware
  - [ ] train first AI model (wavenet maybe?)

## Data layout
Tensor -> TensorData -> rawTensor (C++ tensor) -> data

## Operations
Tensor -> TensorOperations -> TensorData -> C++ operations on data