# MetalNeedle
Inspired by taking the course 10-714, I am going to be porting needle to use metal. We will also bench mark with a bunch of different architectures like LSTMs, RNNs, CNNs, Transformers (vs Pytorch).

### Goals
  - [ ] recreate needle (maybe I will ref code I wrote)
    - just basic operations for now ig
  - [ ] start to incorporate metal stuff 
  - [ ] benchmark performance (RNNs / LTSMs, CNNs, Transformers, Alexnet?)
  - [ ] compiler stuff?


## Run instructions 
  - ./scripts/c.sh to setup pybind stuff
  - run ./scripts/setup.sh allows python to find Needle package
  - run ./scripts/uts.sh or /scripts/benchmark.sh

## TODO 
  - [ ] improve matmul
  - [ ] benchmark performance and tweak
  - [ ] more operations + AD (Automatic differentiation)
