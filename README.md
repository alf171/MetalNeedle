# NEEDLEMetal
Inspired by taking the course 10-714, I am going to be porting needle to use metal. We will also bench mark with a bunch of different architectures like LSTMs, RNNs, CNNs, Transformers (vs Pytorch).

Goals
  - [] recreate needle (maybe I will ref code I wrote)
    - just basic operations for now ig
  - [] start to incorporate metal stuff 
  - [] benchmark performance 
  - [] compiler stuff?


## Run instructions 
  - downloading metal (import torch; torch.backends.mps.is_available())