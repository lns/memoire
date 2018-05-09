# Memoire
A Distributed Replay Memory

**Memoire** is a distributed replay memory for reinforcement learning. Industrial application of reinforcement learning usually requires large amount of computation, both for environemnt exploration and neural network training. Our goal is to make it easier to write high-performance distributed reinforcement learning algorithm.

## How it works
![DistRL](/docs/imgs/DistRL.png "Framework of Distributed RL")
The distributed reinforcement learning platform consists of two types of workers: **Actors** and **Learners**.

An actor is responsible for exploring the environment and generating data for learners. In its mainloop, it works as
1. Get latest model from learners.
2. Run current model in the environment.
3. Put generated experience in the client side of replay memory.
4. Client push samples to the server.

An learner is responsible for updating the model(policy) with batch data. In its mainloop, it works as
1. Get batch of samples from the server side of replay memory.
2. Update model with batch samples, according to different algorithms.
3. Publish latest model to actors.

In summary, we distribute actors and learners over several machines (CPU and GPU) to fully utilize heterogeneous computing resources.

|      | Actor | Learner |
|:----:|:-----:|:-------:|
|Computing resource| CPU | GPU |
|DNN operation | Forward | F/B |
|Numbers | ~300 | ~10 |
|Memory usage | ~10G | ~1G |
|Bandwidth usage | ~1G | ~20G |

## Features
+ Prioritized Sampling

  Prioritized experience replay [1] is a method of selecting high-priority samples for training. It is argubly the most effective technique for good performance of (distributed) reinforcement learning [2] [3].

+ Framework Independent

  The replay memory module is seperated from the training of neural network, thus making it independent of the deep learning framework used to implement the neural network (e.g. TensorFlow, PyTorch, etc.). We hope the modulized design can provide more flexibility for deep learning practitioners.

## Build
```shell
cd build/
make
```

## Dependency
ZeroMQ, google-test, pybind11, libbfd

## Usage
See `example/`

## Documentation
(TODO) See source code

## TODO
+ Variable size of state

## Reference
[[1] T.Schaul et al. **Prioritized Experience Replay**](https://arxiv.org/abs/1511.05952)
[[2] M.Hessel et al. **Rainbow: Combining Improvements in Deep Reinforcement Learning**](https://arxiv.org/abs/1710.02298)
[[3] D.Horgan et al. **Distributed Prioritized Experience Replay**](https://arxiv.org/abs/1803.00933)
