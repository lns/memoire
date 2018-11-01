#!/usr/bin/env python
from memoire import ReplayMemoryServer, ReplayMemoryClient, Bind, Conn
import numpy as np
import pickle, time
from copy import deepcopy

client = ReplayMemoryClient(
    "client_test",
    "tcp://localhost:10101",
    "tcp://localhost:10102")

client.get_info()
print(pickle.loads(client.x_descr_pickle))

s = np.ndarray([2,2], dtype=np.float32)
r = np.ndarray([], dtype=np.float32)
p = np.ndarray([], dtype=np.float32)
v = np.ndarray([], dtype=np.float32)

entry = (s,r,p,v)

if False:
  rollout = []
  for i in range(12):
    entry[0].fill(i)
    entry[1].fill(1)
    entry[2].fill(0)
    entry[3].fill(-1)
    rollout.append(deepcopy(entry))
  print(rollout)
  client.push_data(rollout, False)

for j in range(4):
  rollout = []
  for i in range(3*j+0,3*j+3):
    entry[0].fill(i)
    entry[1].fill(1)
    entry[2].fill(0)
    entry[3].fill(-1)
    rollout.append(deepcopy(entry))
  print(rollout)
  client.push_data(rollout, False)
  time.sleep(5)

