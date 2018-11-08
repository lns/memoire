#!/usr/bin/env python
from memoire import ReplayMemoryServer, ReplayMemoryClient, Bind, Conn
import numpy as np
import pickle, time
from copy import deepcopy

use_proxy = False

client = ReplayMemoryClient("client_test")
if use_proxy:
  client.req_endpoint = "ipc:///tmp/memoire_reqrep_test"
  client.push_endpoint = "ipc:///tmp/memoire_pushpull_test"
else:
  client.req_endpoint = "tcp://localhost:10101"
  client.push_endpoint = "tcp://localhost:10102"

client.get_info()

s = np.ndarray([2,2], dtype=np.float32)
r = np.ndarray([], dtype=np.float32)
p = np.ndarray([], dtype=np.float32)
v = np.ndarray([], dtype=np.float32)

entry = (s,r,p,v)

for j in range(1):
  rollout = []
  for i in range(10*j+0,10*j+6):
    entry[0].fill(i)
    entry[1].fill(1)
    entry[2].fill(0)
    entry[3].fill(-1)
    rollout.append(deepcopy(entry))
  print(rollout)
  client.push_data(rollout, True)
  time.sleep(1)
client.close()
