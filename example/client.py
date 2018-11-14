#!/usr/bin/env python
from memoire import ReplayMemoryServer, ReplayMemoryClient, Bind, Conn
import numpy as np
import pickle, time
from copy import deepcopy
from threading import Thread

use_proxy = False

client = ReplayMemoryClient("client_test")
client.push_length = 6
if use_proxy:
  client.req_endpoint = "ipc:///tmp/memoire_reqrep_test"
  client.push_endpoint = "ipc:///tmp/memoire_pushpull_test"
else:
  client.req_endpoint = "tcp://localhost:10101"
  client.push_endpoint = "tcp://localhost:10102"

client.get_info()

Thread(target=client.push_worker_main).start()

s = np.ndarray([1], dtype=np.float32)
a = np.ndarray([1], dtype=np.float32)
r = np.ndarray([], dtype=np.float32)
p = np.ndarray([], dtype=np.float32)
v = np.ndarray([], dtype=np.float32)

entry = (s,a,r,p,v)

for j in range(1):
  for i in range(10*j+0,10*j+6):
    entry[0].fill(i)
    entry[1].fill(-i)
    entry[2].fill(1)
    entry[3].fill(0)
    entry[4].fill(-1)
    client.add_entry(entry, i==10*j+5)
  time.sleep(1)
client.close()
