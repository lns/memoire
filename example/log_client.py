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

try:
  increment = 0
  while True:
    client.push_log("uuid: %s, message: %d\n" % (client.uuid, increment))
    increment += 1
    time.sleep(1)
except KeyboardInterrupt:
  pass
client.close()

