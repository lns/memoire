#!/usr/bin/env python

import numpy as np
import os
import time
from memoire import ReplayMemory, ReplayMemoryServer, ReplayMemoryClient, Bind, Conn

#pub_ep = "epgm://eth1;224.0.2.1:12345"
pub_ep = "tcp://localhost:25560"
print(pub_ep)
client = ReplayMemoryClient(pub_ep, "", "")

try:
  print(client.sub_bytes("M01"))
  print('A: ' + str(time.time()))
  print(client.sub_bytes("M02"))
  print('B: ' + str(time.time()))
  print(client.sub_bytes("M01"))
  print('C: ' + str(time.time()))
  print(client.sub_bytes("M02"))
  print('D: ' + str(time.time()))
except KeyboardInterrupt:
  pass
os.kill(os.getpid(), 9)

