#!/usr/bin/env python

import numpy as np
import os
import time
from memoire import ReplayMemory, ReplayMemoryServer, ReplayMemoryClient, Bind, Conn

client = ReplayMemoryClient("tcp://localhost:5560", "", "")

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

