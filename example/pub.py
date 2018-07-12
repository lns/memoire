#!/usr/bin/env python

import numpy as np
import os, time
from memoire import ReplayMemory, ReplayMemoryServer, Bind, Conn
from threading import Thread

s = np.ndarray([1], dtype=np.float32)
a = np.ndarray([1], dtype=np.float32)
r = np.ndarray([1], dtype=np.float32)
p = np.ndarray([], dtype=np.float32)
v = np.ndarray([1], dtype=np.float32)
q = np.ndarray([1], dtype=np.float32)
i = np.ndarray([1], dtype=np.float32)
template = (s,a,r,p,v,q,i)
server = ReplayMemoryServer(*template, max_step=0, n_caches=0, pub_endpoint="tcp://*:5560")

try:
  time.sleep(1)
  server.pub_bytes("M01", "Hello 1")
  time.sleep(1)
  server.pub_bytes("M02", "Hello 2" + ' '*512 + '%')
  time.sleep(1)
  server.pub_bytes("M01", "Hello 3")
  time.sleep(1)
  server.pub_bytes("M02", "Hello 4" )
except KeyboardInterrupt:
  pass
os.kill(os.getpid(), 9)

