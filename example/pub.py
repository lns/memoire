#!/usr/bin/env python

import numpy as np
import os, time
from memoire import ReplayMemory, ReplayMemoryServer, Bind, Conn
from threading import Thread

sizes = (1,1,1,0,1,1,1)
server = ReplayMemoryServer(*sizes, max_step=0, n_caches=0, pub_endpoint="tcp://*:5560")

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

