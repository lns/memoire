#!/usr/bin/env python

from memoire import ReplayMemoryServer, ReplayMemoryClient, Bind, Conn
import numpy as np
import time, os
from threading import Thread

r = np.ndarray([], dtype=np.float32)
p = np.ndarray([], dtype=np.float32)
v = np.ndarray([], dtype=np.float32)

entry = (r,p,v)

server = ReplayMemoryServer(entry, 0, 0)
server.pub_endpoint = "tcp://*:10100"

try:
  index = 0
  while True:
    server.pub_bytes("topic", "message:%d" % index)
    index += 1
    time.sleep(1)
except KeyboardInterrupt:
  pass
os.kill(os.getpid(), 9)

