#!/usr/bin/env python

from memoire import ReplayMemoryServer, ReplayMemoryClient, Bind, Conn
import numpy as np
import time, os
from threading import Thread

s = np.ndarray([2,2], dtype=np.float32)
r = np.ndarray([], dtype=np.float32)
p = np.ndarray([], dtype=np.float32)
v = np.ndarray([], dtype=np.float32)

entry = (s,r,p,v)

server = ReplayMemoryServer(entry, 4, 64)
server.rem.rollout_len = 4
server.rem.max_episode = 0
server.print_info()
server.set_logfile("test.log", "w")

try:
  batch_size = 1
  #
  threads = []
  threads.append(Thread(target=server.rep_worker_main,  args=("tcp://*:10101", Bind)))
  threads.append(Thread(target=server.pull_worker_main, args=("tcp://*:10102", Bind)))
  for th in threads:
    th.start()
  while True:
    time.sleep(1)
except KeyboardInterrupt:
  pass
os.kill(os.getpid(), 9)

