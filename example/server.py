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

try:
  batch_size = 1
  #
  threads = []
  threads.append(Thread(target=server.rep_worker_main,  args=("tcp://*:10101", Bind)))
  threads.append(Thread(target=server.pull_worker_main, args=("tcp://*:10102", Bind)))
  #threads.append(Thread(target=server.rep_worker_main,  args=("ipc:///tmp/memoire_reqrep_test", Bind)))
  #threads.append(Thread(target=server.pull_worker_main, args=("ipc:///tmp/memoire_pushpull_test", Bind)))
  for th in threads:
    th.start()
  while True:
    data, weight = server.get_data(batch_size)
    print(data)
    print(weight)
    time.sleep(1)
except KeyboardInterrupt:
  pass
os.kill(os.getpid(), 9)

