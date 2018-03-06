#!/usr/bin/env python

import numpy as np
import os
import time
from memoire import ReplayMemory, ReplayMemoryServer, Bind, Conn
from threading import Thread

sizes = (12,1,1,0,1)
rem = ReplayMemory(*sizes, max_episode=8, episode_max_length=1024)
#rem.discount_factor = 1.0 # not used
rem.pre_skip = 0
rem.post_skip = 1
rem.print_info()

server = ReplayMemoryServer(rem)

threads = []
threads.append(Thread(target=server.pull_proxy_main, args=("tcp://*:5562", "inproc://pull_workers")))
threads.append(Thread(target=server.rep_worker_main, args=("tcp://*:5561", Bind)))
for i in range(4):
  threads.append(Thread(target=server.pull_worker_main, args=("inproc://pull_workers", Conn)))

for th in threads:
  th.start()

try:
  while True:
    time.sleep(3)
    try:
      prev_e, next_e, info = rem.get_batch(1)
    except RuntimeError: # get_batch() failed
      continue
    for each in prev_e:
      print each
    for each in next_e:
      print each
    for each in info:
      print each
    epi_idx, entry_idx, epi_inc, entry_weight = info
    print epi_idx, entry_idx, epi_inc, entry_weight
    entry_weight.fill(0)
    rem.update_weight(1, epi_idx, entry_idx, epi_inc, entry_weight)
  #time.sleep(86400 * 365)
except KeyboardInterrupt:
  os.kill(os.getpid(), 9)

