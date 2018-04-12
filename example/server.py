#!/usr/bin/env python

import numpy as np
import os, time
from memoire import ReplayMemory, ReplayMemoryServer, Bind, Conn
from threading import Thread

sizes = (1,1,1,0,1)
rem = ReplayMemory(*sizes, max_episode=64, episode_max_length=1024)
rem.discount_factor = 0.0
rem.priority_exponent = 0.0
rem.td_lambda = 1.0
rem.frame_stack = 4
rem.multi_step = 4
rem.print_info()
batch_size = 4

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
    time.sleep(5)
    try:
      prev_e, next_e, info = rem.get_batch(batch_size)
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
    rem.update_weight(batch_size, epi_idx, entry_idx, epi_inc, entry_weight)
except KeyboardInterrupt:
  os.kill(os.getpid(), 9)

