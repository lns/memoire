#!/usr/bin/env python

import numpy as np
import os, time
from memoire import ReplayMemory, ReplayMemoryServer, Bind, Conn
from threading import Thread

sizes = (1,1,1,0,1)
rem = ReplayMemory(*sizes, capacity=0) # capacity not used
rem.discount_factor = 0.0
rem.priority_exponent = 0.0
rem.td_lambda = 1.0
rem.frame_stack = 4
rem.multi_step = 4
rem.cache_size = 4
rem.print_info()
print(rem.rwd_coeff)
print(rem.cache_flags)
rem.cache_flags = [1,1,1,0,1, 1,1,1,0,1]
batch_size = 4

server = ReplayMemoryServer(rem, n_caches=16)

threads = []
for i in range(4):
  threads.append(Thread(target=server.pull_worker_main, args=("tcp://localhost:5563", Conn)))

for th in threads:
  th.start()

try:
  while True:
    time.sleep(5)
    try:
      prev_e, next_e, info = server.get_batch(batch_size)
    except RuntimeError: # get_batch() failed
      continue
    for each in prev_e:
      print each
    for each in next_e:
      print each
    for each in info:
      print each
    entry_weight, = info
except KeyboardInterrupt:
  os.kill(os.getpid(), 9)

