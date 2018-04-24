#!/usr/bin/env python

import numpy as np
import os, time
from memoire import ReplayMemory, ReplayMemoryServer, Bind, Conn
from threading import Thread

sizes = (1,1,1,0,1)
rem = ReplayMemory(*sizes, capacity=1)
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

server = ReplayMemoryServer(rem, n_caches=0)

threads = []
threads.append(Thread(target=server.pull_proxy_main, args=("tcp://*:5562", "tcp://*:5563")))
threads.append(Thread(target=server.rep_worker_main, args=("tcp://*:5561", Bind)))

for th in threads:
  th.start()

try:
  time.sleep(86400 * 365)
except KeyboardInterrupt:
  os.kill(os.getpid(), 9)

