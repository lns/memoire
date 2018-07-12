#!/usr/bin/env python

import numpy as np
import os, time
from memoire import ReplayMemory, ReplayMemoryServer, Bind, Conn
from threading import Thread

# Here we config the state_size, action_size, reward_size, prob_size, value_size, qvest_size, info_size in the server
s = np.ndarray([2], dtype=np.uint64)
a = np.ndarray([1], dtype=np.int32)
r = np.ndarray([], dtype=np.float32)
p = np.ndarray([], dtype=np.float32)
v = np.ndarray([], dtype=np.float32)
q = np.ndarray([], dtype=np.float32)
i = np.ndarray([1], dtype=np.uint8)
template = (s,a,r,p,v,q,i)
server = ReplayMemoryServer(*template, max_step=0, n_caches=4, pub_endpoint="tcp://*:5560")
rem = server.rem

rem.priority_exponent = 0.0
rem.mix_lambda = 1.0
rem.frame_stack = 4
rem.multi_step = 2
rem.cache_size = 16
rem.discount_factor = [0.0]
rem.reward_coeff = [1.0]
rem.cache_flags = [1,1,1,0,1,1,1, 1,1,1,0,1,1,1]
server.print_info()
batch_size = 4

threads = []
threads.append(Thread(target=server.rep_worker_main, args=("tcp://*:5561", Bind)))
threads.append(Thread(target=server.pull_proxy_main, args=("tcp://*:5562", "inproc://pull_workers")))
for i in range(2): # NOTE: number of worker should be less than n_caches
   threads.append(Thread(target=server.pull_worker_main, args=("inproc://pull_workers", Conn)))

for th in threads:
  th.start()

try:
  while True:
    time.sleep(5)
    try:
      prev_e, next_e, weight = server.get_batch(batch_size)
      print("get_batch(%d) done." % batch_size)
    except RuntimeError: # get_batch() failed
      continue
    print(prev_e)
    print(next_e)
    print(weight)
except KeyboardInterrupt:
  pass
os.kill(os.getpid(), 9)

