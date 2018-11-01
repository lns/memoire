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
server = ReplayMemoryServer(entry, 4, 128)
server.rem.post_skip = 3
server.print_info()

threads = []
threads.append(Thread(target=server.rep_worker_main,  args=("tcp://*:10101", Bind)))
threads.append(Thread(target=server.pull_worker_main, args=("tcp://*:10102", Bind)))

for th in threads:
  th.start()

try:
  while True:
    time.sleep(1) # Catch KeyboardInterrupt
    data, weight = server.get_data(1,4)
    print(data)
    print(weight)
  server.close() # Prevent auto-deletion
except KeyboardInterrupt:
  pass
os.kill(os.getpid(), 9)

