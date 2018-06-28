#!/usr/bin/env python

import numpy as np
import os, time
from memoire import ReplayMemory, ReplayMemoryServer, Bind, Conn
from threading import Thread

sizes = (1,1,1,0,1)
server = ReplayMemoryServer(*sizes, max_step=0, n_caches=0, pub_endpoint="tcp://*:5560")
server.rem.print_info()

threads = []
threads.append(Thread(target=server.rep_worker_main, args=("tcp://*:5561", Bind)))

for th in threads:
  th.start()

try:
  time.sleep(3)
  server.pub_bytes("M01", "Hello!")
except KeyboardInterrupt:
  pass
os.kill(os.getpid(), 9)

