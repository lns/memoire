#!/usr/bin/env python

import numpy as np
import os
import time
from memoire import ReplayMemory, ReplayMemoryServer, Bind, Conn
from threading import Thread

sizes = (12,1,1,0,1)
rem = ReplayMemory(*sizes, max_episode=0)
rem.discount_factor = 1.0
rem.print_info()

server = ReplayMemoryServer(rem)

threads = []
threads.append(Thread(target=server.pull_proxy_main, args=("tcp://*:5562", "tcp://*:5563")))
threads.append(Thread(target=server.rep_worker_main, args=("tcp://*:5561", Bind)))

for th in threads:
  th.start()

try:
  time.sleep(86400 * 365)
except KeyboardInterrupt:
  os.kill(os.getpid(), 9)

