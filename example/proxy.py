#!/usr/bin/env python

import numpy as np
import os, time
from memoire import ReplayMemory, ReplayMemoryServer, Bind, Conn
from threading import Thread

# Not used
s = a = r = p = v = q = i = np.ndarray([], dtype=np.float32)
template = (s,a,r,p,v,q,i)
server = ReplayMemoryServer(*template, max_step=0, n_caches=0, pub_endpoint="")
rem = server.rem

front_ep = 'tcp://localhost:5560'
back_ep  = 'tcp://*:25560'

threads = []
threads.append(Thread(target=server.pub_proxy_main,  args=(front_ep, Conn, back_ep, Bind)))

print("Starting proxy '%s' -> '%s'" % (front_ep, back_ep))

for th in threads:
  th.start()

try:
  th.join()
except KeyboardInterrupt:
  pass
os.kill(os.getpid(), 9)

