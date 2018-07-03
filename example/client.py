#!/usr/bin/env python

import numpy as np
import os
import time
from memoire import ReplayMemory, ReplayMemoryServer, ReplayMemoryClient, Bind, Conn
from threading import Thread

client = ReplayMemoryClient("tcp://localhost:5560", "tcp://localhost:5561", "tcp://localhost:5562")
client.sync_sizes(65536)
rem = client.rem
rem.print_info()

s = np.ndarray((rem.state_size), dtype=np.uint8)
a = np.ndarray((rem.action_size), dtype=np.float32)
r = np.ndarray((rem.reward_size), dtype=np.float32)
p = np.ndarray((rem.prob_size), dtype=np.float32)
v = np.ndarray((rem.value_size), dtype=np.float32)

time.sleep(1)
try:
  for n_games in range(10):
    rem.new_episode()
    for step in range(1000):
      s.fill(n_games)
      a[0] = step
      r[0] = 1
      v[0] = -1
      rem.add_entry(s, a, r, p, v, weight=1.0)
    rem.close_episode()
    client.update_counter()
    assert 0 == client.push_cache()
except KeyboardInterrupt:
  pass
os.kill(os.getpid(), 9)

