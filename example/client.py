#!/usr/bin/env python

import numpy as np
import os
import time
from memoire import ReplayMemory, ReplayMemoryServer, ReplayMemoryClient, Bind, Conn
from threading import Thread

client = ReplayMemoryClient("tcp://localhost:5561", "tcp://localhost:5562", 1)
rem = client.prm
rem.print_info()
rem.discount_factor = 0.0

s = np.ndarray((rem.state_size), dtype=np.float32)
a = np.ndarray((rem.action_size), dtype=np.float32)
r = np.ndarray((rem.reward_size), dtype=np.float32)

for n_games in range(100):
  epi_idx = rem.new_episode()
  for step in range(2):
    s.fill(n_games)
    a[0] = step
    r[0] = 1
    rem.add_entry(epi_idx, s, a, r, weight=1.0)
  rem.close_episode(epi_idx)
  client.push_episode_to_remote(epi_idx)

