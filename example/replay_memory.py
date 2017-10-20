#!/usr/bin/env python

import numpy as np
from memoire import ReplayMemory
from multiprocessing import Process

sizes = (12,1,1,0,1)
rem = ReplayMemory(*sizes, max_episode=64)
rem.discount_factor = 1.0
rem.print_info()

#rem.close_episode(100)

s = np.ndarray((sizes[0]), dtype=np.float32)
a = np.ndarray((sizes[1]), dtype=np.float32)
r = np.ndarray((sizes[2]), dtype=np.float32)

# This will fail as the replay memory is currently empty
#prev_e, next_e = rem.get_batch(4, True, 1)

for n_games in range(100):
  epi_idx = rem.new_episode()
  for step in range(1000):
    s.fill(epi_idx)
    a[0] = step
    r[0] = 1
    rem.add_entry(epi_idx, s, a, r)
  rem.close_episode(epi_idx)

prev_e, next_e = rem.get_batch(4, True, 1)
for each in prev_e:
  print each
for each in next_e:
  print each

rem.clear()
# This will fail as the rem is cleared
#prev_e, next_e = rem.get_batch(4, True, 1)

for i in range(1000):
  epi_idx = rem.new_episode()

