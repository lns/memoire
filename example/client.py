#!/usr/bin/env python

import numpy as np
import os
import time
from memoire import ReplayMemory, ReplayMemoryServer, ReplayMemoryClient, Bind, Conn
from threading import Thread

client = ReplayMemoryClient("tcp://localhost:5560", "tcp://localhost:5561", "tcp://localhost:5562")
client.sync_sizes(6)
rem = client.rem
rem.print_info()

s,a,r,p,v,i = rem.get_entry_buf();

time.sleep(1)
try:
  for game_idx in xrange(10):
    rem.new_episode()
    for step in xrange(1000):
      s.fill(game_idx*10 + step)
      a.fill(step)
      r.fill(1)
      p.fill(0)
      v.fill(0.5)
      i.fill(False)
      rem.add_entry(s,a,r,p,v,i)
      if step % 4 == 3:
        assert 0 == client.push_cache()
        time.sleep(10)
    rem.close_episode()
    client.update_counter()
    client.write_log("%d" % game_idx)
    assert 0 == client.push_cache()
except KeyboardInterrupt:
  pass
os.kill(os.getpid(), 9)

