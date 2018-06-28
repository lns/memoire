#!/usr/bin/env python

import numpy as np
import os
import time
from memoire import ReplayMemory, ReplayMemoryServer, ReplayMemoryClient, Bind, Conn

client = ReplayMemoryClient("tcp://localhost:5560", "tcp://localhost:5561", "", 0)
client.rem.print_info()

try:
  print(client.sub_bytes("M01"))
except KeyboardInterrupt:
  pass
os.kill(os.getpid(), 9)

