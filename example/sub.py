#!/usr/bin/env python
from memoire import ReplayMemoryServer, ReplayMemoryClient, Bind, Conn
import numpy as np
import pickle, time
from copy import deepcopy

use_proxy = False

client = ReplayMemoryClient("client_test")
if use_proxy:
  client.sub_endpoint = "ipc:///tmp/memoire_pubsub_test"
else:
  client.sub_endpoint = "tcp://localhost:10100"

try:
  while True:
    msg = client.sub_bytes("topic")
    print(msg)
except KeyboardInterrupt:
  pass
client.close()

