#!/usr/bin/env python

import numpy as np
import os, time
from memoire import Proxy, Bind, Conn
from threading import Thread

proxy = Proxy()

threads = []
threads.append(Thread(target=proxy.rep_proxy_main,
  args=("ipc:///tmp/memoire_reqrep_test", Bind, "tcp://localhost:10101", Conn, 1)))
threads.append(Thread(target=proxy.pull_proxy_main,
  args=("ipc:///tmp/memoire_pushpull_test", Bind, "tcp://localhost:10102", Conn, 1)))

try:
  for th in threads:
    th.start()
  while True:
    time.sleep(1)
except KeyboardInterrupt:
  pass
os.kill(os.getpid(), 9)

