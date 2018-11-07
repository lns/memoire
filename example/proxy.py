#!/usr/bin/env python

import numpy as np
import os, time
from memoire import Proxy, Bind, Conn
from threading import Thread

proxy = Proxy()

threads = []
# Please note that the queue length should not be too short (e.g. 1), in which case
# subscribe/unsubscribe messages may lost, and cause an assertion failed in ZeroMQ
# "Assertion failed: erased == 1 (src/mtrie.cpp:297)"
# See also https://github.com/zeromq/libzmq/issues/2942
threads.append(Thread(target=proxy.pub_proxy_main,
  args=("tcp://localhost:10100", Conn, "ipc:///tmp/memoire_pubsub_test", Bind, 32)))
threads.append(Thread(target=proxy.rep_proxy_main,
  args=("ipc:///tmp/memoire_reqrep_test", Bind, "tcp://localhost:10101", Conn, 32)))
threads.append(Thread(target=proxy.pull_proxy_main,
  args=("ipc:///tmp/memoire_pushpull_test", Bind, "tcp://localhost:10102", Conn, 32)))

try:
  for th in threads:
    th.start()
  while True:
    time.sleep(1)
except KeyboardInterrupt:
  pass
os.kill(os.getpid(), 9)

