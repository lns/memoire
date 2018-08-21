#!/usr/bin/env python

import numpy as np
import os
import time
from memoire import ReplayMemory, ReplayMemoryServer, ReplayMemoryClient, Bind, Conn
from threading import Thread
import socket
import struct

def aton(addr):                                                               
  return struct.unpack("I", socket.inet_aton(addr))[0]                       

def ntoa(addr):                                                               
  return socket.inet_ntoa(struct.pack("I", addr)) 

def get_ip_address():
  s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  s.connect(("10.223.133.20", 80))
  return s.getsockname()[0]

client_uuid = aton(get_ip_address())

push_time_interval = 1.0

client = ReplayMemoryClient("tcp://localhost:5560", "tcp://localhost:5561", "tcp://localhost:5562", client_uuid)
client.sync_sizes(65536)
rem = client.rem
rem.print_info()

s,a,r,p,v,i = rem.get_entry_buf();

def push_worker_main(time_interval):
  cache_count = 0
  if time_interval < 0:
    return
  while True:
    if 0 == client.push_cache():
      cache_count += 1
    else:
      pass # failed
    time.sleep(time_interval)

def sub_worker_main():
  while True:
    msg = client.sub_bytes("Test")
    assert len(msg) == 1048576

threads = []
threads.append(Thread(target = push_worker_main, args=(push_time_interval,)))

time.sleep(1)

for th in threads:
  th.start()

try:
  for game_idx in range(100000):
    rem.new_episode()
    for step in range(24):
      s.fill(game_idx*10 + step)
      a.fill(step)
      r.fill(1)
      p.fill(0)
      v.fill(0.5)
      i.fill(False)
      rem.add_entry(s,a,r,p,v,i)
    rem.close_episode()
    client.update_counter()
    client.write_log("%d" % game_idx)
    assert 0 == client.push_cache()
except KeyboardInterrupt:
  pass
os.kill(os.getpid(), 9)

