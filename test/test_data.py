#!/usr/bin/env python
from memoire import ReplayMemoryServer, ReplayMemoryClient, Bind, Conn
import numpy as np
import time, os
from threading import Thread

s = np.ndarray([], dtype=np.float32)
r = np.ndarray([], dtype=np.float32)
p = np.ndarray([], dtype=np.float32)
v = np.ndarray([], dtype=np.float32)

entry = (s,r,p,v)

def client_main(n_episode=1, first_len=3, last_len=3):
  from copy import deepcopy
  client = ReplayMemoryClient("client_test")
  client.req_endpoint = "tcp://localhost:10101"
  client.push_endpoint = "tcp://localhost:10102"
  client.get_info()

  for j in range(n_episode):
    rollout = []
    for i in range(10*j+0,10*j+first_len):
      entry[0].fill(i)
      entry[1].fill(1)
      entry[2].fill(0)
      entry[3].fill(-1)
      rollout.append(deepcopy(entry))
    #print(rollout)
    client.push_data(rollout, False)
    time.sleep(1)
    rollout = []
    for i in range(10*j+first_len,10*j+(first_len+last_len)):
      entry[0].fill(i)
      entry[1].fill(1)
      entry[2].fill(0)
      entry[3].fill(-1)
      rollout.append(deepcopy(entry))
    #print(rollout)
    client.push_data(rollout, True)
    time.sleep(1)
  client.close()

def test_01():
  server = ReplayMemoryServer(entry, 4, 64)
  server.rem.rollout_len = 4
  server.rem.max_episode = 0
  batch_size = 1
  #
  threads = []
  threads.append(Thread(target=server.rep_worker_main,  args=("tcp://*:10101", Bind)))
  threads.append(Thread(target=server.pull_worker_main, args=("tcp://*:10102", Bind)))
  threads.append(Thread(target=client_main, args=(1,3,3)))

  for th in threads:
    th.start()

  data, weight = server.get_data(batch_size)
  assert len(data) == len(entry) + 1
  assert len(weight) == batch_size
  s = data[0]
  assert len(s) == batch_size
  assert len(s[0]) == server.rem.rollout_len
  traj = s[0]
  assert traj[0] == 2
  assert traj[1] == 3
  assert traj[2] == 4
  assert traj[3] == 5
  time.sleep(1)
  server.close() # Prevent auto-deletion
  time.sleep(1)

def test_02():
  server = ReplayMemoryServer(entry, 4, 64)
  server.rem.rollout_len = 4
  server.rem.max_episode = 0
  batch_size = 1
  #
  threads = []
  threads.append(Thread(target=server.rep_worker_main,  args=("tcp://*:10101", Bind)))
  threads.append(Thread(target=server.pull_worker_main, args=("tcp://*:10102", Bind)))
  threads.append(Thread(target=client_main, args=(1,6,0)))

  for th in threads:
    th.start()

  data, weight = server.get_data(batch_size)
  assert len(data) == len(entry) + 1
  assert len(weight) == batch_size
  s = data[0]
  assert len(s) == batch_size
  assert len(s[0]) == server.rem.rollout_len
  traj = s[0]
  assert traj[0] == 2
  assert traj[1] == 3
  assert traj[2] == 4
  assert traj[3] == 5
  time.sleep(1)
  server.close() # Prevent auto-deletion
  time.sleep(1)

def test_03():
  # Test do_padding
  server = ReplayMemoryServer(entry, 4, 64)
  server.rem.rollout_len = 4
  server.rem.max_episode = 0
  server.rem.do_padding = True
  batch_size = 1
  #
  threads = []
  threads.append(Thread(target=server.rep_worker_main,  args=("tcp://*:10101", Bind)))
  threads.append(Thread(target=server.pull_worker_main, args=("tcp://*:10102", Bind)))
  threads.append(Thread(target=client_main, args=(1,3,0)))

  for th in threads:
    th.start()

  data, weight = server.get_data(batch_size)
  assert len(data) == len(entry) + 1
  assert len(weight) == batch_size
  s = data[0]
  assert len(s) == batch_size
  assert len(s[0]) == server.rem.rollout_len
  traj = s[0]
  assert traj[0] == 0
  assert traj[1] == 0
  #assert traj[2] == 0
  #assert traj[3] == 0
  time.sleep(1)
  server.close() # Prevent auto-deletion
  time.sleep(1)

def test_04():
  # Test not_first
  server = ReplayMemoryServer(entry, 4, 64)
  server.rem.rollout_len = 4
  server.rem.max_episode = 0
  server.rem.do_padding = False
  batch_size = 1
  #
  threads = []
  threads.append(Thread(target=server.rep_worker_main,  args=("tcp://*:10101", Bind)))
  threads.append(Thread(target=server.pull_worker_main, args=("tcp://*:10102", Bind)))
  threads.append(Thread(target=client_main, args=(1,3,0)))

  for th in threads:
    th.start()

  def term():
    time.sleep(3)
    server.close()
  Thread(target=term).start()

  data, weight = server.get_data(batch_size)
  assert False

def test_05():
  # Test not_first
  server = ReplayMemoryServer(entry, 4, 64)
  server.rem.rollout_len = 4
  server.rem.max_episode = 0
  server.rem.do_padding = False
  batch_size = 1
  #
  threads = []
  threads.append(Thread(target=server.rep_worker_main,  args=("tcp://*:10101", Bind)))
  threads.append(Thread(target=server.pull_worker_main, args=("tcp://*:10102", Bind)))
  threads.append(Thread(target=client_main, args=(2,3,0)))

  for th in threads:
    th.start()

  data, weight = server.get_data(batch_size)
  assert len(data) == len(entry) + 1
  assert len(weight) == batch_size
  s = data[0]
  assert len(s) == batch_size
  assert len(s[0]) == server.rem.rollout_len
  traj = s[0]
  print(traj)
  assert traj[0] == 2
  assert traj[1] == 10
  assert traj[2] == 11
  assert traj[3] == 12
  time.sleep(1)
  server.close() # Prevent auto-deletion
  time.sleep(1)

def test_06():
  server = ReplayMemoryServer(entry, 8, 64)
  server.rem.rollout_len = 4
  server.rem.max_episode = 0
  server.rem.do_padding = False
  batch_size = 1
  #
  threads = []
  threads.append(Thread(target=server.rep_worker_main,  args=("tcp://*:10101", Bind)))
  threads.append(Thread(target=server.pull_worker_main, args=("tcp://*:10102", Bind)))
  threads.append(Thread(target=client_main, args=(2,2,0)))

  for th in threads:
    th.start()

  data, weight = server.get_data(batch_size)
  assert len(data) == len(entry) + 1
  assert len(weight) == batch_size
  s = data[0]
  assert len(s) == batch_size
  assert len(s[0]) == server.rem.rollout_len
  traj = s[0]
  print(traj)
  assert traj[0] == 0
  assert traj[1] == 1
  assert traj[2] == 10
  assert traj[3] == 11
  time.sleep(1)
  server.close() # Prevent auto-deletion
  time.sleep(1)

if __name__ == '__main__':
  #test_01()
  test_02()
  #test_03()
  #test_04()
  #test_05()
  #test_06()

