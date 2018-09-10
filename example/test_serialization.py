#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import memoire
import pickle
#import cPickle as pickle
import time

def check_eq(a, b):
  if isinstance(a, list) or isinstance(a, tuple):
    if not (isinstance(b, list) or isinstance(b, tuple)):
      return False
    if len(a) != len(b):
      return False
    eq = [check_eq(a[i], b[i]) for i in range(len(a))]
    return all(eq)
  else:
    test = (a==b) | (np.isnan(a) & np.isnan(b))
    if isinstance(test, bool):
      return test
    else:
      return test.all()

def descr_serial_test(a):
  start = time.time()
  d = memoire.get_descr(a)
  s = memoire.descr_serialize(a,d)
  b = memoire.descr_unserialize(s,d)
  print("descr:  %.6f, size: %d" % (time.time() - start, len(s)))
  assert check_eq(a,b)

def pickle_serial_test(a):
  start = time.time()
  s = pickle.dumps(a)
  b = pickle.loads(s)
  print("pickle: %.6f, size: %d" % (time.time() - start, len(s)))
  assert check_eq(a,b)

def serial_test(a):
  descr_serial_test(a)
  pickle_serial_test(a)

def descr_serial_test_1(shape, batch_size):
  d = memoire.get_descr(np.random.random(shape))
  l = [np.random.random(shape) for i in range(batch_size)]
  s = [memoire.descr_serialize(each, d) for each in l]
  t = np.concatenate(s)
  a = np.concatenate(l).reshape((batch_size,) + tuple(shape))
  b = memoire.descr_unserialize_1(t, d, batch_size)
  assert check_eq(a,b)

def descr_serial_test_2(shape, batch_size, frame_stack):
  d = memoire.get_descr(np.random.random(shape))
  l = [np.random.random(shape) for i in range(batch_size * frame_stack)]
  s = [memoire.descr_serialize(each, d) for each in l]
  t = np.concatenate(s)
  a = np.concatenate(l).reshape((batch_size,frame_stack) + tuple(shape))
  b = memoire.descr_unserialize_2(t, d, batch_size, frame_stack)
  assert check_eq(a,b)

def main_test():
  a = np.ndarray([84,84], dtype=np.uint8)
  b = np.ndarray([84,84], dtype=np.float64)
  c = np.ndarray([84,84], dtype=np.int32)

  serial_test( a )
  serial_test( (a,) )
  serial_test( (a,(b,(b,),c)) )
  serial_test( (a,(b,c),[]) )
  serial_test( (c,[([b,]),a],) )
  serial_test( (c,[([b,],[]),],a) )
  serial_test( ([],[],) )
  serial_test( ([([(a)])]) )

if __name__ == '__main__':
  main_test()
  descr_serial_test_1([2,3], 4)
  descr_serial_test_2([2,3], 4, 3)

