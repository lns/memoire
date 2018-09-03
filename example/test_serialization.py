#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import memoire
#import pickle
import cPickle as pickle
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
    return (a == b).all()

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

