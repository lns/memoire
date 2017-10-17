#pragma once

#include "zmq.h"

#ifndef MIN
#define MIN(a,b) ((a)>(b)?(b):(a))
#endif

#ifndef MAX
#define MAX(a,b) ((a)<(b)?(b):(a))
#endif

#ifndef DIVUP
#define DIVUP(x,y) (((x)+(y)-1)/(y))
#endif

class non_copyable
{
protected:
  non_copyable() = default;
  ~non_copyable() = default;

  non_copyable(non_copyable const &) = delete;
  void operator=(non_copyable const &x) = delete;
};

#define ZMQ_CALL(x) do { \
  int rc = (x); \
  if(-1==rc) { \
    int e = zmq_errno(); \
    fprintf(stderr, "In %s(), %s:%d\n",__func__,__FILE__,__LINE__); \
    fprintf(stderr, "[%d]'%s' (rc: %d)\n", e, zmq_strerror(e), rc); \
  } \
} while(0)

