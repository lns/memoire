#pragma once

#include "zmq.h"
#include "qlog.hpp"

#ifndef MIN
#define MIN(a,b) ((a)>(b)?(b):(a))
#endif

#ifndef MAX
#define MAX(a,b) ((a)<(b)?(b):(a))
#endif

#ifndef DIVUP
#define DIVUP(x,y) (((x)+(y)-1)/(y))
#endif

#ifndef NON_COPYABLE
#define NON_COPYABLE
class non_copyable
{
protected:
  non_copyable() = default;
  ~non_copyable() = default;

  non_copyable(non_copyable const &) = delete;
  void operator=(non_copyable const &x) = delete;
};
#endif

#ifndef ZMQ_CALL
#define ZMQ_CALL(x) do { \
  int rc = (x); \
  if(unlikely(-1==rc)) { \
    int e = zmq_errno(); \
    qlog_error("[%d]'%s' (rc: %d)\n", e, zmq_strerror(e), rc); \
    throw std::runtime_error("ZeroMQ call failed."); \
  } \
} while(0)
#endif

