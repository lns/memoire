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
  } \
} while(0)
#endif

int64_t check_multipart(void * soc) {
  int64_t more = 0;
  size_t more_size = sizeof(more);
  ZMQ_CALL(zmq_getsockopt(soc, ZMQ_RCVMORE, &more, &more_size));
  return more;
}

#ifdef TIME_CONNECTION
#define THREAD_LOCAL_TIMER thread_local qlib::Timer timer
#define START_TIMER() timer.start()
#define STOP_TIMER() timer.stop()
#define PRINT_TIMER_STATS(N) if(timer.cnt() % N == 0) { \
  qlog_info("%s(): n: %lu, min: %f, avg: %f, max: %f (msec)\n", \
      __func__, timer.cnt(), timer.min(), timer.avg(), timer.max()); \
  timer.clear(); \
} while(0)
#else
#define THREAD_LOCAL_TIMER
#define START_TIMER()
#define STOP_TIMER()
#define PRINT_TIMER_STATS(N)
#endif

