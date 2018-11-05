#pragma once

#include "zmq.h"
#include "qlog.hpp"
#include <sys/types.h>
#include <unistd.h>
#include <netinet/in.h> 
#include <arpa/inet.h>
#include <string>

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
    qlog_warning("[ERRNO %d] '%s' (rc: %d). Abort.\n", e, zmq_strerror(e), rc); \
		throw std::runtime_error("Critical error."); \
  } \
} while(0)
#endif

int64_t check_multipart(void * soc) {
  int64_t more = 0;
  size_t more_size = sizeof(more);
  ZMQ_CALL(zmq_getsockopt(soc, ZMQ_RCVMORE, &more, &more_size));
  return more;
}

/**
 * Get host ip
 * @param ref_ip   reference ip for connection (e.g. "8.8.8.8")
 * @param ref_port reference port for connection (e.g. 53 for DNS)
 *
 * See https://stackoverflow.com/a/3120382
 */
std::string get_host_ip(const char* ref_ip, uint16_t ref_port) 
{
  std::string ret(16, '\0');

  int sock = socket(AF_INET, SOCK_DGRAM, 0);
  assert(sock != -1);

  struct sockaddr_in serv;
  memset(&serv, 0, sizeof(serv));
  serv.sin_family = AF_INET;
  serv.sin_addr.s_addr = inet_addr(ref_ip);
  serv.sin_port = htons(ref_port);

  int err = connect(sock, (const sockaddr*) &serv, sizeof(serv));
  assert(err != -1);

  sockaddr_in name;
  socklen_t namelen = sizeof(name);
  err = getsockname(sock, (sockaddr*) &name, &namelen);
  assert(err != -1);

  const char* p = inet_ntop(AF_INET, &name.sin_addr, &ret[0], ret.size());
  assert(p);

  close(sock);
  return ret;
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

