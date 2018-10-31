#pragma once

#include "utils.hpp"
#include "qlog.hpp"
#include <stack>

class ZMQBase : public non_copyable {
private:
  // ZeroMQ socket is not thread-safe, so a socket should only
  // be used by a single thread at the same time.
  void * ctx;
  std::stack<void*> opened_soc;
  std::mutex ctx_mutex;

public:
  /**
   * Initialization
   */
  ZMQBase() : ctx{nullptr}
  {
    std::lock_guard<std::mutex> guard(ctx_mutex);
    ctx = zmq_ctx_new(); qassert(ctx);
  }

  ~ZMQBase() {
    if(ctx)
      qlog_warning("Destruction called without calling close() before.\n");
    close();
  }

  void close() {
    std::lock_guard<std::mutex> guard(ctx_mutex);
    while(not opened_soc.empty()) {
      qlog_debug("Close socket %p in ctx %p\n", opened_soc.top(), ctx);
      ZMQ_CALL(zmq_close(opened_soc.top()));
      opened_soc.pop();
    }
    if(ctx) {
      qlog_info("Terminating context. This may hang due to unclosed socket or unsent messages.\n");
      ZMQ_CALL(zmq_ctx_term(ctx));
      ctx = nullptr;
      qlog_debug("Closed.\n");
    }
  }

  void * new_zmq_socket(int type) {
    std::lock_guard<std::mutex> guard(ctx_mutex);
    void * soc = zmq_socket(ctx, type);
    ZMQ_CALL((soc == nullptr ? -1 : 0));
    qlog_debug("New socket %p in ctx %p\n", soc, ctx);
    opened_soc.push(soc);
    // setsockopt
    int linger = 0; // linger time in milliseconds
    ZMQ_CALL(zmq_setsockopt(soc, ZMQ_LINGER, &linger, sizeof(linger)));
    return soc;
  }

};

