#pragma once

#include "utils.hpp"
#include <vector>
#include <thread>
#include <cstdlib>
#include <functional>
#include "replay_memory.hpp"

template<class RM>
class ReplayMemoryServer : public non_copyable {
public:
  RM * prm;
  void * ctx;

  ReplayMemoryServer(RM * p_rem) : prm{p_rem}
  {
    ctx = zmq_ctx_new(); qassert(ctx);
  }

  ~ReplayMemoryServer() { zmq_ctx_destroy(ctx); }

  void rep_worker_main(const char * endpoint, typename RM::Mode mode) {
    RM::rep_thread_main(prm, ctx, endpoint, mode);
  }

  void pull_worker_main(const char * endpoint, typename RM::Mode mode) {
    RM::pull_thread_main(prm, ctx, endpoint, mode);
  }

  void rep_proxy_main(const char * front_ep, const char * back_ep) {
    RM::proxy_main(ctx, ZMQ_ROUTER, front_ep, ZMQ_DEALER, back_ep);
  }

  void pull_proxy_main(const char * front_ep, const char * back_ep) {
    RM::proxy_main(ctx, ZMQ_PULL, front_ep, ZMQ_PUSH, back_ep);
  }

};

