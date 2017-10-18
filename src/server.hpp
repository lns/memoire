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
  std::vector<std::thread> thread;
  RM * prm;
  void * ctx;

  ReplayMemoryServer(RM * p_rem,
    void * context) : prm{p_rem}, ctx{context}
  {
    prm->print_info();
  }

  //@param mode: MODE_BIND or MODE_CONN

  void start_rep_worker(const char * endpoint, int n, int mode) {
    if(mode==MODE_BIND)
      assert(n==1);
    for(int i=0;i<n;i++)
      thread.emplace_back(RM::rep_thread_main, prm, ctx, endpoint, mode);
  }

  void start_pull_worker(const char * endpoint, int n, int mode) {
    if(mode==MODE_BIND)
      assert(n==1);
    for(int i=0;i<n;i++)
      thread.emplace_back(RM::pull_thread_main, prm, ctx, endpoint, mode);
  }

  void start_rep_proxy(const char * front_ep, const char * back_ep) {
    thread.emplace_back(RM::proxy_main, ctx, ZMQ_ROUTER, front_ep, ZMQ_DEALER, back_ep);
  }

  void start_pull_proxy(const char * front_ep, const char * back_ep) {
    thread.emplace_back(RM::proxy_main, ctx, ZMQ_PULL, front_ep, ZMQ_PUSH, back_ep);
  }

  void join_all() {
    for(auto&& each : thread)
      each.join();
  }

};

