#pragma once

#include "utils.hpp"
#include <vector>
#include <thread>
#include <cstdlib>
#include <functional>
#include "zmq_base.hpp"

class Proxy : public ZMQBase {
public:

  Proxy() {}

  ~Proxy() {}

  /**
   * Responsible for connecting multiple frontends and backends
   */
  void proxy_main(
      int front_soc_type,
      const char * front_endpoint,
      typename RM::Mode front_mode,
      int front_hwm,
      int back_soc_type,
      const char * back_endpoint,
      typename RM::Mode back_mode,
      int back_hwm)
  {
    void * front = new_zmq_socket(front_soc_type);
    void * back  = new_zmq_socket( back_soc_type);
    ZMQ_CALL(zmq_setsockopt(front, ZMQ_SNDHWM, &front_hwm, sizeof(front_hwm)));
    ZMQ_CALL(zmq_setsockopt(front, ZMQ_RCVHWM, &front_hwm, sizeof(front_hwm)));
    ZMQ_CALL(zmq_setsockopt(back,  ZMQ_SNDHWM, &back_hwm,  sizeof(back_hwm)));
    ZMQ_CALL(zmq_setsockopt(back,  ZMQ_RCVHWM, &back_hwm,  sizeof(back_hwm)));
    if(front_mode == RM::Bind)
      ZMQ_CALL(zmq_bind(front, front_endpoint));
    else if(front_mode == RM::Conn)
      ZMQ_CALL(zmq_connect(front, front_endpoint));
    if(back_mode == RM::Bind)
      ZMQ_CALL(zmq_bind(back, back_endpoint));
    else if(back_mode == RM::Conn)
      ZMQ_CALL(zmq_connect(back, back_endpoint));
    ZMQ_CALL(zmq_proxy(front, back, nullptr));
  }

  void rep_proxy_main(const char * front_ep, typename RM::Mode front_mode,
      const char * back_ep, typename RM::Mode back_mode, int hwm) {
    proxy_main(ZMQ_ROUTER, front_ep, front_mode, hwm, ZMQ_DEALER, back_ep, back_mode, hwm);
  }

  void pull_proxy_main(const char * front_ep, typename RM::Mode front_mode,
      const char * back_ep, typename RM::Mode back_mode, int hwm) {
    proxy_main(ZMQ_PULL, front_ep, front_mode, hwm, ZMQ_PUSH, back_ep, back_mode, hwm);
  }

  void pub_proxy_main(const char * front_ep, typename RM::Mode front_mode,
      const char * back_ep, typename RM::Mode back_mode, int hwm) {
    proxy_main(ZMQ_XSUB, front_ep, front_mode, hwm, ZMQ_XPUB, back_ep, back_mode, hwm);
  }

};

