#pragma once

#include "utils.hpp"
#include "replay_memory.hpp"
#include "qlog.hpp"
#include "msg.pb.h"

template<class RM>
class ReplayMemoryClient : public non_copyable {
public:
  std::string x_descr_pickle;
  uint32_t remote_slot_index;
  std::vector<BufView> view;

  std::string req_endpoint;
  std::string push_endpoint;
  std::string client_uuid;

protected:
  // ZeroMQ socket is not thread-safe, so a socket should only
  // be used by a single thread at the same time.
  void * ctx;

public:
  /**
   * Initialization
   *
   * endpoint can be set to nullptr or "" to disable this protocal
   */
  ReplayMemoryClient(
      const std::string req_ep,
      const std::string push_ep,
      const std::string input_uuid)
    : remote_slot_index(~0u),
      req_endpoint(req_ep),
      push_endpoint(push_ep),
      client_uuid(input_uuid)
  {
    ctx = zmq_ctx_new(); qassert(ctx);
  }

  ~ReplayMemoryClient() {
    zmq_ctx_destroy(ctx);
  }

  void get_info() {
    thread_local void * soc = nullptr;
    thread_local std::string reqbuf;
    thread_local std::string repbuf;
    THREAD_LOCAL_TIMER;
    if(not soc) {
      qassert(soc = zmq_socket(ctx, ZMQ_REQ));
      ZMQ_CALL(zmq_connect(soc, req_endpoint.c_str()));
      reqbuf.resize(1024, '\0');
      repbuf.resize(1024, '\0');
    }
    proto::Msg req;
    req.set_type(proto::REQ_GET_INFO);
    req.set_version(req.version());
    req.set_sender(client_uuid);
    req.SerializeToString(&reqbuf);
    do {
      START_TIMER();
      int size;
      ZMQ_CALL(zmq_send(soc, reqbuf.data(), reqbuf.size(), 0));
      ZMQ_CALL(size = zmq_recv(soc, repbuf.data(), repbuf.size(), 0));
      STOP_TIMER();
      if(not (size <= (int)repbuf.size())) { // resize and wait for next
        repbuf.resize(size);
        qlog_warning("Resize repbuf to %lu. Resending.. \n", repbuf.size());
        continue;
      }
    } while(false);
    PRINT_TIMER_STATS(100);
    proto::Msg rep;
    rep.ParseFromString(repbuf);
    qassert(rep.version() == req.version());
    qassert(rep.type() == proto::REP_GET_INFO);
    // Get info
    proto::RepGetInfo& info = rep.rep_get_info();
    x_descr_pickle = info.x_descr_pickle();
    remote_slot_index = info.slot_index();
    qassert(info.view_size() == N_VIEW);
    view.resize(info.view_size());
    for(int i=0; i<info.view_size(); i++)
      view[i].from_pb(&info.view(i));
  }

};

typedef ReplayMemoryClient<RM> RMC;

