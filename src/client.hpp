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
  uint32_t entry_size;
  std::vector<BufView> view;

  std::string req_endpoint;
  std::string push_endpoint;
  std::string client_uuid;

  uint32_t start_step;

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
    : remote_slot_index{~0u},
      req_endpoint{req_ep},
      push_endpoint{push_ep},
      client_uuid{input_uuid},
      start_step{0}
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
    if(not soc) {
      qassert(soc = zmq_socket(ctx, ZMQ_REQ));
      ZMQ_CALL(zmq_connect(soc, req_endpoint.c_str()));
      reqbuf.resize(1024, '\0');
      repbuf.resize(1024, '\0'); // TODO(qing): adjust default size
    }
    proto::Msg req;
    req.set_type(proto::REQ_GET_INFO);
    req.set_version(req.version());
    req.set_sender(client_uuid);
    req.SerializeToString(&reqbuf);
    do {
      int size;
      ZMQ_CALL(zmq_send(soc, reqbuf.data(), reqbuf.size(), 0));
      ZMQ_CALL(size = zmq_recv(soc, repbuf.data(), repbuf.size(), 0));
      if(not (size <= (int)repbuf.size())) { // resize and wait for next
        repbuf.resize(size);
        qlog_warning("Resize repbuf to %lu. Resending.. \n", repbuf.size());
        continue;
      }
    } while(false);
    proto::Msg rep;
    rep.ParseFromString(repbuf);
    qassert(rep.version() == req.version());
    qassert(rep.type() == proto::REP_GET_INFO);
    // Get info
    proto::RepGetInfo& info = rep.rep_get_info();
    x_descr_pickle = info.x_descr_pickle();
    remote_slot_index = info.slot_index();
    entry_size = info.entry_size();
    qassert(info.view_size() == N_VIEW);
    view.resize(info.view_size());
    for(int i=0; i<info.view_size(); i++)
      view[i].from_pb(&info.view(i));
  }

  void push_data(void * data, uint32_t n_step, bool is_episode_end) {
    thread_local void * soc = nullptr;
    thread_local std::string pushbuf;
    if(not soc) {
      qassert(soc = zmq_socket(ctx, ZMQ_PUSH));
      ZMQ_CALL(zmq_connect(soc, push_endpoint.c_str()));
      pushbuf.resize(1024, '\0');
    }
    proto::Msg push;
    push.set_type(proto::PUSH_DATA);
    push.set_version(push.version()); // TODO(qing): check this
    push.set_sender(client_uuid);
    auto * d = push.mutable_push_data();
    d->set_is_episode_end(is_episode_end);
    d->set_start_step(start_step);
    d->set_n_step(n_step);
    d->set_slot_index(remote_slot_index);
    d->set_data(data, n_step * entry_size);
    push.SerializeToString(&pushbuf);
    do {
      ZMQ_CALL(zmq_send(soc, pushbuf.data(), pushbuf.size(), 0));
    } while(false);
    if(is_episode_end)
      start_step = 0;
    else
      start_step += n_step;
  }

};

typedef ReplayMemoryClient<RM> RMC;

