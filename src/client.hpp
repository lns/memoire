#pragma once

#include "utils.hpp"
#include "replay_memory.hpp"
#include "qlog.hpp"
#include <future>

template<class RM>
class ReplayMemoryClient : public non_copyable {
public:
  typedef typename RM::Message Message;

  qlib::LCG64 lcg64;
  RM * prm;
  void * ctx;
  void * rrsoc; // for request-reply
  void * ppsoc; // for push-pull
  int reqbuf_size;
  char * reqbuf;
  int repbuf_size;
  char * repbuf;
  int pushbuf_size;
  char * pushbuf;
  Message *req, *rep, *push;

  ReplayMemoryClient(
      const char * req_endpoint,
      const char * push_endpoint,
      size_t rem_max_capacity) : prm(nullptr)
  {
    ctx = zmq_ctx_new(); qassert(ctx);
    rrsoc = zmq_socket(ctx, ZMQ_REQ); qassert(rrsoc);
    ZMQ_CALL(zmq_connect(rrsoc, req_endpoint));
    ppsoc = zmq_socket(ctx, ZMQ_PUSH); qassert(ppsoc);
    ZMQ_CALL(zmq_connect(ppsoc, push_endpoint));  
    // Make bufs
    reqbuf_size  = RM::reqbuf_size();
    reqbuf = (char*)malloc(reqbuf_size);
    req  = reinterpret_cast<Message*>(reqbuf);
    repbuf_size  = RM::repbuf_size();
    repbuf = (char*)malloc(repbuf_size);
    rep  = reinterpret_cast<Message*>(repbuf);
    pushbuf_size = RM::pushbuf_size();
    pushbuf = (char*)malloc(pushbuf_size);
    push = reinterpret_cast<Message*>(pushbuf);
    // Get sizes
    sync_sizes(rem_max_capacity);
  }

  ~ReplayMemoryClient() {
    zmq_close(rrsoc);
    zmq_close(ppsoc);
    free(reqbuf);
    free(repbuf);
    free(pushbuf);
    delete prm;
    zmq_ctx_destroy(ctx);
  }

  void push_episode_to_remote(int epi_idx) {
    push->type = Message::AddEpisode;
    push->length = prm->episode[epi_idx].size();
    ZMQ_CALL(zmq_send(ppsoc, pushbuf, pushbuf_size, ZMQ_SNDMORE));
    size_t send_size = prm->entry_size * push->length;
    ZMQ_CALL(zmq_send(ppsoc, prm->episode[epi_idx].data.data(), send_size, ZMQ_SNDMORE));
    ZMQ_CALL(zmq_send(ppsoc, prm->episode[epi_idx].prt.w_.data(), 2*prm->episode[epi_idx].prt.size_*sizeof(float), 0));
  }

protected:
  void sync_sizes(size_t max_capacity) {
    if(prm)
      delete prm;
    req->type = Message::GetSizes;
    ZMQ_CALL(zmq_send(rrsoc, reqbuf, reqbuf_size, 0));
    ZMQ_CALL(zmq_recv(rrsoc, repbuf, repbuf_size, 0));
    qassert(rep->type == Message::Success);
    // Get sizes
    size_t * p = reinterpret_cast<size_t*>(&rep->entry);
    prm = new RM{p[0], p[1], p[2], p[3], p[4], max_capacity, p[5], &lcg64};
    prm->pre_skip = p[6];
    prm->post_skip = p[7];
  }

};

