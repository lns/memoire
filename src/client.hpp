#pragma once

#include "utils.hpp"
#include "replay_memory.hpp"
#include "qlog.hpp"
#include <future>

template<class RM>
class ReplayMemoryClient : public non_copyable {
public:
  typedef typename RM::Message Message;
  typedef typename RM::DataCache Cache;

  qlib::LCG64 lcg64;
  RM * prm;

protected:
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
  Cache * cache_buf;

public:
  ReplayMemoryClient(
      const char * req_endpoint,
      const char * push_endpoint,
      size_t rem_capacity) : prm(nullptr)
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
    sync_sizes(rem_capacity);
  }

  ~ReplayMemoryClient() {
    if(prm)
      delete prm;
    if(cache_buf)
      free(cache_buf);
    zmq_close(rrsoc);
    zmq_close(ppsoc);
    free(reqbuf);
    free(repbuf);
    free(pushbuf);
    zmq_ctx_destroy(ctx);
  }

  void sync_sizes(size_t capa) {
    if(prm) {
      delete prm;
      prm = nullptr;
    }
    if(cache_buf) {
      free(cache_buf);
      cache_buf = nullptr;
    }
    req->type = Message::ProtocalSizes;
    req->sender = 0;
    ZMQ_CALL(zmq_send(rrsoc, reqbuf, reqbuf_size, 0));
    ZMQ_CALL(zmq_recv(rrsoc, repbuf, repbuf_size, 0));
    qassert(rep->type == req->type);
    // Get sizes
    RM * p = reinterpret_cast<RM*>(&rep->payload);
    prm = new RM{p->state_size, p->action_size, p->reward_size, p->prob_size, p->value_size,
      capa, &lcg64};
    // Sync parameters
    prm->discount_factor = p->discount_factor;
    prm->priority_exponent = p->priority_exponent;
    prm->td_lambda = p->td_lambda;
    prm->frame_stack = p->frame_stack;
    prm->multi_step  = p->multi_step;
    prm->cache_size  = p->cache_size;
    std::copy(std::begin(p->rwd_coeff), std::end(p->rwd_coeff), std::begin(prm->rwd_coeff));
    std::copy(std::begin(p->cache_flags), std::end(p->cache_flags), std::begin(prm->cache_flags));
    // Make cache buf
    cache_buf = (Cache*)malloc(Cache::nbytes(prm));
  }

  /**
   * Should be called after closing an episode
   */
  void update_counter() {
    req->type = Message::ProtocalCounter;
    req->sender = prm->uuid;
    int * p_length = reinterpret_cast<int *>(&req->payload);
    *p_length = prm->new_length;
    //qlog_info("%s(): length: %d\n",__func__, *p_length);
    ZMQ_CALL(zmq_send(rrsoc, reqbuf, reqbuf_size, 0));
    ZMQ_CALL(zmq_recv(rrsoc, repbuf, repbuf_size, 0));
  }

  void push_cache() {
    bool ret = prm->get_cache(cache_buf, push->sum_weight);
    if(not ret) // failed
      return;
    //cache_buf->get(prm->cache_size-1, prm).print_first(prm); // DEBUG
    push->type = Message::ProtocalCache;
    push->length = Cache::nbytes(prm);
    push->sender = prm->uuid;
    //qlog_info("%s(): cache_size: %d, cache::nbytes: %lu\n", __func__, prm->cache_size, Cache::nbytes(prm));
    ZMQ_CALL(zmq_send(ppsoc, pushbuf, pushbuf_size, ZMQ_SNDMORE));
    ZMQ_CALL(zmq_send(ppsoc, cache_buf, push->length, 0));
  }

};

