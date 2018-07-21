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

  std::string sub_endpoint;
  std::string req_endpoint;
  std::string push_endpoint;

  qlib::LCG64 lcg64;
  RM * prm;

protected:
  // ZeroMQ socket is not thread-safe, so a socket should only
  // be used by a single thread at the same time.
  void * ctx;
  Cache * cache_buf;

public:
  /**
   * Initialization
   *
   * endpoint can be set to nullptr or "" to disable this protocal
   */
  ReplayMemoryClient(
      const std::string sub_ep,
      const std::string req_ep,
      const std::string push_ep)
    : sub_endpoint(sub_ep),
      req_endpoint(req_ep),
      push_endpoint(push_ep),
      prm(nullptr),
      cache_buf(nullptr)
  {
    ctx = zmq_ctx_new(); qassert(ctx);
  }

  ~ReplayMemoryClient() {
    if(prm)
      delete prm;
    if(cache_buf)
      free(cache_buf);
    zmq_ctx_destroy(ctx);
  }

  void sync_sizes(size_t max_step) {
    if(prm) {
      delete prm;
      prm = nullptr;
    }
    if(cache_buf) {
      free(cache_buf);
      cache_buf = nullptr;
    }
    thread_local void * soc = nullptr;
    thread_local Mem reqbuf;
    thread_local Mem repbuf;
    THREAD_LOCAL_TIMER;
    if(not soc) {
      qassert(soc = zmq_socket(ctx, ZMQ_REQ));
      ZMQ_CALL(zmq_connect(soc, req_endpoint.c_str()));
      reqbuf.resize(RM::reqbuf_size());
      repbuf.resize(RM::repbuf_size());
    }
    Message * req = reinterpret_cast<Message*>(reqbuf.data());
    Message * rep = reinterpret_cast<Message*>(repbuf.data());
    req->type = Message::ProtocalSizes;
    req->sender = 0;
    START_TIMER();
    ZMQ_CALL(zmq_send(soc, reqbuf.data(), reqbuf.size(), 0));
    ZMQ_CALL(zmq_recv(soc, repbuf.data(), repbuf.size(), 0));
    STOP_TIMER();
    PRINT_TIMER_STATS(100);
    qassert(rep->type == req->type);
    // Get sizes
    auto vw = ArrayView<BufView::Data>(&rep->payload, N_VIEW);
    RM * p = reinterpret_cast<RM*>((char*)vw.data() + vw.nbytes());
    BufView view[N_VIEW];
    for(int i=0; i<N_VIEW; i++)
      vw[i].to(view[i]);
    prm = new RM{view, max_step, &lcg64};
    // Sync parameters
    prm->priority_exponent = p->priority_exponent;
    prm->mix_lambda  = p->mix_lambda;
    prm->frame_stack = p->frame_stack;
    prm->multi_step  = p->multi_step;
    prm->cache_size  = p->cache_size;
    prm->reuse_cache = p->reuse_cache;
    std::copy(std::begin(p->discount_factor), std::end(p->discount_factor), std::begin(prm->discount_factor));
    std::copy(std::begin(p->reward_coeff), std::end(p->reward_coeff), std::begin(prm->reward_coeff));
    std::copy(std::begin(p->cache_flags), std::end(p->cache_flags), std::begin(prm->cache_flags));
    // Make cache buf
    cache_buf = (Cache*)malloc(Cache::nbytes(prm));
  }

  /**
   * Blocked Receive of Bytestring
   */
  std::string sub_bytes(std::string topic) {
    thread_local void * soc = nullptr;
    thread_local Mem tpcbuf;
    thread_local Mem subbuf;
    thread_local std::string last_topic("");
    THREAD_LOCAL_TIMER;
    if(not soc) {
      qassert(soc = zmq_socket(ctx, ZMQ_SUB));
      ZMQ_CALL(zmq_connect(soc, sub_endpoint.c_str()));
      tpcbuf.resize(256);
      subbuf.resize(256);
    }
    if(topic.size() >= tpcbuf.size())
      qlog_error("topic: '%s' is too long.\n", topic.c_str());
    if(last_topic != "")
      ZMQ_CALL(zmq_setsockopt(soc, ZMQ_UNSUBSCRIBE, last_topic.c_str(), last_topic.size()));
    last_topic = topic;
    if(topic != "")
      ZMQ_CALL(zmq_setsockopt(soc, ZMQ_SUBSCRIBE, topic.c_str(), topic.size()));
    int size;
    while(true) {
      memset(tpcbuf.data(), 0, tpcbuf.size());
      memset(subbuf.data(), 0, subbuf.size());
      START_TIMER();
      // Recv topic
      ZMQ_CALL(size = zmq_recv(soc, tpcbuf.data(), tpcbuf.size(), 0)); qassert(size <= (int)tpcbuf.size());
      // Recv Message
      ZMQ_CALL(size = zmq_recv(soc, subbuf.data(), subbuf.size(), 0));
      STOP_TIMER();
      PRINT_TIMER_STATS(100);
      // Check topic
      if(strcmp((const char*)tpcbuf.data(), topic.data())) { // topic mismatch, this should not happen
        qlog_error("topic mismatch: '%s' != '%s'\n", (const char *)tpcbuf.data(), topic.c_str());
      }
      // Check msg size
      if(not (size <= (int)subbuf.size())) { // resize and wait for next
        subbuf.resize(size);
        qlog_warning("Resize subbuf to %lu\n", subbuf.size());
        continue;
      }
      else
        return std::string((char*)subbuf.data(), size);
    }
  }

  /**
   * Should be called after closing an episode
   */
  void update_counter() {
    thread_local void * soc = nullptr;
    thread_local Mem pushbuf;
    THREAD_LOCAL_TIMER;
    if(not soc) {
      qassert(soc = zmq_socket(ctx, ZMQ_PUSH));
      ZMQ_CALL(zmq_connect(soc, push_endpoint.c_str()));
      pushbuf.resize(RM::pushbuf_size());
    }
    Message * push = reinterpret_cast<Message*>(pushbuf.data());
    push->type = Message::ProtocalCounter;
    push->length = sizeof(int);
    push->sender = prm->uuid;
    push->sum_weight = 0.0;
    int * p_length = reinterpret_cast<int *>(&push->payload);
    *p_length = prm->new_length;
    START_TIMER();
    ZMQ_CALL(zmq_send(soc, pushbuf.data(), pushbuf.size(), 0));
    STOP_TIMER();
    PRINT_TIMER_STATS(100);
  }

  /**
   * Try to push a cache from local to remote
   * return 0 iff success
   */
  int push_cache() {
    thread_local void * soc = nullptr;
    thread_local Mem pushbuf;
    THREAD_LOCAL_TIMER;
    if(not soc) {
      qassert(soc = zmq_socket(ctx, ZMQ_PUSH));
      int snd_hwm = 2;
      ZMQ_CALL(zmq_setsockopt(soc, ZMQ_SNDHWM, &snd_hwm, sizeof(snd_hwm)));
      ZMQ_CALL(zmq_connect(soc, push_endpoint.c_str()));
      pushbuf.resize(RM::pushbuf_size());
    }
    Message * push = reinterpret_cast<Message*>(pushbuf.data());
    bool ret = prm->get_cache(cache_buf, push->sum_weight);
    if(not ret) // failed
      return -1;
    push->type = Message::ProtocalCache;
    push->length = Cache::nbytes(prm);
    push->sender = prm->uuid;
    START_TIMER();
    ZMQ_CALL(zmq_send(soc, pushbuf.data(), pushbuf.size(), ZMQ_SNDMORE));
    ZMQ_CALL(zmq_send(soc, cache_buf, push->length, 0));
    STOP_TIMER();
    PRINT_TIMER_STATS(100);
    return 0;
  }

  /**
   * Write a log to server
   */
  void write_log(const std::string& msg) {
    thread_local void * soc = nullptr;
    thread_local Mem pushbuf;
    THREAD_LOCAL_TIMER;
    if(not soc) {
      qassert(soc = zmq_socket(ctx, ZMQ_PUSH));
      ZMQ_CALL(zmq_connect(soc, push_endpoint.c_str()));
      pushbuf.resize(RM::pushbuf_size());
    }
    Message * push = reinterpret_cast<Message*>(pushbuf.data());
    push->type = Message::ProtocalLog;
    push->length = (int)msg.size();
    push->sender = prm->uuid;
    push->sum_weight = 0.0;
    START_TIMER();
    ZMQ_CALL(zmq_send(soc, pushbuf.data(), pushbuf.size(), ZMQ_SNDMORE));
    ZMQ_CALL(zmq_send(soc, msg.data(), push->length, 0));
    STOP_TIMER();
    PRINT_TIMER_STATS(100);
  }

};

typedef ReplayMemoryClient<RM> RMC;

