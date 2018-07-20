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
  // ZeroMQ socket is not thread-safe, so a socket should only
  // be used by a single thread at the same time.
  void * ctx;
  void * rrsoc; // for sync_sizes()
  void * ppsoc; // for push_cache()
  void * lgsoc; // for write_log()
  void * pssoc; // for sub_bytes()
  Mem reqbuf, repbuf, pushbuf, subbuf, logbuf;
  Message *req, *rep, *push, *sub, *log;
  Cache * cache_buf;

public:
  /**
   * Initialization
   *
   * endpoint can be set to nullptr or "" to disable this protocal
   */
  ReplayMemoryClient(
      const char * sub_endpoint,
      const char * req_endpoint,
      const char * push_endpoint) : prm(nullptr), cache_buf(nullptr)
  {
    ctx = zmq_ctx_new(); qassert(ctx);
    if(sub_endpoint and 0!=strcmp(sub_endpoint, "")) {
      pssoc = zmq_socket(ctx, ZMQ_SUB); qassert(pssoc);
      ZMQ_CALL(zmq_connect(pssoc, sub_endpoint));
    }
    if(req_endpoint and 0!=strcmp(req_endpoint, "")) {
      rrsoc = zmq_socket(ctx, ZMQ_REQ); qassert(rrsoc);
      ZMQ_CALL(zmq_connect(rrsoc, req_endpoint));
    }
    if(push_endpoint and 0!=strcmp(push_endpoint, "")) {
      ppsoc = zmq_socket(ctx, ZMQ_PUSH); qassert(ppsoc);
      ZMQ_CALL(zmq_connect(ppsoc, push_endpoint));  
      lgsoc = zmq_socket(ctx, ZMQ_PUSH); qassert(lgsoc);
      ZMQ_CALL(zmq_connect(lgsoc, push_endpoint));  
    }
    // Make bufs
    subbuf.resize(256); // TODO
    sub  = reinterpret_cast<Message*>(subbuf.data());
    reqbuf.resize(RM::reqbuf_size());
    req  = reinterpret_cast<Message*>(reqbuf.data());
    repbuf.resize(RM::repbuf_size());
    rep  = reinterpret_cast<Message*>(repbuf.data());
    pushbuf.resize(RM::pushbuf_size());
    push = reinterpret_cast<Message*>(pushbuf.data());
    logbuf.resize(RM::pushbuf_size());
    log  = reinterpret_cast<Message*>(logbuf.data());
    // Empty RM
    prm = nullptr; //new RM{0,0,0,0,0,0,0,0,&lcg64};
  }

  ~ReplayMemoryClient() {
    if(prm)
      delete prm;
    if(cache_buf)
      free(cache_buf);
    zmq_close(rrsoc);
    zmq_close(ppsoc);
    zmq_close(pssoc);
    zmq_close(lgsoc);
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
    if(not rrsoc)
      qlog_error("REQ/REP socket is not connected.\n");
    req->type = Message::ProtocalSizes;
    req->sender = 0;
    ZMQ_CALL(zmq_send(rrsoc, reqbuf.data(), reqbuf.size(), 0));
    ZMQ_CALL(zmq_recv(rrsoc, repbuf.data(), repbuf.size(), 0));
    qassert(rep->type == req->type);
    // Get sizes
    Message * rets = reinterpret_cast<Message*>(repbuf.data());
    auto vw = ArrayView<BufView::Data>(&rets->payload, N_VIEW);
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
    if(not pssoc)
      qlog_error("PUB/SUB socket is not connected.\n");
    thread_local Mem topicbuf(256);
    thread_local std::string last_topic("");
    if(last_topic != "")
      ZMQ_CALL(zmq_setsockopt(pssoc, ZMQ_UNSUBSCRIBE, last_topic.c_str(), last_topic.size()));
    last_topic = topic;
    if(topic != "")
      ZMQ_CALL(zmq_setsockopt(pssoc, ZMQ_SUBSCRIBE, topic.c_str(), topic.size()));
    int size;
    while(true) {
      memset(topicbuf.data(), 0, topicbuf.size());
      memset(subbuf.data(), 0, subbuf.size());
      // Recv topic
      ZMQ_CALL(size = zmq_recv(pssoc, topicbuf.data(), topicbuf.size(), 0)); qassert(size <= (int)topicbuf.size());
      // Recv Message
      ZMQ_CALL(size = zmq_recv(pssoc, subbuf.data(), subbuf.size(), 0));
      if(strcmp((const char*)topicbuf.data(), topic.data())) { // topic mismatch
        qlog_warning("topic mismatch: '%s' != '%s'\n", (const char *)topicbuf.data(), topic.c_str());
        continue;
      }
      if(not (size <= (int)subbuf.size())) {
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
    thread_local qlib::Timer timer;
    if(not rrsoc)
      qlog_error("REQ/REP socket is not connected.\n");
    timer.start();
    req->type = Message::ProtocalCounter;
    req->sender = prm->uuid;
    int * p_length = reinterpret_cast<int *>(&req->payload);
    *p_length = prm->new_length;
    //qlog_info("%s(): length: %d\n",__func__, *p_length);
    ZMQ_CALL(zmq_send(rrsoc, reqbuf.data(), reqbuf.size(), 0));
    ZMQ_CALL(zmq_recv(rrsoc, repbuf.data(), repbuf.size(), 0));
    timer.stop();
    if(timer.cnt() % 100 == 0)
      qlog_info("%s(): n: %lu, min: %f , avg: %f, max: %f (msec)\n",
          __func__, timer.cnt(), timer.min(), timer.avg(), timer.max());
  }

  /**
   * Try to push a cache from local to remote
   * return 0 iff success
   */
  int push_cache() {
    if(not ppsoc)
      qlog_error("PUSH/PULL socket is not connected.\n");
    bool ret = prm->get_cache(cache_buf, push->sum_weight);
    if(not ret) // failed
      return -1;
    //cache_buf->get(prm->cache_size-1, prm).print_first(prm); // DEBUG
    push->type = Message::ProtocalCache;
    push->length = Cache::nbytes(prm);
    push->sender = prm->uuid;
    //qlog_info("%s(): cache_size: %d, cache::nbytes: %lu\n", __func__, prm->cache_size, Cache::nbytes(prm));
    ZMQ_CALL(zmq_send(ppsoc, pushbuf.data(), pushbuf.size(), ZMQ_SNDMORE));
    ZMQ_CALL(zmq_send(ppsoc, cache_buf, push->length, 0));
    return 0;
  }

  /**
   * Write a log to server
   */
  void write_log(const std::string& msg) {
    if(not lgsoc)
      qlog_error("PUSH/PULL (log) socket is not connected.\n");
    log->type = Message::ProtocalLog;
    log->length = (int)msg.size();
    log->sender = prm->uuid;
    log->sum_weight = 0.0;
    ZMQ_CALL(zmq_send(lgsoc, logbuf.data(), logbuf.size(), ZMQ_SNDMORE));
    ZMQ_CALL(zmq_send(lgsoc, msg.data(), log->length, 0));
  }

};

typedef ReplayMemoryClient<RM> RMC;

