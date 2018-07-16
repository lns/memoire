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
  typedef typename RM::Message Message;
  typedef typename RM::DataCache Cache;

  RM rem;
  void * ctx;
  void * pssoc;                      ///< PUB/SUB socket

  Vector<Cache> caches;              ///< caches of data collected from actors
  PrtTree cache_prt;                 ///< PrtTree for sampling
  std::vector<size_t> sample_index;  ///< number of samples used for each cache
  int cache_index;                   ///< index of oldest cache (to be overwritten by new one)
  size_t total_caches;               ///< counter of total caches
  std::mutex cache_mutex;

  size_t total_episodes;             ///< counter of total episodes
  size_t total_steps;                ///< counter of total steps
  std::mutex counter_mutex;

  ReplayMemoryServer(const BufView * vw, size_t max_step, qlib::RNG * prt_rng, const char* pub_endpoint, int n_caches)
    : rem{vw, max_step, prt_rng}, ctx{nullptr},
    caches{0}, cache_prt{prt_rng, n_caches},
    total_caches{0}, total_episodes{0}, total_steps{0}
  {
    ctx = zmq_ctx_new(); qassert(ctx);
    caches.resize(n_caches);
    sample_index.resize(n_caches, 0);
    cache_index = 0;
    // PUB endpoint
    if(pub_endpoint and strcmp(pub_endpoint, "")) {
      pssoc = zmq_socket(ctx, ZMQ_PUB); qassert(pssoc);
      ZMQ_CALL(zmq_bind(pssoc, pub_endpoint));
    }
  }

  ~ReplayMemoryServer() {
    zmq_close(pssoc);
    zmq_ctx_destroy(ctx);
  }

  void print_info(FILE * f = stderr) const
  {
    fprintf(f, "total_episodes:%lu\n", total_episodes);
    fprintf(f, "total_caches:  %lu\n", total_caches);
    fprintf(f, "total_steps:   %lu\n", total_steps);
    rem.print_info(f);
  }

  /**
   * Publish a byte string to clients
   */
  void pub_bytes(const std::string& topic, const std::string& message) {
    static qlib::Timer timer;
    if(not pssoc)
      qlog_error("PUB/SUB socket is not opened.\n");
    //qlog_info("Send topic: %s\n", topic.c_str());
    timer.start();
    ZMQ_CALL(zmq_send(pssoc, topic.data(), topic.size(), ZMQ_SNDMORE));
    ZMQ_CALL(zmq_send(pssoc, message.data(), message.size(), 0));
    timer.stop();
    if(timer.cnt() % 100 == 0)
      qlog_info("%s(): n: %lu, min: %f, avg: %f, max: %f (msec)\n",
          __func__, timer.cnt(), timer.min(), timer.avg(), timer.max());
  }
  
  /**
   * Get a batch of samples from caches, to be used by learner
   *
   * Assuming the memory layout for data (state, action ..) as
   *  data[batch_size][data_size]
   * 
   * Suppose frame_stack = l, multi_step = k, this function will get a batch of
   *
   * s_{t-l},   s_{t-l+1},   ..., s_{t}    in prev_s, and
   * s_{t+k-l}, s_{t+k-l+1}, ..., s_{t+k}  in next_s.
   *
   * for action, reward, logp and value, this function will get
   * a_{t}                                 in prev_a, and
   * a_{t+k}                               in next_a.
   *
   * @param batch_size             batch_size
   *
   * @return true iff success
   */
  bool get_batch(size_t batch_size,
      void * prev_s,
      void * prev_a,
      void * prev_r,
      void * prev_p,
      void * prev_v,
      void * prev_q,
      void * prev_i,
      void * next_s,
      void * next_a,
      void * next_r,
      void * next_p,
      void * next_v,
      void * next_q,
      void * next_i,
      float * entry_weight_arr)
  {
    if(total_caches < caches.size()) {
      qlog_warning("get_batch() failed as caches are not all filled (%lu < %lu).\n", total_caches, caches.size());
      return false;
    }
    if(cache_prt.get_weight_sum() <= 0.0) {
      qlog_warning("get_batch() failed as no cached samples are available. (reuse_cache? too many pull_workers?)\n");
      return false;
    }
    for(size_t i=0; i<batch_size; i++) {
      long c_idx; // cache index
      int s_idx; // sample index
      if(true) {
        std::lock_guard<std::mutex> guard(cache_mutex);
        c_idx = cache_prt.sample_index();
        s_idx = sample_index[c_idx];
        if(s_idx >= rem.cache_size) {
          if(rem.reuse_cache) {
            s_idx = sample_index[c_idx] = 0;
          } else {
            qlog_warning("all caches pushed by actors are used. (total_caches: %lu)\n", total_caches);
            return false;
          }
        }
        sample_index[c_idx] += 1;
        if(sample_index[c_idx] >= rem.cache_size and !rem.reuse_cache) { // cache is fully used
          cache_prt.set_weight(c_idx, 0.0);
        }
      }
      // add caches[c_idx][s_idx] to batch
      auto& s = caches[c_idx].get(s_idx, &rem);
      s.to_memory(&rem, i,
          prev_s, prev_a, prev_r, prev_p, prev_v, prev_q, prev_i,
          next_s, next_a, next_r, next_p, next_v, next_q, next_i,
          entry_weight_arr);
    }
    return true;
  }

  /**
   * Get BufView of batch data
   *
   * @param batch_size
   *
   * @return prev s,a,r,p,v,q,i, next s,a,r,p,v,q,i
   */
  std::vector<BufView> get_batch_view(size_t batch_size) {
    std::vector<BufView> ret(2*N_VIEW);
    for(int i=0; i<2*N_VIEW; i++) {
      ret[i] = rem.view[i % N_VIEW];
      // frame_stack
      if(i % N_VIEW == 0)
        ret[i].shape_.insert(ret[i].shape_.begin(), (ssize_t)rem.frame_stack);
      if(rem.cache_flags[i])
        ret[i].shape_.insert(ret[i].shape_.begin(), (ssize_t)batch_size);
      else
        ret[i].shape_.insert(ret[i].shape_.begin(), 0);
      ret[i].make_c_stride();
      if(i % N_VIEW == 0)
        assert(ret[i][0][0].is_consistent_with(rem.view[i % N_VIEW]));
      else
        assert(ret[i][0].is_consistent_with(rem.view[i % N_VIEW]));
    }
    return ret;
  }

  /**
   * Responsible for answering the request of GetSizes.
   */
  void rep_worker_main(const char * endpoint, typename RM::Mode mode)
  {
    void * soc = zmq_socket(ctx, ZMQ_REP); qassert(soc);
    if(mode == RM::Bind)
      ZMQ_CALL(zmq_bind(soc, endpoint));
    else if(mode == RM::Conn)
      ZMQ_CALL(zmq_connect(soc, endpoint));
    Mem reqbuf(rem.reqbuf_size());
    Mem repbuf(rem.repbuf_size());
    Message * args = reinterpret_cast<Message*>(reqbuf.data());
    Message * rets = reinterpret_cast<Message*>(repbuf.data());
    int size;
    while(true) {
      ZMQ_CALL(size = zmq_recv(soc, reqbuf.data(), reqbuf.size(), 0)); qassert(size <= (int)reqbuf.size());
      rets->type = args->type;
      rets->sender = rem.uuid;
      if(args->type == Message::ProtocalSizes) {
        // return sizes
        auto vw = ArrayView<BufView::Data>(&rets->payload, N_VIEW);
        RM * p = reinterpret_cast<RM *>((char*)vw.data() + vw.nbytes());
        for(int i=0; i<N_VIEW; i++)
          vw[i].from(rem.view[i]);
        // memcpy. Only primary objects are valid.
        memcpy(p, &rem, sizeof(RM));
        //qlog_info("REP: ProtocalSizes\n");
        ZMQ_CALL(zmq_send(soc, repbuf.data(), repbuf.size(), 0));
      }
      else if(args->type == Message::ProtocalCounter) {
        // Update counters
        if(true) {
          std::lock_guard<std::mutex> guard(counter_mutex);
          int * p_length = reinterpret_cast<int *>(&args->payload);
          total_episodes += 1;
          total_steps += *p_length;
        }
        //qlog_info("REP: ProtocalCounter: total_steps: %lu\n", total_steps);
        ZMQ_CALL(zmq_send(soc, repbuf.data(), sizeof(Message), 0)); //
      }
      else
        qthrow("Unknown args->type");
    }
    // never
    zmq_close(soc);
  }

  /**
   * Responsible for accepting pushed caches
   */
  void pull_worker_main(const char * endpoint, typename RM::Mode mode) {
    void * soc = zmq_socket(ctx, ZMQ_PULL); qassert(soc);
    if(mode == RM::Bind)
      ZMQ_CALL(zmq_bind(soc, endpoint));
    else if(mode == RM::Conn)
      ZMQ_CALL(zmq_connect(soc, endpoint));
    if(caches.entry_size != Cache::nbytes(&rem)) { 
      std::lock_guard<std::mutex> guard(cache_mutex);
      size_t n_caches = caches.size();
      caches.~Vector<Cache>();
      new(&caches) Vector<Cache>(Cache::nbytes(&rem));
      caches.resize(n_caches);
      sample_index.resize(n_caches, 0);
      cache_index = 0;
    }
    Mem buf(rem.pushbuf_size());
    Message * args = reinterpret_cast<Message*>(buf.data());
    int size;
    int idx;
    const size_t expected_size = Cache::nbytes(&rem);
    if(true) {
      std::lock_guard<std::mutex> guard(cache_mutex);
      cache_prt.set_weight(cache_index, 0.0);
      idx = cache_index;
      cache_index = (cache_index + 1) % caches.size();
    }
    while(true) {
      ZMQ_CALL(size = zmq_recv(soc, buf.data(), buf.size(), 0)); qassert(size <= (int)buf.size());
      if(args->type == Message::ProtocalCache) {
        qassert(args->length == (int)expected_size);
        //qlog_info("PULL: ProtocalCache: idx: %d, sum_weight: %lf\n", idx, args->sum_weight);
        ZMQ_CALL(size = zmq_recv(soc, &caches[idx], expected_size, 0));
        qassert(size == (int)expected_size);
        sample_index[idx] = 0;
        if(true) {
          std::lock_guard<std::mutex> guard(cache_mutex);
          cache_prt.set_weight(idx, args->sum_weight); // payload as sum of weight
          total_caches += 1;
          cache_prt.set_weight(cache_index, 0.0);
          idx = cache_index;
          cache_index = (cache_index + 1) % caches.size();
        }
      }
      else
        qthrow("Unknown args->type");
    }
    // never
    zmq_close(soc);
  }

  /**
   * Responsible for connecting multiple frontends and backends
   */
  void proxy_main(
      int front_soc_type,
      const char * front_endpoint,
      int back_soc_type,
      const char * back_endpoint)
  {
    void * front = zmq_socket(ctx, front_soc_type); qassert(front);
    void * back  = zmq_socket(ctx,  back_soc_type); qassert(back);
    ZMQ_CALL(zmq_bind(front, front_endpoint));
    ZMQ_CALL(zmq_bind(back, back_endpoint));
    ZMQ_CALL(zmq_proxy(front, back, nullptr));
    // never
    zmq_close(front);
    zmq_close(back);
  }

  void rep_proxy_main(const char * front_ep, const char * back_ep) {
    proxy_main(ZMQ_ROUTER, front_ep, ZMQ_DEALER, back_ep);
  }

  void pull_proxy_main(const char * front_ep, const char * back_ep) {
    proxy_main(ZMQ_PULL, front_ep, ZMQ_PUSH, back_ep);
  }

  // TODO: Proxy for PUB/SUB protocal

};

typedef ReplayMemoryServer<RM> RMS;

