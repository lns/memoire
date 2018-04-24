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

  RM * prm;
  void * ctx;

  Vector<Cache> caches;              ///< caches of data collected from actors
  PrtTree cache_prt;                 ///< PrtTree for sampling
  std::vector<int> sample_index;     ///< number of samples used for each cache
  int cache_index;                   ///< index of oldest cache (to be overwritten by new one)
  size_t total_caches;               ///< counter of total caches
  std::mutex cache_mutex;

  size_t total_episodes;             ///< counter of total episodes
  size_t total_steps;                ///< counter of total steps
  std::mutex counter_mutex;

  ReplayMemoryServer(RM * p_rem, int n_caches) : prm{p_rem}, ctx{nullptr},
    caches{Cache::nbytes(prm)}, cache_prt{prm->rng, n_caches}, total_caches{0},
    total_episodes{0}, total_steps{0}
  {
    ctx = zmq_ctx_new(); qassert(ctx);
    caches.resize(n_caches);
    sample_index.resize(n_caches, 0);
    cache_index = 0;
  }

  ~ReplayMemoryServer() { zmq_ctx_destroy(ctx); }

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
      typename RM::state_t  * prev_s,
      typename RM::action_t * prev_a,
      typename RM::reward_t * prev_r,
      typename RM::prob_t   * prev_p,
      typename RM::value_t  * prev_v,
      typename RM::state_t  * next_s,
      typename RM::action_t * next_a,
      typename RM::reward_t * next_r,
      typename RM::prob_t   * next_p,
      typename RM::value_t  * next_v,
      float * entry_weight_arr)
  {
    if(total_caches < caches.size()) {
      qlog_warning("get_batch() failed as caches are not all filled.\n");
      return false;
    }
    for(size_t i=0; i<batch_size; i++) {
      long c_idx; // cache index
      int s_idx; // sample index
      if(true) {
        std::lock_guard<std::mutex> guard(cache_mutex);
        c_idx = cache_prt.sample_index();
        s_idx = sample_index[c_idx];
        qlog_info("c_idx: %ld, s_idx: %d\n", c_idx, s_idx); // DEBUG
        qassert(s_idx < prm->cache_size); // current sample is valid
        sample_index[c_idx] += 1;
        if(sample_index[c_idx] >= prm->cache_size) { // cache is fully used
          cache_prt.set_weight(c_idx, 0.0);
        }
      }
      // add caches[c_idx][s_idx] to batch
      auto& s = caches[c_idx].get(s_idx, prm);
      s.print_first(prm); // DEBUG
      s.to_memory(prm, i,
          prev_s, prev_a, prev_r, prev_p, prev_v,
          next_s, next_a, next_r, next_p, next_v,
          entry_weight_arr);
    }
    return true;
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
    int reqbuf_size = prm->reqbuf_size();
    char * reqbuf = (char*)malloc(reqbuf_size); qassert(reqbuf);
    int repbuf_size = prm->repbuf_size();
    char * repbuf = (char*)malloc(repbuf_size); qassert(repbuf);
    Message * args = reinterpret_cast<Message*>(reqbuf);
    Message * rets = reinterpret_cast<Message*>(repbuf);
    int size;
    while(true) {
      ZMQ_CALL(size = zmq_recv(soc, reqbuf, reqbuf_size, 0)); qassert(size <= reqbuf_size);
      rets->type = args->type;
      rets->sender = prm->uuid;
      if(args->type == Message::ProtocalSizes) {
        // return sizes
        RM * p = reinterpret_cast<RM *>(&rets->payload);
        // memcpy. Only primary objects are valid.
        memcpy(p, prm, sizeof(RM));
        qlog_info("REP: ProtocalSizes\n");
        ZMQ_CALL(zmq_send(soc, repbuf, repbuf_size, 0));
      }
      else if(args->type == Message::ProtocalCounter) {
        // Update counters
        if(true) {
          std::lock_guard<std::mutex> guard(counter_mutex);
          int * p_length = reinterpret_cast<int *>(&args->payload);
          total_episodes += 1;
          total_steps += *p_length;
        }
        qlog_info("REP: ProtocalCounter: total_steps: %lu\n", total_steps);
        ZMQ_CALL(zmq_send(soc, repbuf, repbuf_size, 0));
      }
      else
        qthrow("Unknown args->type");
    }
    // never
    free(reqbuf);
    free(repbuf);
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
    int buf_size = prm->pushbuf_size();
    char * buf = (char*)malloc(buf_size); qassert(buf);
    Message * args = reinterpret_cast<Message*>(buf);
    int size;
    while(true) {
      ZMQ_CALL(size = zmq_recv(soc, buf, buf_size, 0)); qassert(size <= buf_size);
      if(args->type == Message::ProtocalCache) {
        int idx;
        if(true) {
          std::lock_guard<std::mutex> guard(cache_mutex);
          cache_prt.set_weight(cache_index, 0.0);
          idx = cache_index;
          cache_index = (cache_index + 1) % caches.size();
          total_caches += 1;
        }
        size_t expected_size = Cache::nbytes(prm);
        qassert(args->length == expected_size);
        ZMQ_CALL(size = zmq_recv(soc, &caches[idx], expected_size, 0));
        qlog_info("PULL: ProtocalCache: nbytes(): %lu\n", expected_size);
        for(int i=0; i<prm->cache_size; i++)
          caches[idx].get(i, prm).print_first(prm);
        qassert(size == expected_size);
        sample_index[idx] = 0;
        if(true) {
          std::lock_guard<std::mutex> guard(cache_mutex);
          cache_prt.set_weight(idx, args->sum_weight); // payload as sum of weight
        }
      }
      else
        qthrow("Unknown args->type");
    }
    // never
    free(buf);
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

};

