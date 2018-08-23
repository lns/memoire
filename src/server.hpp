#pragma once

#include "utils.hpp"
#include <vector>
#include <thread>
#include <cstdlib>
#include <functional>
#include "replay_memory.hpp"
#include "hexdump.hpp"
#include <arpa/inet.h>

template<class RM>
class ReplayMemoryServer : public non_copyable {
public:
  typedef typename RM::Message Message;
  typedef typename RM::DataCache Cache;

  RM rem;
  std::string pub_endpoint;
  void * ctx;

  Vector<Cache> * caches;            ///< caches of data collected from actors
  PrtTree cache_prt;                 ///< PrtTree for sampling
  std::vector<size_t> sample_index;  ///< number of samples used for each cache
  int cache_index;                   ///< index of oldest cache (to be overwritten by new one)
  size_t total_caches;               ///< counter of total caches
  std::mutex cache_mutex;

  size_t total_episodes;             ///< counter of total episodes
  size_t total_steps;                ///< counter of total steps
  std::mutex counter_mutex;

  std::string logfile_path;          ///< path of logfile
  FILE * logfile;                    ///< logfile object
  std::mutex logfile_mutex;

  ReplayMemoryServer(const BufView * vw, size_t max_step, qlib::RNG * prt_rng, const std::string& pub_ep, int n_caches)
    : rem{vw, max_step, prt_rng}, pub_endpoint{pub_ep},
      ctx{nullptr}, caches{nullptr}, cache_prt{prt_rng, n_caches},
      total_caches{0}, total_episodes{0}, total_steps{0}, logfile{nullptr}
  {
    ctx = zmq_ctx_new(); qassert(ctx);
    sample_index.resize(n_caches, 0);
    cache_index = 0;
  }

  ~ReplayMemoryServer() {
    if(caches)
      delete caches;
    if(logfile)
      fclose(logfile);
    zmq_ctx_destroy(ctx);
  }

  void print_info(FILE * f = stderr) const
  {
    fprintf(f, "total_episodes:%lu\n", total_episodes);
    fprintf(f, "total_caches:  %lu\n", total_caches);
    fprintf(f, "total_steps:   %lu\n", total_steps);
    fprintf(f, "logfile:       %s\n",  logfile_path.c_str());
    rem.print_info(f);
  }

  void set_logfile(const std::string& filepath, const std::string& mode) {
    std::lock_guard<std::mutex> guard(logfile_mutex);
    if(logfile)
      fclose(logfile);
    logfile = fopen(filepath.c_str(), mode.c_str());
    if(not logfile)
      qlog_warning("Failed to open file '%s' with mode '%s'.\n", filepath.c_str(), mode.c_str());
    logfile_path = filepath;
  }

  /**
   * Publish a byte string to clients
   */
  void pub_bytes(const std::string& topic, const std::string& message) {
    thread_local void * soc = nullptr;
    THREAD_LOCAL_TIMER;
    if(not soc) {
      qassert(soc = zmq_socket(ctx, ZMQ_PUB));
      ZMQ_CALL(zmq_bind(soc, pub_endpoint.c_str()));
    }
    START_TIMER();
    ZMQ_CALL(zmq_send(soc, topic.data(), topic.size(), ZMQ_SNDMORE));
    ZMQ_CALL(zmq_send(soc, message.data(), message.size(), 0));
    STOP_TIMER();
    PRINT_TIMER_STATS(10);
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
    if(not caches) {
      qlog_warning("caches are not initialized yet.\n");
      return false;
    }
    if(total_caches < caches->size()) {
      qlog_warning("get_batch() failed as caches are not all filled (%lu < %lu).\n", total_caches, caches->size());
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
        if(s_idx >= (int)rem.cache_size) {
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
      auto& s = (*caches)[c_idx].get(s_idx, &rem);
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
    THREAD_LOCAL_TIMER;
    Mem reqbuf(rem.reqbuf_size());
    Mem repbuf(rem.repbuf_size());
    Message * args = reinterpret_cast<Message*>(reqbuf.data());
    Message * rets = reinterpret_cast<Message*>(repbuf.data());
    int size;
    while(true) {
      ZMQ_CALL(size = zmq_recv(soc, reqbuf.data(), reqbuf.size(), 0)); qassert(size <= (int)reqbuf.size());
      qassert(args->check_version());
      rets->version = VERSION;
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
        START_TIMER();
        ZMQ_CALL(zmq_send(soc, repbuf.data(), repbuf.size(), 0));
        STOP_TIMER();
        PRINT_TIMER_STATS(10);
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
    if(not caches) {
      std::lock_guard<std::mutex> guard(cache_mutex);
      size_t n_caches = sample_index.size();
      caches = new Vector<Cache>(Cache::nbytes(&rem));
      caches->resize(n_caches);
    }
    THREAD_LOCAL_TIMER;
    Mem buf(rem.pushbuf_size());
    Message * args = reinterpret_cast<Message*>(buf.data());
    int size;
    int idx;
    const size_t expected_size = Cache::nbytes(&rem);
    if(true) {
      std::lock_guard<std::mutex> guard(cache_mutex);
      cache_prt.set_weight(cache_index, 0.0);
      idx = cache_index;
      cache_index = (cache_index + 1) % caches->size();
    }
    std::string msg_buf;
    while(true) {
      START_TIMER();
      ZMQ_CALL(size = zmq_recv(soc, buf.data(), buf.size(), 0)); qassert(size == (int)buf.size());
      STOP_TIMER();
      qassert(args->check_version());
      if(args->type == Message::ProtocalCache) {
        qassert(args->length == (int)expected_size);
        qassert(check_multipart(soc));
        ZMQ_CALL(size = zmq_recv(soc, &(*caches)[idx], expected_size, 0));
        if(size != (int)expected_size) {
          qlog_warning("To be fixed: %d != %d\n", size, (int)expected_size);
          hexdump(stderr, &(*caches)[idx], size);
        }
        qassert(size == (int)expected_size);
        sample_index[idx] = 0;
        if(true) {
          std::lock_guard<std::mutex> guard(cache_mutex);
          cache_prt.set_weight(idx, args->sum_weight); // payload as sum of weight
          total_caches += 1;
          cache_prt.set_weight(cache_index, 0.0);
          idx = cache_index;
          cache_index = (cache_index + 1) % caches->size();
        }
      }
      else if(args->type == Message::ProtocalCounter) {
        // Update counters
        std::lock_guard<std::mutex> guard(counter_mutex);
        long * p_data = reinterpret_cast<long *>(&args->payload);
        total_episodes += p_data[0];
        total_steps += p_data[1];
      }
      else if(args->type == Message::ProtocalLog) {
        if(args->length >= (int)msg_buf.size())
          msg_buf.resize(args->length + 1, '\0');
        qassert(check_multipart(soc));
        ZMQ_CALL(size = zmq_recv(soc, &msg_buf[0], msg_buf.size(), 0));
        if(size != args->length) {
          qlog_warning("To be fixed: %d != %d, buf: %lu\n", size, args->length, msg_buf.size());
          hexdump(stderr, &msg_buf[0], size);
        }
        qassert(size == args->length);
        msg_buf[size] = '\0';
        // Convert args->sender to IP address
        struct in_addr ip_addr;
        ip_addr.s_addr = args->sender;
        char * sender_ip = inet_ntoa(ip_addr);
        if(logfile) {
          std::lock_guard<std::mutex> guard(logfile_mutex);
          fprintf(logfile, "%s,%s,%s\n", qlib::timestr().c_str(), sender_ip, msg_buf.data());
          fflush(logfile);
        }
      }
      else
        qthrow("Unknown args->type");
      PRINT_TIMER_STATS(100);
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

  void pub_proxy_main(const char * front_ep, const char * back_ep) {
    proxy_main(ZMQ_XSUB, front_ep, ZMQ_XPUB, back_ep);
  }

};

typedef ReplayMemoryServer<RM> RMS;

