#pragma once
#include <cstdint>
#include <atomic>
#include <mutex>
#include <queue>
#include "qlog.hpp"
#include "qtime.hpp"
#include "array_view.hpp"
#include "vector.hpp"
#include "qrand.hpp"
//#include <signal.h>
#include "utils.hpp" // non_copyable
#include "prt_tree.hpp"
#include "buffer_view.hpp"

#define MAX_RWD_DIM (256)
#define N_VIEW (7)
#define VERSION (20180823ul)

/**
 * Distributed Replay Memory
 * 
 * Each instance will be created for each actor.
 * There may be multiple push threads to push batches to remote batch buffer simutanously.
 * However there will be only one thread to modify the local replay memory instance.
 * We hope this design will simplify the concurrent code to avoid conflict.
 *
 * We require reward, prob, value, qvest to be float32
 */
class ReplayMemory
{
public:
  /**
   * Memory structure of an entry. Does not own the memory.
   * (Actually, we should not have member variables, except for static ones)
   */
  class DataEntry : public non_copyable
  {
    BufView get(int i, const ReplayMemory * p) {
      assert(0<= i and i < N_VIEW);
      BufView cur = p->view[i];
      if(i > 0) {
        BufView last = get(i-1, p);
        cur.ptr_ = (char*)last.ptr_ + last.nbytes();
      } else {
        cur.ptr_ = this;
      }
      return cur;
    }
  public:
    BufView state(const ReplayMemory * p)  { return get(0, p); }
    BufView action(const ReplayMemory * p) { return get(1, p); }
    BufView reward(const ReplayMemory * p) { return get(2, p); }
    BufView prob(const ReplayMemory * p)   { return get(3, p); }
    BufView value(const ReplayMemory * p)  { return get(4, p); }
    BufView qvest(const ReplayMemory * p)  { return get(5, p); }
    BufView info(const ReplayMemory * p)   { return get(6, p); }

    static size_t nbytes(const ReplayMemory * p) {
      DataEntry * dummy = (DataEntry*)nullptr;
      auto last = dummy->get(N_VIEW-1, p);
      return ((char*)last.ptr_ - (char*)dummy) + last.nbytes();
    }

    /**
     * Copy from memory
     * @param index  denotes the offset in memory
     */
    void from_memory(const ReplayMemory * p, size_t index,
        const void * p_s,
        const void * p_a,
        const void * p_r,
        const void * p_p,
        const void * p_v,
        const void * p_q,
        const void * p_i)
    {
      if(p_s) {
        auto buf = state(p);
        buf.from_memory((char*)p_s + index * buf.nbytes());
      }
      if(p_a) {
        auto buf = action(p);
        buf.from_memory((char*)p_a + index * buf.nbytes());
      }
      if(p_r) {
        auto buf = reward(p);
        buf.from_memory((char*)p_r + index * buf.nbytes());
      }
      if(p_p) {
        auto buf = prob(p);
        buf.from_memory((char*)p_p + index * buf.nbytes());
      }
      if(p_v) {
        auto buf = value(p);
        buf.from_memory((char*)p_v + index * buf.nbytes());
      }
      if(p_q) {
        auto buf = qvest(p);
        buf.from_memory((char*)p_q + index * buf.nbytes());
      }
      if(p_i) {
        auto buf = info(p);
        buf.from_memory((char*)p_i + index * buf.nbytes());
      }
    }

    /**
     * Copy to memory
     * @param index  denotes the offset in memory (see ReplayMemory::get_batch())
     */
    void to_memory(const ReplayMemory * p, size_t index,
        void * p_s,
        void * p_a,
        void * p_r,
        void * p_p,
        void * p_v,
        void * p_q,
        void * p_i)
    {
      if(p_s)
        state(p).to_memory((char*)p_s + index * state(p).nbytes());
      if(p_a)
        action(p).to_memory((char*)p_a + index * action(p).nbytes());
      if(p_r)
        reward(p).to_memory((char*)p_r + index * reward(p).nbytes());
      if(p_p)
        prob(p).to_memory((char*)p_p + index * prob(p).nbytes());
      if(p_v)
        value(p).to_memory((char*)p_v + index * value(p).nbytes());
      if(p_q)
        qvest(p).to_memory((char*)p_q + index * qvest(p).nbytes());
      if(p_i)
        info(p).to_memory((char*)p_i + index * info(p).nbytes());
    }

  };

  /**
   * Memory structure of a sample / transition. Does not own the memory.
   * This is used in constructing cache.
   */
  class DataSample : public non_copyable
  {
    template<int N>
    static size_t ceil(size_t a) {
      return (a+N-1) / N * N;
    }
    BufView get(int i, const ReplayMemory * p, bool nullify_ptr_if_not_cached) {
      assert(0<= i and i <= 2 * N_VIEW);
      BufView cur = p->view[i % N_VIEW];
      if(i % N_VIEW == 0) { // state
        cur.shape_.insert(cur.shape_.begin(), (ssize_t)p->frame_stack);
        cur.make_c_stride();
      }
      if(i == 2 * N_VIEW) // entry_weight
        cur = BufView(nullptr, sizeof(float), "f", std::vector<ssize_t>(), std::vector<ssize_t>());
      if(i > 0) {
        BufView last = get(i-1, p, false);
        cur.ptr_ = (char*)last.ptr_ + (p->cache_flags[i-1] ? last.nbytes() : 0);
      } else {
        cur.ptr_ = this;
      }
      if(nullify_ptr_if_not_cached and i < 2 * N_VIEW and p->cache_flags[i] == 0)
        cur.ptr_ = nullptr;
      return cur;
    }
  public:
    BufView prev_state(const ReplayMemory * p)  { return get( 0, p, true); }
    BufView prev_action(const ReplayMemory * p) { return get( 1, p, true); }
    BufView prev_reward(const ReplayMemory * p) { return get( 2, p, true); }
    BufView prev_prob(const ReplayMemory * p)   { return get( 3, p, true); }
    BufView prev_value(const ReplayMemory * p)  { return get( 4, p, true); }
    BufView prev_qvest(const ReplayMemory * p)  { return get( 5, p, true); }
    BufView prev_info(const ReplayMemory * p)   { return get( 6, p, true); }
    BufView next_state(const ReplayMemory * p)  { return get( 7, p, true); }
    BufView next_action(const ReplayMemory * p) { return get( 8, p, true); }
    BufView next_reward(const ReplayMemory * p) { return get( 9, p, true); }
    BufView next_prob(const ReplayMemory * p)   { return get(10, p, true); }
    BufView next_value(const ReplayMemory * p)  { return get(11, p, true); }
    BufView next_qvest(const ReplayMemory * p)  { return get(12, p, true); }
    BufView next_info(const ReplayMemory * p)   { return get(13, p, true); }
    BufView entry_weight(const ReplayMemory * p){ return get(14, p, true); }

    static size_t nbytes(const ReplayMemory * p) {
      DataSample * dummy = (DataSample*)nullptr;
      auto last = dummy->get(2*N_VIEW, p, false);
      return ((char*)(last.ptr_) - (char*)dummy) + last.nbytes();
    }

    void print(const ReplayMemory * p, FILE * f = stderr) {
      if(p->cache_flags[ 0]) prev_state(p).print(f);
      if(p->cache_flags[ 1]) prev_action(p).print(f);
      if(p->cache_flags[ 2]) prev_reward(p).print(f);
      if(p->cache_flags[ 3]) prev_prob(p).print(f);
      if(p->cache_flags[ 4]) prev_value(p).print(f);
      if(p->cache_flags[ 5]) prev_qvest(p).print(f);
      if(p->cache_flags[ 6]) prev_info(p).print(f);
      if(p->cache_flags[ 7]) next_state(p).print(f);
      if(p->cache_flags[ 8]) next_action(p).print(f);
      if(p->cache_flags[ 9]) next_reward(p).print(f);
      if(p->cache_flags[10]) next_prob(p).print(f);
      if(p->cache_flags[11]) next_value(p).print(f);
      if(p->cache_flags[12]) next_qvest(p).print(f);
      if(p->cache_flags[13]) next_info(p).print(f);
      entry_weight(p).print(f);
    }

    void to_memory(const ReplayMemory * p, int idx,
        void * prev_s, void * prev_a, void * prev_r, void * prev_p,
        void * prev_v, void * prev_q, void * prev_i,
        void * next_s, void * next_a, void * next_r, void * next_p,
        void * next_v, void * next_q, void * next_i,
        float * entry_w) {
      if(prev_s and p->cache_flags[0]) {
        auto buf = prev_state(p);
        buf.to_memory((char*)prev_s + idx * buf.nbytes());
      }
      if(prev_a and p->cache_flags[1]) {
        auto buf = prev_action(p);
        buf.to_memory((char*)prev_a + idx * buf.nbytes());
      }
      if(prev_r and p->cache_flags[2]) {
        auto buf = prev_reward(p);
        buf.to_memory((char*)prev_r + idx * buf.nbytes());
      }
      if(prev_p and p->cache_flags[3]) {
        auto buf = prev_prob(p);
        buf.to_memory((char*)prev_p + idx * buf.nbytes());
      }
      if(prev_v and p->cache_flags[4]) {
        auto buf = prev_value(p);
        buf.to_memory((char*)prev_v + idx * buf.nbytes());
      }
      if(prev_q and p->cache_flags[5]) {
        auto buf = prev_qvest(p);
        buf.to_memory((char*)prev_q + idx * buf.nbytes());
      }
      if(prev_i and p->cache_flags[6]) {
        auto buf = prev_info(p);
        buf.to_memory((char*)prev_i + idx * buf.nbytes());
      }
      if(next_s and p->cache_flags[7]) {
        auto buf = next_state(p);
        buf.to_memory((char*)next_s + idx * buf.nbytes());
      }
      if(next_a and p->cache_flags[8]) {
        auto buf = next_action(p);
        buf.to_memory((char*)next_a + idx * buf.nbytes());
      }
      if(next_r and p->cache_flags[9]) {
        auto buf = next_reward(p);
        buf.to_memory((char*)next_r + idx * buf.nbytes());
      }
      if(next_p and p->cache_flags[10]) {
        auto buf = next_prob(p);
        buf.to_memory((char*)next_p + idx * buf.nbytes());
      }
      if(next_v and p->cache_flags[11]) {
        auto buf = next_value(p);
        buf.to_memory((char*)next_v + idx * buf.nbytes());
      }
      if(next_q and p->cache_flags[12]) {
        auto buf = next_qvest(p);
        buf.to_memory((char*)next_q + idx * buf.nbytes());
      }
      if(next_i and p->cache_flags[13]) {
        auto buf = next_info(p);
        buf.to_memory((char*)next_i + idx * buf.nbytes());
      }
      if(entry_w) {
        auto buf = entry_weight(p);
        buf.to_memory((char*)entry_w + idx * buf.nbytes());
      }
    }

  };

  class DataCache : public non_copyable
  {
  public:
    DataSample& get(size_t idx, const ReplayMemory * p) {
      assert(idx < p->cache_size);
      return *reinterpret_cast<DataSample*>((char*)(this) + DataSample::nbytes(p) * idx);
    }
    static size_t nbytes(const ReplayMemory * p) {
      return p->cache_size * DataSample::nbytes(p);
    }
  };

protected:
  class Episode {
  public:
    const long offset;
    const long length;
    Episode(long o, long l) : offset{o}, length{l} {}
  };

public:
  const BufView view[N_VIEW];        ///< buffer view of s, a, r, p, v, q, i

  const size_t entry_size;           ///< size of each entry (in bytes)
  const size_t max_step;             ///< max number of steps can be stored

  size_t max_episode;                ///< Max number of stored episodes allowed (0 for not checking)
  float priority_exponent;           ///< exponent of priority term (default 0.5)
  float mix_lambda;                  ///< mixture factor for computing multi-step return
  int frame_stack;                   ///< Number of frames stacked for each state (default 1)
  int multi_step;                    ///< Number of steps between prev and next (default 1)
  unsigned cache_size;               ///< Cache size, number of sample in a cache
  int reuse_cache;                   ///< whether we will reuse cache for sampling batches (see server.hpp)
  int autosave_step;                 ///< number of step for auto save (default 0 for no autosave)
  int replace_data;                  ///< whether cache is sampled with/without replacement. Not compatiable with priority.
  float discount_factor[MAX_RWD_DIM];///< discount factor for calculate R with rewards
  float reward_coeff[MAX_RWD_DIM];   ///< reward coefficient
  uint8_t cache_flags[2*N_VIEW];     ///< Whether we should collect prev s,a,r,p,v,q,i and next s,a,r,p,v,q,i in caches

  uint32_t uuid;                     ///< uuid for current instance

protected:
  Vector<DataEntry> data;            ///< Actual data of all samples
  std::queue<Episode> episode;       ///< Episodes
  PrtTree prt;                       ///< Priority tree for sampling
  std::mutex prt_mutex;              ///< mutex for priority tree
  int stage;                         ///< init -> 0 -> new -> 10 -> close -> 0
public:
  qlib::RNG * rng;                   ///< Random Number Generator

  long new_offset;                   ///< offset for new episode
  long new_length;                   ///< current length of new episode
  long incre_episode;                ///< incremental episode (for statistics in update_counter())
  long incre_step;                   ///< incremental step (for statistics in update_counter())

public:
  /**
   * Construct a new replay memory
   * @param e_size       size of a single entry (in bytes)
   * @param max_epi      max number of episodes kept in the memory
   * @param input_uuid   specified uuid, usually used for recording actor ip address
   */
  ReplayMemory(const BufView * vw,
      size_t m_step,
      qlib::RNG * prt_rng,
      uint32_t input_uuid = 0) :
    view{
      BufView(nullptr, vw[0].itemsize_, vw[0].format_, vw[0].shape_, vw[0].stride_),
      BufView(nullptr, vw[1].itemsize_, vw[1].format_, vw[1].shape_, vw[1].stride_),
      BufView(nullptr, vw[2].itemsize_, vw[2].format_, vw[2].shape_, vw[2].stride_),
      BufView(nullptr, vw[3].itemsize_, vw[3].format_, vw[3].shape_, vw[3].stride_),
      BufView(nullptr, vw[4].itemsize_, vw[4].format_, vw[4].shape_, vw[4].stride_),
      BufView(nullptr, vw[5].itemsize_, vw[5].format_, vw[5].shape_, vw[5].stride_),
      BufView(nullptr, vw[6].itemsize_, vw[6].format_, vw[6].shape_, vw[6].stride_),
    },
    entry_size{DataEntry::nbytes(this)},
    max_step{m_step},
    max_episode{0},
    cache_size{0},
    reuse_cache{0},
    autosave_step{0},
    replace_data{1},
    uuid{0},
    data{entry_size},
    prt{prt_rng, static_cast<int>(max_step)},
    rng{prt_rng},
    incre_episode{0},
    incre_step{0}
  {
    check();
    data.reserve(max_step);
    if(0!=input_uuid)
      uuid = input_uuid;
    else
      uuid = qlib::get_nsec();
    for(auto&& each : discount_factor)
      each = 1.0f;
    for(auto&& each : reward_coeff)
      each = 1.0f;
    for(auto&& each : cache_flags)
      each = 1;
    stage = 0;
  }

  void check() const {
    qassert(prob_buf().format_   == "f");
    qassert(reward_buf().format_ == "f");
    qassert(value_buf().format_  == "f");
    qassert(qvest_buf().format_  == "f");
    if(prob_buf().ndim() > 0)
      qlog_error("prob (%s) should be a single number.\n",
          prob_buf().str().c_str());
    if(reward_buf().ndim() > 1)
      qlog_error("reward (%s) should be a single number or a 1-dim array.\n",
          reward_buf().str().c_str());
    if(value_buf().size() and value_buf().shape_ != reward_buf().shape_)
      qlog_error("value shape (%s) should match reward shape (%s) or be zero.\n",
          value_buf().str().c_str(), reward_buf().str().c_str());
    if(qvest_buf().size() and qvest_buf().shape_ != reward_buf().shape_)
      qlog_error("qvest shape (%s) should match reward shape (%s) or be zero.\n",
          qvest_buf().str().c_str(), reward_buf().str().c_str());
    qassert(state_buf().is_c_contiguous());
    qassert(action_buf().is_c_contiguous());
    qassert(reward_buf().is_c_contiguous());
    qassert(prob_buf().is_c_contiguous());
    qassert(value_buf().is_c_contiguous());
    qassert(qvest_buf().is_c_contiguous());
    qassert(info_buf().is_c_contiguous());
    qassert(reward_buf().size() <= MAX_RWD_DIM);
  }

  const BufView& state_buf()  const { return view[0]; }
  const BufView& action_buf() const { return view[1]; }
  const BufView& reward_buf() const { return view[2]; }
  const BufView& prob_buf()   const { return view[3]; }
  const BufView& value_buf()  const { return view[4]; }
  const BufView& qvest_buf()  const { return view[5]; }
  const BufView& info_buf()   const { return view[6]; }

  void print_info(FILE * f = stderr) const
  {
    fprintf(f, "uuid:          0x%08x\n", uuid);
    fprintf(f, "state_buf:     %s\n",  state_buf().str().c_str());
    fprintf(f, "action_buf:    %s\n",  action_buf().str().c_str());
    fprintf(f, "reward_buf:    %s\n",  reward_buf().str().c_str());
    fprintf(f, "prob_buf:      %s\n",  prob_buf().str().c_str());
    fprintf(f, "value_buf:     %s\n",  value_buf().str().c_str());
    fprintf(f, "qvest_buf:     %s\n",  qvest_buf().str().c_str());
    fprintf(f, "info_buf:      %s\n",  info_buf().str().c_str());
    fprintf(f, "max_step:      %lu\n", max_step);
    fprintf(f, "max_episode:   %lu\n", max_episode);
    fprintf(f, "priority_e:    %lf\n", priority_exponent);
    fprintf(f, "mix_lambda:    %lf\n", mix_lambda);
    fprintf(f, "frame_stack:   %d\n",  frame_stack);
    fprintf(f, "multi_step:    %d\n",  multi_step);
    fprintf(f, "cache_size:    %u\n",  cache_size);
    fprintf(f, "reuse_cache:   %d\n",  reuse_cache);
    fprintf(f, "autosave_step: %d\n",  autosave_step);
    fprintf(f, "replace_data:  %d\n",  replace_data);
    fprintf(f, "entry::nbytes  %lu\n", DataEntry::nbytes(this));
    fprintf(f, "cache::nbytes  %lu\n", DataCache::nbytes(this));
    fprintf(f, "reqbuf_size    %d\n", reqbuf_size());
    fprintf(f, "repbuf_size    %d\n", repbuf_size());
    fprintf(f, "pushbuf_size   %d\n", pushbuf_size());
    fprintf(f, "discount_f:    [");
    for(int i=0; i<std::min<int>(reward_buf().size(), MAX_RWD_DIM); i++)
      fprintf(f, "%lf,", discount_factor[i]);
    fprintf(f, "]\n");
    fprintf(f, "reward_coeff:  [");
    for(int i=0; i<std::min<int>(reward_buf().size(), MAX_RWD_DIM); i++)
      fprintf(f, "%lf,", reward_coeff[i]);
    fprintf(f, "]\n");
    fprintf(f, "cache_flags:   [");
    for(const auto& each : cache_flags)
      fprintf(f, "%d,", each);
    fprintf(f, "]\n");
  }

  size_t num_episode() const { return episode.size(); }

protected:
  /**
   * Get offset for a new episode
   */
  long get_offset_for_new() const {
    long offset = 0;
    if(episode.size() > 0)
      offset = (episode.back().offset + episode.back().length) % max_step;
    return offset;
  }

  /**
   * Get length for a new episode
   */
  long get_length_for_new() const {
    long length = max_step;
    if(episode.size() > 0)
      length = (episode.front().offset - episode.back().offset - episode.back().length) % max_step;
    return length;
  }

  /**
   * Remove oldest episode from memory
   */
  void remove_oldest() {
    if(episode.empty())
      qthrow("episode is empty.");
    if(true) {
      // clear priority
      // TODO: Optimize this
      std::lock_guard<std::mutex> guard(prt_mutex);
      const auto& front = episode.front();
      for(int i=0; i<front.length; i++) {
        long idx = (front.offset + i) % max_step;
        prt.set_weight(idx, 0);
      }
    }
    // pop episode
    episode.pop();
  }

  /**
   * Update value in an episode
   */
  void update_value(long off, long len, bool is_finished) {
    assert(reward_buf().size() != 0);
    assert(value_buf().size() != 0);
    assert(qvest_buf().size() != 0);
    const auto& gamma = discount_factor;
    // Fill qvest
    for(int i=len-1; i>=0; i--) {
      long post = (off + i + 1) % max_step;
      long prev = (off + i) % max_step;
      auto post_value  = data[post].value(this).as_array<float>();
      auto prev_reward = data[prev].reward(this).as_array<float>();
      auto prev_qvest  = data[prev].qvest(this).as_array<float>();
      auto post_qvest  = data[post].qvest(this).as_array<float>();
      for(int j=0; j<(int)prev_reward.size(); j++) { // OPTIMIZE:
        // TD-Lambda
        if(i == len-1) // last
          prev_qvest[j] = (is_finished ? prev_reward[j] : data[prev].value(this).as_array<float>()[j]);
        else
          prev_qvest[j] = prev_reward[j] + mix_lambda * gamma[j] * post_qvest[j] + (1-mix_lambda) * gamma[j] * post_value[j];
      }
    }
  }
  void update_value(const Episode& epi) {
    return update_value(epi.offset, epi.length, true);
  }

  /**
   * Update weight in an episode.
   */
  void update_weight(long off, long len, float episodic_weight_multiplier) {
    assert(reward_buf().size() != 0);
    assert(value_buf().size() != 0);
    assert(qvest_buf().size() != 0);
    for(int i=0; i<std::min<int>(frame_stack-1, len); i++) {
      long idx = (off + i) % max_step;
      prt.set_weight(idx, 0.0);
    }
    for(int i=0; i<std::min<int>(multi_step, len); i++) {
      long idx = (off + len - 1 - i) % max_step;
      prt.set_weight(idx, 0.0);
    }
    for(int i=(frame_stack-1); i<len-multi_step; i++) {
      long idx = (off + i) % max_step;
      auto prev_value  = data[idx].value(this).as_array<float>();
      auto prev_qvest  = data[idx].qvest(this).as_array<float>();
      // R is computed with TD-lambda, while V is the original value in prediction
      float priority = 0;
      for(int i=0; i<(int)prev_qvest.size(); i++)
        priority += reward_coeff[i] * fabs(prev_qvest[i] - prev_value[i]);
      priority = pow(priority, priority_exponent) * episodic_weight_multiplier;
      prt.set_weight(idx, priority);
    }
  }
  void update_weight(const Episode& epi, float episodic_weight_multiplier = 1.0) {
    return update_weight(epi.offset, epi.length, episodic_weight_multiplier);
  }

public:
  /**
   * Prepare for a new episode
   */
  void new_episode() {
    if(stage == 10)
      qlog_warning("Renew an existing episode. Possible corruption of data.\n");
    new_offset = get_offset_for_new();
    new_length = 0;
    stage = 10;
  }

  /**
   * Make filled entries as a new episode
   */
  void close_episode(float episodic_weight_multiplier = 1.0, bool do_update_value = true, bool do_update_weight = true)
  {
    qassert(stage == 10);
    qassert(new_length <= get_length_for_new());
    episode.emplace(new_offset, new_length);
    if(do_update_value)
      update_value(episode.back());
    if(do_update_weight) {
      qassert(do_update_value);
      std::lock_guard<std::mutex> guard(prt_mutex);
      update_weight(episode.back(), episodic_weight_multiplier);
    }
    while(max_episode > 0 and episode.size() > max_episode)
      remove_oldest();
    incre_episode += 1;
    stage = 0;
  }

  /**
   * Add an entry to an existing episode.
   * Not thread-safe with identical epi_idx
   *
   * @param p_s     pointer to state (can be nullptr to be omitted)
   * @param p_a     pointer to action (can be nullptr to be omitted)
   * @param p_r     pointer to reward (can be nullptr to be omitted)
   * @param p_p     pointer to prob (can be nullptr to be omitted)
   * @param p_v     pointer to value (can be nullptr to be omitted)
   * @param p_i     pointer to info (can be nullptr to be omitted)
   */
  void add_entry(
      const void * p_s,
      const void * p_a,
      const void * p_r,
      const void * p_p,
      const void * p_v,
      const void * p_i)
  {
    qassert(stage == 10);
    // Check space
    while(!episode.empty() and new_length == get_length_for_new()) {
      //qlog_info("new_length: %ld, get_length_for_new: %ld\n", new_length, get_length_for_new());
      remove_oldest();
    }
    if(new_length == (long)max_step) {
      if(autosave_step <= 0)
        qthrow("ReplayMemory is full and autosave_step <= 0.");
      // Remove first autosave_step
      qassert(autosave_step < new_length);
      if(true) {
        // clear priority
        std::lock_guard<std::mutex> guard(prt_mutex);
        for(int i=0; i<autosave_step; i++) {
          long idx = (new_offset + i) % max_step;
          prt.set_weight(idx, 0);
        }
      }
      new_offset = (new_offset + autosave_step) % max_step;
      new_length -= autosave_step;
    }
    qassert(new_length < get_length_for_new());
    long idx = (new_offset + new_length) % max_step;
    auto& entry = data[idx];
    entry.from_memory(this, 0, p_s, p_a, p_r, p_p, p_v, nullptr, p_i);
    new_length += 1;
    incre_step += 1;
    qassert(new_length <= get_length_for_new());
    if(autosave_step > 0 and new_length % autosave_step == 0) {
      // autosave
      update_value(new_offset, new_length, false);
      if(true) {
        std::lock_guard<std::mutex> guard(prt_mutex);
        update_weight(new_offset, new_length, 1.0);
      }
    }
  }

  void add_entry(
      const BufView& s,
      const BufView& a,
      const BufView& r,
      const BufView& p,
      const BufView& v,
      const BufView& i)
  {
    // Check buffer info
    qassert(state_buf().is_consistent_with(s));
    qassert(action_buf().is_consistent_with(a));
    qassert(reward_buf().is_consistent_with(r));
    qassert(prob_buf().is_consistent_with(p));
    qassert(value_buf().is_consistent_with(v));
    qassert(info_buf().is_consistent_with(i));
    return add_entry(s.ptr_, a.ptr_, r.ptr_, p.ptr_, v.ptr_, i.ptr_);
  }

  /**
   * Get a data cache to be pushed to remote.
   * 
   * @return true iff success
   */
  bool get_cache(DataCache* p_cache, float& out_sum_weight, uint32_t& cache_idx)
  {
    if(prt.get_weight_sum() <= 0) {
      qlog_warning("%s() failed as local weight sum is %lf <= 0.\n", __func__, prt.get_weight_sum());
      return false;
    }
#ifndef LOCKFREE_GET_CACHE
    // Use global lock to get a snapshot of current PrtTree
    std::lock_guard<std::mutex> guard(prt_mutex);
#else
    qassert(replace_data); // LOCKFREE_GET_CACHE is only supported when replace_data is true.
#endif
    out_sum_weight = prt.get_weight_sum();
    for( ; cache_idx < cache_size; cache_idx ++) {
      size_t i = cache_idx;
      if(prt.get_weight_sum() <= 0) {
        qlog_warning("%s() failed as local weight sum is %lf <= 0. (current cache_idx: %d).\n",
            __func__, prt.get_weight_sum(), cache_idx);
        return false;
      }
      long idx;
      do {
        idx = prt.sample_index();
      } while (prt.get_weight(idx) <= 0);
      DataSample& s = p_cache->get(i, this);
      // add to batch
      for(int j=frame_stack-1; j>=0; j--) {
        auto& prev_entry = data[(idx - j) % max_step];
        prev_entry.to_memory(this, (frame_stack-1-j),
            s.prev_state(this).ptr_, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        if(j==0)
          prev_entry.to_memory(this, 0,
              nullptr,
              s.prev_action(this).ptr_,
              s.prev_reward(this).ptr_,
              s.prev_prob(this).ptr_,
              s.prev_value(this).ptr_,
              s.prev_qvest(this).ptr_,
              s.prev_info(this).ptr_);
      }
      for(int j=frame_stack-1; j>=0; j--) {
        auto& next_entry = data[(idx - j + multi_step) % max_step];
        next_entry.to_memory(this, (frame_stack-1-j),
            s.next_state(this).ptr_, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        if(j==0)
          next_entry.to_memory(this, 0,
              nullptr,
              s.next_action(this).ptr_,
              s.next_reward(this).ptr_,
              s.next_prob(this).ptr_,
              s.next_value(this).ptr_,
              s.next_qvest(this).ptr_,
              s.next_info(this).ptr_);
      }
      float w = prt.get_weight(idx);
#ifdef LOCKFREE_GET_CACHE
      // Lock free version, may get old data or new data.
      std::atomic_thread_fence(std::memory_order_acquire);
      if(w <= 0) { // redo
        i--;
        continue;
      }
#endif
      assert(w > 0);
      // When replace_data is false, you should make sure that the data generating speed is
      // faster than the data transmission (cache_size x push_cache() frequency) speed.
      if(not replace_data) {
        qassert(priority_exponent == 0.0); // sampling without replacement is not compatiable with priority_exponent.
        prt.set_weight(idx, 0);
      }
      s.entry_weight(this).as_array<float>()[0] = w;
    }
    return true;
  }

public:
  /**
   * Message protocal
   */
  class Message : public non_copyable
  {
  public:
    static const int ProtocalCache   = 31;  // PUSH, PULL
    static const int ProtocalSizes   = 32;  // REP,  REQ
    static const int ProtocalCounter = 33;  // PUSH, PULL
    static const int ProtocalLog     = 34;  // PUSH, PULL

    uint64_t version;    // version control
    int type;            // Message type
    int length;          // length of payload (in bytes)
    uint32_t sender;     // sender's uuid
    float sum_weight;    // sum of weight, used by ProtocalCache
    DataEntry payload;   // placeholder for actual payload. Note that sizeof(DataEntry) is 0.

    bool check_version() const {
      if(version != VERSION) {
        qlog_warning("Message version (%lu) != expected VERSION (%lu)\n", version, VERSION);
        return false;
      }
      return true;
    }

  };

  static int reqbuf_size()  { return sizeof(Message); }
  static int repbuf_size()  { return sizeof(Message) + N_VIEW * sizeof(BufView::Data) + sizeof(ReplayMemory); } // ProtocalSizes
  static int pushbuf_size() { return sizeof(Message) + 2 * sizeof(long); }  // ProtocalCounter

  enum Mode {
    Conn = 0,
    Bind = 1,
  };

};

/**
 * Default data type for ReplayMemory 
 */
typedef ReplayMemory RM;

