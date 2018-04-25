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

#define MAX_RWD_DIM (256)

/**
 * Distributed Replay Memory
 * 
 * Each instance will be created for each actor.
 * There may be multiple push threads to push batches to remote batch buffer simutanously.
 * However there will be only one thread to modify the local replay memory instance.
 * We hope this design will simplify the concurrent code to avoid conflict.
 */
template<typename _state_t, typename _action_t, typename _reward_t>
class ReplayMemory
{
public:
  //to be visible in derived class
  typedef _state_t state_t;   
  typedef _action_t action_t;
  typedef _reward_t reward_t;
  typedef float prob_t;        ///< probabilities are saved in float
  typedef _reward_t value_t;   ///< values should have the same type as reward

  /**
   * Memory structure of an entry. Does not own the memory.
   * (Actually, we should not have member variables, except for static ones)
   */
  class DataEntry : public non_copyable
  {
  public:
    static size_t nbytes(const ReplayMemory * p) {
      return p->state_size * sizeof(state_t)
        + p->action_size * sizeof(action_t)
        + p->reward_size * sizeof(reward_t)
        + p->prob_size * sizeof(prob_t)
        + p->value_size * sizeof(value_t);
    }

    ArrayView<state_t> state(const ReplayMemory * p) {
      char * head = reinterpret_cast<char*>(this);
      size_t offset = 0;
      return ArrayView<state_t>(head + offset, p->state_size);
    }

    ArrayView<action_t> action(const ReplayMemory * p) {
      char * head = reinterpret_cast<char*>(this);
      size_t offset = p->state_size * sizeof(state_t);
      return ArrayView<action_t>(head + offset, p->action_size);
    }

    ArrayView<reward_t> reward(const ReplayMemory * p) {
      char * head = reinterpret_cast<char*>(this);
      size_t offset = p->state_size * sizeof(state_t)
        + p->action_size * sizeof(action_t);
      return ArrayView<reward_t>(head + offset, p->reward_size);
    }

    ArrayView<prob_t> prob(const ReplayMemory * p) {
      char * head = reinterpret_cast<char*>(this);
      size_t offset = p->state_size * sizeof(state_t)
        + p->action_size * sizeof(action_t)
        + p->reward_size * sizeof(reward_t);
      return ArrayView<prob_t>(head + offset, p->prob_size);
    }

    ArrayView<value_t> value(const ReplayMemory * p) {
      char * head = reinterpret_cast<char*>(this);
      size_t offset = p->state_size * sizeof(state_t)
        + p->action_size * sizeof(action_t)
        + p->reward_size * sizeof(reward_t)
        + p->prob_size * sizeof(prob_t);
      return ArrayView<value_t>(head + offset, p->value_size);
    }

    /**
     * Copy from memory
     * @param index  denotes the offset in memory
     */
    void from_memory(const ReplayMemory * p, size_t index,
        const state_t  * p_s,
        const action_t * p_a,
        const reward_t * p_r,
        const prob_t   * p_p,
        const value_t  * p_v)
    {
      if(p_s)
        state(p).from_memory(p_s + index * p->state_size);
      if(p_a)
        action(p).from_memory(p_a + index * p->action_size);
      if(p_r)
        reward(p).from_memory(p_r + index * p->reward_size);
      if(p_p)
        prob(p).from_memory(p_p + index * p->prob_size);
      if(p_v)
        value(p).from_memory(p_v + index * p->value_size);
    }

    /**
     * Copy to memory
     * @param index  denotes the offset in memory (see ReplayMemory::get_batch())
     */
    void to_memory(const ReplayMemory * p, size_t index,
        state_t  * p_s,
        action_t * p_a,
        reward_t * p_r,
        prob_t   * p_p,
        value_t  * p_v)
    {
      if(p_s)
        state(p).to_memory(p_s + index * p->state_size);
      if(p_a)
        action(p).to_memory(p_a + index * p->action_size);
      if(p_r)
        reward(p).to_memory(p_r + index * p->reward_size);
      if(p_p)
        prob(p).to_memory(p_p + index * p->prob_size);
      if(p_v)
        value(p).to_memory(p_v + index * p->value_size);
    }

  };

  /**
   * Memory structure of a sample. Does not own the memory.
   * This is used in constructing cache.
   */
  class DataSample : public non_copyable
  {
    template<int N>
    static size_t ceil(size_t a) {
      return (a+N-1) / N * N;
    }
  public:
    static size_t prev_state_offset(const ReplayMemory * p) {
      return 0;
    }
    static size_t prev_action_offset(const ReplayMemory * p) {
      return ceil<4>(prev_state_offset(p) + (p->cache_flags[0] ? p->frame_stack * p->state_size * sizeof(state_t) : 0));
    }
    static size_t prev_reward_offset(const ReplayMemory * p) {
      return prev_action_offset(p) + (p->cache_flags[1] ? p->action_size * sizeof(action_t) : 0);
    }
    static size_t prev_prob_offset(const ReplayMemory * p) {
      return prev_reward_offset(p) + (p->cache_flags[2] ? p->reward_size * sizeof(reward_t) : 0);
    }
    static size_t prev_value_offset(const ReplayMemory * p) {
      return prev_prob_offset(p)   + (p->cache_flags[3] ? p->prob_size   * sizeof(prob_t)   : 0);
    }
    static size_t next_state_offset(const ReplayMemory * p) {
      return prev_value_offset(p)  + (p->cache_flags[4] ? p->value_size  * sizeof(value_t)  : 0);
    }
    static size_t next_action_offset(const ReplayMemory * p) {
      return ceil<4>(next_state_offset(p) + (p->cache_flags[5] ? p->frame_stack * p->state_size * sizeof(state_t) : 0));
    }
    static size_t next_reward_offset(const ReplayMemory * p) {
      return next_action_offset(p) + (p->cache_flags[6] ? p->action_size * sizeof(action_t) : 0);
    }
    static size_t next_prob_offset(const ReplayMemory * p) {
      return next_reward_offset(p) + (p->cache_flags[7] ? p->reward_size * sizeof(reward_t) : 0);
    }
    static size_t next_value_offset(const ReplayMemory * p) {
      return next_prob_offset(p)   + (p->cache_flags[8] ? p->prob_size   * sizeof(prob_t)   : 0);
    }
    static size_t entry_weight_offset(const ReplayMemory * p) {
      return next_value_offset(p)  + (p->cache_flags[9] ? p->value_size  * sizeof(value_t)  : 0);
    }
    static size_t nbytes(const ReplayMemory * p) {
      return entry_weight_offset(p)+ sizeof(float);
    }

    ArrayView<state_t> prev_state(const ReplayMemory * p) {
      char * hd = p->cache_flags[0] ? (char*)this + prev_state_offset(p) : nullptr;
      size_t sz = p->cache_flags[0] ? p->frame_stack * p->state_size : 0;
      return ArrayView<state_t>(hd, sz);
    }
    ArrayView<action_t> prev_action(const ReplayMemory * p) {
      char * hd = p->cache_flags[1] ? (char*)this + prev_action_offset(p) : nullptr;
      size_t sz = p->cache_flags[1] ? p->action_size : 0;
      return ArrayView<action_t>(hd, sz);
    }
    ArrayView<reward_t> prev_reward(const ReplayMemory * p) {
      char * hd = p->cache_flags[2] ? (char*)this + prev_reward_offset(p) : nullptr;
      size_t sz = p->cache_flags[2] ? p->reward_size : 0;
      return ArrayView<reward_t>(hd, sz);
    }
    ArrayView<prob_t> prev_prob(const ReplayMemory * p) {
      char * hd = p->cache_flags[3] ? (char*)this + prev_prob_offset(p) : nullptr;
      size_t sz = p->cache_flags[3] ? p->prob_size : 0;
      return ArrayView<prob_t>(hd, sz);
    }
    ArrayView<value_t> prev_value(const ReplayMemory * p) {
      char * hd = p->cache_flags[4] ? (char*)this + prev_value_offset(p) : nullptr;
      size_t sz = p->cache_flags[4] ? p->value_size : 0;
      return ArrayView<value_t>(hd, sz);
    }
    ArrayView<state_t> next_state(const ReplayMemory * p) {
      char * hd = p->cache_flags[5] ? (char*)this + next_state_offset(p) : nullptr;
      size_t sz = p->cache_flags[5] ? p->frame_stack * p->state_size : 0;
      return ArrayView<state_t>(hd, sz);
    }
    ArrayView<action_t> next_action(const ReplayMemory * p) {
      char * hd = p->cache_flags[6] ? (char*)this + next_action_offset(p) : nullptr;
      size_t sz = p->cache_flags[6] ? p->action_size : 0;
      return ArrayView<action_t>(hd, sz);
    }
    ArrayView<reward_t> next_reward(const ReplayMemory * p) {
      char * hd = p->cache_flags[7] ? (char*)this + next_reward_offset(p) : nullptr;
      size_t sz = p->cache_flags[7] ? p->reward_size : 0;
      return ArrayView<reward_t>(hd, sz);
    }
    ArrayView<prob_t> next_prob(const ReplayMemory * p) {
      char * hd = p->cache_flags[8] ? (char*)this + next_prob_offset(p) : nullptr;
      size_t sz = p->cache_flags[8] ? p->prob_size : 0;
      return ArrayView<prob_t>(hd, sz);
    }
    ArrayView<value_t> next_value(const ReplayMemory * p) {
      char * hd = p->cache_flags[9] ? (char*)this + next_value_offset(p) : nullptr;
      size_t sz = p->cache_flags[9] ? p->value_size : 0;
      return ArrayView<value_t>(hd, sz);
    }
    ArrayView<float> entry_weight(const ReplayMemory * p) {
      char * hd = (char*)this + entry_weight_offset(p);
      return ArrayView<float_t>(hd, 1);
    }

    void print_first(const ReplayMemory * p, FILE * f = stderr) {
      if(p->cache_flags[0])
        fprintf(f, "%u ",  prev_state(p)[0]);
      if(p->cache_flags[1])
        fprintf(f, "%lf ", prev_action(p)[0]);
      if(p->cache_flags[2])
        fprintf(f, "%lf ", prev_reward(p)[0]);
      if(p->cache_flags[3])
        fprintf(f, "%lf ", prev_prob(p)[0]);
      if(p->cache_flags[4])
        fprintf(f, "%lf ", prev_value(p)[0]);
      if(p->cache_flags[5])
        fprintf(f, "%u ",  next_state(p)[0]);
      if(p->cache_flags[6])
        fprintf(f, "%lf ", next_action(p)[0]);
      if(p->cache_flags[7])
        fprintf(f, "%lf ", next_reward(p)[0]);
      if(p->cache_flags[8])
        fprintf(f, "%lf ", next_prob(p)[0]);
      if(p->cache_flags[9])
        fprintf(f, "%lf ", next_value(p)[0]);
      fprintf(f, "%lf\n",entry_weight(p)[0]);
    }

    void to_memory(const ReplayMemory * p, int idx,
        state_t * prev_s, action_t * prev_a, reward_t * prev_r, prob_t * prev_p, value_t * prev_v,
        state_t * next_s, action_t * next_a, reward_t * next_r, prob_t * next_p, value_t * next_v,
        float * entry_w) {
      if(prev_s and p->cache_flags[0]) {
        auto av = prev_state(p);
        av.to_memory(prev_s + idx * av.size());
      }
      if(prev_a and p->cache_flags[1]) {
        auto av = prev_action(p);
        av.to_memory(prev_a + idx * av.size());
      }
      if(prev_r and p->cache_flags[2]) {
        auto av = prev_reward(p);
        av.to_memory(prev_r + idx * av.size());
      }
      if(prev_p and p->cache_flags[3]) {
        auto av = prev_prob(p);
        av.to_memory(prev_p + idx * av.size());
      }
      if(prev_v and p->cache_flags[4]) {
        auto av = prev_value(p);
        av.to_memory(prev_v + idx * av.size());
      }
      if(next_s and p->cache_flags[5]) {
        auto av = next_state(p);
        av.to_memory(next_s + idx * av.size());
      }
      if(next_a and p->cache_flags[6]) {
        auto av = next_action(p);
        av.to_memory(next_a + idx * av.size());
      }
      if(next_r and p->cache_flags[7]) {
        auto av = next_reward(p);
        av.to_memory(next_r + idx * av.size());
      }
      if(next_p and p->cache_flags[8]) {
        auto av = next_prob(p);
        av.to_memory(next_p + idx * av.size());
      }
      if(next_v and p->cache_flags[9]) {
        auto av = next_value(p);
        av.to_memory(next_v + idx * av.size());
      }
      if(entry_w) {
        auto av = entry_weight(p);
        av.to_memory(entry_w + idx * av.size());
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
  typedef DataEntry T;

  class Episode {
  public:
    const long offset;
    const long length;
    Episode(long o, long l) : offset{o}, length{l} {}
  };

public:
  const size_t state_size;           ///< num of state
  const size_t action_size;          ///< num of action
  const size_t reward_size;          ///< num of reward
  const size_t prob_size;            ///< num of base probability
  const size_t value_size;           ///< num of discounted future reward sum

  const size_t entry_size;           ///< size of each entry (in bytes)
  const size_t capacity;             ///< max number of steps can be stored

  float discount_factor;             ///< discount factor for calculate R with rewards
  float priority_exponent;           ///< exponent of priority term (default 0.5)
  float td_lambda;                   ///< mixture factor for TD-lambda
  int frame_stack;                   ///< Number of frames stacked for each state (default 1)
  int multi_step;                    ///< Number of steps between prev and next (default 1)
  int cache_size;                    ///< Cache size, number of sample in a cache
  float rwd_coeff[MAX_RWD_DIM];      ///< rwd coefficient
  uint8_t cache_flags[10];           ///< Whether we should collect prev s,a,r,p,v and next s,a,r,p,v in caches

  size_t uuid;                       ///< uuid for current instance

protected:
  Vector<T> data;                    ///< Actual data of all samples
  std::queue<Episode> episode;       ///< Episodes
  PrtTree prt;                       ///< Priority tree for sampling
  std::mutex prt_mutex;              ///< mutex for priority tree
public:
  qlib::RNG * rng;                   ///< Random Number Generator

  long new_offset;                   ///< offset for new episode
  long new_length;                   ///< current length of new episode

public:
  /**
   * Construct a new replay memory
   * @param e_size   size of a single entry (in bytes)
   * @param max_epi  max number of episodes kept in the memory
   */
  ReplayMemory(size_t s_size,
      size_t a_size,
      size_t r_size,
      size_t p_size,
      size_t v_size,
      size_t capa,
      qlib::RNG * prt_rng) :
    state_size{s_size},
    action_size{a_size},
    reward_size{r_size},
    prob_size{p_size},
    value_size{v_size},
    entry_size{T::nbytes(this)},
    capacity{capa},
    cache_size{0},
    uuid{0},
    data{entry_size},
    prt{prt_rng, static_cast<int>(capacity)},
    rng{prt_rng}
  {
    data.reserve(capacity);
    uuid = qlib::get_nsec();
    for(auto&& each : rwd_coeff)
      each = 1.0f;
    for(auto&& each : cache_flags)
      each = 1;
  }

  void print_info(FILE * f = stderr) const
  {
    fprintf(f, "uuid:          0x%lx\n", uuid);
    fprintf(f, "state_size:    %lu\n", state_size);
    fprintf(f, "action_size:   %lu\n", action_size);
    fprintf(f, "reward_size:   %lu\n", reward_size);
    fprintf(f, "prob_size:     %lu\n", prob_size);
    fprintf(f, "value_size:    %lu\n", value_size);
    fprintf(f, "capacity:      %lu\n", capacity);
    fprintf(f, "discount_f:    %lf\n", discount_factor);
    fprintf(f, "priority_e:    %lf\n", priority_exponent);
    fprintf(f, "td_lambda:     %lf\n", td_lambda);
    fprintf(f, "frame_stack:   %d\n",  frame_stack);
    fprintf(f, "multi_step:    %d\n",  multi_step);
    fprintf(f, "cache_size:    %d\n",  cache_size);
    fprintf(f, "cache::nbytes  %lu\n", DataCache::nbytes(this));
    fprintf(f, "sizes: %lu %lu %lu %lu %lu, %lu %lu %lu %lu %lu, %lu: %ld\n",
        DataSample::prev_action_offset(this) - DataSample::prev_state_offset(this),
        DataSample::prev_reward_offset(this) - DataSample::prev_action_offset(this),
        DataSample::prev_prob_offset(this) - DataSample::prev_reward_offset(this),
        DataSample::prev_value_offset(this) - DataSample::prev_prob_offset(this),
        DataSample::next_state_offset(this) - DataSample::prev_value_offset(this),
        DataSample::next_action_offset(this) - DataSample::next_state_offset(this),
        DataSample::next_reward_offset(this) - DataSample::next_action_offset(this),
        DataSample::next_prob_offset(this) - DataSample::next_reward_offset(this),
        DataSample::next_value_offset(this) - DataSample::next_prob_offset(this),
        DataSample::entry_weight_offset(this) - DataSample::next_value_offset(this),
        DataSample::nbytes(this) - DataSample::entry_weight_offset(this),
        DataSample::nbytes(this));
  }

  size_t num_episode() const { return episode.size(); }

protected:
  /**
   * Get offset for a new episode
   */
  long get_offset_for_new() const {
    long offset = 0;
    if(episode.size() > 0)
      offset = (episode.back().offset + episode.back().length) % capacity;
    return offset;
  }

  /**
   * Get length for a new episode
   */
  long get_length_for_new() const {
    long length = capacity;
    if(episode.size() > 0)
      length = (episode.front().offset - episode.back().offset - episode.back().length) % capacity;
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
        long idx = (front.offset + i) % capacity;
        prt.set_weight(idx, 0);
      }
    }
    // pop episode
    episode.pop();
  }

  /**
   * Update value in an episode
   */
  void update_value(const Episode& epi) {
    if(value_size==0)
      return;
    // TODO: different discount_factor for different dimension of reward.
    const auto& gamma = discount_factor;
    for(int i=epi.length-1; i>=0; i--) {
      long post = (epi.offset + i + 1) % capacity;
      long prev = (epi.offset + i) % capacity;
      auto post_reward = data[post].reward(this);
      auto post_value  = data[post].value(this);
      auto prev_reward = data[prev].reward(this);
      for(int j=0; j<prev_reward.size(); j++) { // OPTIMIZE:
        prev_reward[j] += td_lambda * gamma * post_reward[j] + (1-td_lambda) * gamma * post_value[j];
      }
    }
  }

  /**
   * Update weight in an episode, but not commit to prt.
   */
  void update_weight(const Episode& epi) {
    // TODO: Optimize this
    for(int i=0; i<std::min<int>(frame_stack-1, epi.length); i++) {
      long idx = (epi.offset + i) % capacity;
      prt.set_weight(idx, 0.0);
    }
    for(int i=0; i<std::min<int>(multi_step, epi.length); i++) {
      long idx = (epi.offset + epi.length - 1 - i) % capacity;
      prt.set_weight(idx, 0.0);
    }
    for(int i=(frame_stack-1); i<epi.length-multi_step; i++) {
      long idx = (epi.offset + i) % capacity;
      auto& prev = data[idx];
      // R is computed with TD-lambda, while V is the original value in prediction
      float priority = 0;
      for(int i=0; i<prev.reward(this).size(); i++)
        priority += rwd_coeff[i] * fabs(prev.reward(this)[i] - prev.value(this)[i]);
      priority = pow(priority, priority_exponent);
      prt.set_weight(idx, prt.get_weight(idx) * priority);
    }
  }

public:
  /**
   * Prepare for a new episode
   */
  void new_episode() {
    new_offset = get_offset_for_new();
    new_length = 0;
  }

  /**
   * Make filled entries as a new episode
   */
  void close_episode(bool do_update_value = true, bool do_update_weight = true)
  {
    qassert(new_length <= get_length_for_new());
    episode.emplace(new_offset, new_length);
    if(do_update_value)
      update_value(episode.back());
    if(do_update_weight) {
      std::lock_guard<std::mutex> guard(prt_mutex);
      update_weight(episode.back());
    }
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
   * @param weight  sample weight (priority)
   */
  void add_entry(
      const state_t  * p_s,
      const action_t * p_a,
      const reward_t * p_r,
      const prob_t   * p_p,
      const value_t  * p_v,
      float weight)
  {
    if(!episode.empty() and new_offset != get_offset_for_new())
      qthrow("Please call new_episode() before add_entry().");
    // Check space
    while(new_length == get_length_for_new()) {
      //qlog_info("new_length: %ld, get_length_for_new: %ld\n", new_length, get_length_for_new());
      remove_oldest();
    }
    long idx = (new_offset + new_length) % capacity;
    auto& entry = data[idx];
    entry.from_memory(this, 0, p_s, p_a, p_r, p_p, p_v);
    prt.set_weight_without_update(idx, weight); // will be update by update_weight() later in close_episode()
    new_length += 1;
  }

  /**
   * Get a data cache to be pushed to remote.
   * 
   * @return true iff success
   */
  bool get_cache(DataCache* p_cache, float& out_sum_weight)
  {
    if(episode.size() == 0) {
      qlog_warning("%s() failed as the ReplayMemory is empty.\n", __func__);
      return false;
    }
    for(size_t i=0; i<cache_size; i++) {
      long idx = prt.sample_index(); 
      DataSample& s = p_cache->get(i, this);
      // add to batch
      for(int j=frame_stack-1; j>=0; j--) {
        auto& prev_entry = data[(idx - j) % capacity];
        prev_entry.to_memory(this, (frame_stack-1-j),
            s.prev_state(this).data(), nullptr, nullptr, nullptr, nullptr);
        if(j==0)
          prev_entry.to_memory(this, 0,
              nullptr, s.prev_action(this).data(), s.prev_reward(this).data(),
              s.prev_prob(this).data(), s.prev_value(this).data());
      }
      for(int j=frame_stack-1; j>=0; j--) {
        auto& next_entry = data[(idx - j + multi_step) % capacity];
        next_entry.to_memory(this, (frame_stack-1-j),
            s.next_state(this).data(), nullptr, nullptr, nullptr, nullptr);
        if(j==0)
          next_entry.to_memory(this, 0,
              nullptr, s.next_action(this).data(), s.next_reward(this).data(),
              s.next_prob(this).data(), s.next_value(this).data());
      }
      if(s.entry_weight(this).data())
        s.entry_weight(this)[0] = prt.get_weight(idx);
    }
    out_sum_weight = prt.get_weight_sum();
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
    static const int ProtocalCounter = 33;  // REP,  REQ

    int type;            // Message type
    int length;          // length of payload (in bytes)
    uint64_t sender;     // sender's uuid
    float sum_weight;    // sum of weight, used by ProtocalCache
    DataEntry payload;   // placeholder for actual payload. Note that sizeof(DataEntry) is 0.
  };

  static int reqbuf_size()  { return sizeof(Message) + sizeof(int); }  // for ProtocalCounter
  static int repbuf_size()  { return sizeof(Message) + sizeof(ReplayMemory); } // for ProtocalSizes
  static int pushbuf_size() { return sizeof(Message); }

  enum Mode {
    Conn = 0,
    Bind = 1,
  };

};

