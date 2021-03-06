#pragma once

#include <cstdint>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include "qlog.hpp"
#include "qtime.hpp"
#include "array_view.hpp"
#include "vector.hpp"
#include "qrand.hpp"
//#include <signal.h>
#include "utils.hpp" // non_copyable
#include <thread>
#include <chrono>
#include "prt_tree.hpp"
#include "buffer_view.hpp"

#define VERSION (20181116ul)

#define EPS (1e-6)

/**
 * Distributed Replay Memory
 */
class ReplayMemory
{
public:
  class DataEntry : public non_copyable
  {
  protected:
    BufView get(int i, const ReplayMemory * p) {
      assert(0<= i and i < static_cast<int>(p->view.size()));
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
    void * data() { return (void*)(this); }
    BufView reward(const ReplayMemory * p) { return get(0, p); }
    BufView prob (const ReplayMemory * p)  { return get(1, p); }
    BufView value(const ReplayMemory * p)  { return get(2, p); }
    BufView qvest(const ReplayMemory * p)  { return get(3, p); }

    static size_t nbytes(const ReplayMemory * p) {
      DataEntry * dummy = (DataEntry*)nullptr;
      auto last = dummy->get(p->view.size()-1, p);
      return ((char*)last.ptr_ - (char*)dummy) + last.nbytes();
    }
  };

  class Episode {
  public:
    long offset;
    long length;
    Episode(long o, long l) : offset{o}, length{l} {}
  };

  class MemorySlot : public non_copyable
  {
  public:
    Vector<DataEntry> data;            ///< Actual data of all samples
    std::deque<Episode> episode;       ///< Episodes
    PrtTree prt;                       ///< Priority tree for sampling
    const unsigned self_index;         ///< slot_index in ReplayMemory
    std::mutex slot_mutex;             ///< mutex for this memory slot
    long new_offset;                   ///< offset for new episode
    long new_length;                   ///< current length of new episode
    long cur_step;                     ///< current length since episode beginning
    long step_count;                   ///< total steps written ever for this slot

    /**
     * MemorySlot for storing raw episodes. It works like a ring-buffer.
     * @param entry_size:  size of an entry
     * @param max_step:    number of entries can be stored in this MemorySlot
     * @param prt_rng:     RNG for the priority tree
     */
    MemorySlot(unsigned index, unsigned entry_size, unsigned max_step, qlib::RNG* prt_rng):
      data{entry_size},
      episode{},
      prt{prt_rng, static_cast<int>(max_step)},
      self_index{index},
      new_offset{0},
      new_length{0},
      cur_step{0},
      step_count{0}
    {
      data.resize(max_step);
    }

    /**
     * Access the ith entry
     * @param idx  index of entry (should be less than size())
     */
    DataEntry& operator[](size_t idx) const {
      return data[idx];
    }

    /**
     * Get offset for a new episode
     */
    long get_new_offset() const {
      long offset = 0;
      if(episode.size() > 0)
        offset = (episode.back().offset + episode.back().length) % data.capacity();
      return offset;
    }

    /**
     * Get length(capacity) for a new episode
     */
    long get_new_length() const {
      long length = data.capacity();
      if(episode.size() > 0)
        length = (episode.front().offset - episode.back().offset - episode.back().length) % data.capacity();
      return length;
    }

    /**
     * Find the offset in the episode belonged
     */
    long get_offset_in_episode(long idx) const {
      // cur
      long offset = (idx - new_offset + data.capacity()) % data.capacity(); 
      if(offset < new_length)
        return offset;
      for(const auto& epi : episode) {
        offset = (idx - epi.offset + data.capacity()) % data.capacity();
        if(offset < epi.length)
          return offset;
      }
      qlog_warning("Invalid sample index(%ld): current offset(%ld), length(%ld)\n", idx, new_offset, new_length);
      print_episodes();
      qassert(false and "Invalid sample index.");
      return -1;
    }

    /**
     * Print episodes with qlog_info()
     */
    void print_episodes() const {
      for(const auto& epi : episode) {
        qlog_info("episode: (%ld, %ld)\n", epi.offset, epi.length);
      }
    }

    /**
     * Remove oldest episode from memory
     */
    void remove_oldest(ReplayMemory * prm) {
      if(episode.empty())
        qthrow("episode is empty.");
      // clear priority
      const auto& front = episode.front();
      clear_priority(front.offset, front.length);
      // pop episode
      episode.pop_front();
    }

    /**
     * Update value and weight in an episode
     */
    void update(ReplayMemory * prm, long off, long len, bool is_finished, long min_traceback_length) {
      assert(prm->reward_buf().size() != 0);
      assert(prm->value_buf().size() != 0);
      assert(prm->qvest_buf().size() != 0);
      qassert(prm->reward_coeff.size() == prm->reward_buf().size()); // reward_coeff size mismatch
      const auto& gamma = prm->discount_factor;
      double local_diff = 0;
      double local_ma = prm->ma_sqa;
      float c = sqrt(prm->ma_sqa/2);
      // Fill qvest
      for(int i=len-1; i>=0; i--) {
        long post = (off + i + 1) % data.capacity();
        long prev = (off + i) % data.capacity();
        auto prev_value  = data[prev].value(prm).as_array<float>();
        auto post_value  = data[post].value(prm).as_array<float>();
        auto prev_reward = data[prev].reward(prm).as_array<float>();
        auto prev_qvest  = data[prev].qvest(prm).as_array<float>();
        auto post_qvest  = data[post].qvest(prm).as_array<float>();
        float sum_diff = 0;
        // R is computed with TD-lambda, while V is the original value in prediction
        float priority = 0;
        for(int j=0; j<(int)prev_reward.size(); j++) { // OPTIMIZE:
          // TD-Lambda
          float qv;
          if(i == len-1) // last
            qv = (is_finished ? prev_reward[j] : data[prev].value(prm).as_array<float>()[j]);
          else
            qv = prev_reward[j] + prm->mix_lambda * gamma[j] * post_qvest[j] +
              (1-prm->mix_lambda) * gamma[j] * post_value[j];
          sum_diff += prm->reward_coeff[j] * std::abs(prev_qvest[j] -  qv);
          prev_qvest[j] = qv;
          // update weight
          priority += prm->reward_coeff[j] * (prev_qvest[j] - prev_value[j]);
        }
        // stop when the difference is neglectable
        if(len - i > min_traceback_length and sum_diff < prm->traceback_threshold)
          break; 
        // priority is (R-V), now update ma_sqa
        local_diff += (priority * priority - local_ma);
        // calculate real priority
        priority = pow(fabs(priority/c), prm->priority_exponent);
        prt.set_weight(prev, priority);
      }
      prm->ma_sqa += 1e-8 * local_diff;
      prm->ma_sqa = std::max<float>(prm->ma_sqa, EPS);
    }

    void clear_priority(long offset, long len) {
      // TODO: Optimize this
      qassert(len >= 0);
      for(int i=0; i<len; i++) {
        long idx = (offset + i) % data.capacity();
        prt.set_weight(idx, 0);
      }
    }

    void clear(ReplayMemory * prm) {
      std::lock_guard<std::mutex> guard(slot_mutex);
      size_t cleared = 0;
      clear_priority(new_offset, new_length);
      cleared += cur_step;
      while(not episode.empty()) {
        const auto& front = episode.front();
        clear_priority(front.offset, front.length);
        episode.pop_front();
        cleared += front.length;
      }
      if(true) {
        std::lock_guard<std::mutex> guard(prm->rem_mutex);
        prm->total_steps -= cleared;
        prm->slot_prt.set_weight(self_index, prt.get_weight_sum());
      }
      new_offset = get_new_offset();
      new_length = 0;
      cur_step = 0;
    }

    void discard_data(ReplayMemory * prm) {
      std::lock_guard<std::mutex> guard(slot_mutex);
      qlog_warning("[slot:%d] Discard unfinished episode (offset:%ld, len:%ld, step:%ld).\n",
          self_index, new_offset, new_length, cur_step);
      clear_priority(new_offset, new_length);
      if(true) {
        std::lock_guard<std::mutex> guard(prm->rem_mutex);
        prm->total_steps -= cur_step;
        prm->slot_prt.set_weight(self_index, prt.get_weight_sum());
      }
      new_offset = get_new_offset();
      new_length = 0;
      cur_step = 0;
    }

    bool add_data(ReplayMemory * prm, void * raw_data, uint32_t start_step, uint32_t n_step, bool is_episode_end) {
      std::lock_guard<std::mutex> guard(slot_mutex);
      if(start_step != cur_step) { // Data lost, usually caused by out of order data
        qlog_warning("[slot:%d] Check failed: start_step:%u n_step:%u is_end:%d cur_step:%ld length:%ld offset:%ld\n",
            self_index, start_step, n_step, is_episode_end, cur_step, new_length, new_offset);
        return false;
      }
      qassert(start_step == cur_step);
      // Check space
      while(!episode.empty() and (new_length + n_step) > get_new_length()) {
        auto& front = episode.front();
        if(prm->do_padding) {
          clear_priority(front.offset, front.length);
          episode.pop_front();
        } else {
          long len = new_length + n_step - get_new_length();
          if(len < front.length) {
            front.offset += len;
            front.length -= len;
          }
          else
            episode.pop_front();
        }
      }
      // Write to memory
      char * mem_data = static_cast<char*>(raw_data);
      long idx = (new_offset + new_length) % data.capacity();
      long len_to_write = n_step;
      while(len_to_write > 0) {
        long len = std::min<long>(len_to_write, data.capacity() - idx);
        memcpy(&data[idx], mem_data, len * data.entry_size);
        idx = (idx + len) % data.capacity();
        mem_data += len * data.entry_size;
        len_to_write -= len;
      }
      // Update variables
      if(prm->do_padding and new_length + n_step > static_cast<long>(data.capacity()))
        qlog_warning("Due to insufficient capacity of memory slot, the beginning samples for current episode is overwritten, "
            "which will cause incorrect padding.\n");
      new_length = std::min<long>(new_length + n_step, data.capacity());
      new_offset = (idx - new_length + data.capacity()) % data.capacity();
      cur_step += n_step;
      // Update value and weight
      update(prm, new_offset, new_length, is_episode_end, n_step);
      if(not prm->do_padding) // Leave first prm->rollout_len-1 entry's weight as zero
        clear_priority(new_offset, std::max<long>(prm->rollout_len - 1 - step_count, 0));
      clear_priority(idx, prm->rollout_len - 1);
      qlog_debug("new_offset: %ld, new_length: %ld, idx: %ld\n", new_offset, new_length, idx);
      qlog_debug("prt.get_weight_sum(): %le\n", prt.get_weight_sum());
      if(is_episode_end) {
        step_count += cur_step;
        episode.emplace_back(new_offset, new_length);
        while(prm->max_episode > 0 and episode.size() > prm->max_episode)
          remove_oldest(prm);
        new_offset = get_new_offset();
        new_length = 0;
        cur_step = 0;
      }
      if(true) {
        std::lock_guard<std::mutex> guard(prm->rem_mutex);
        prm->total_steps += n_step;
        if(is_episode_end)
          prm->total_episodes += 1;
        prm->slot_prt.set_weight(self_index, prt.get_weight_sum());
      }
      return true;
    }

  };

public:
  std::deque<BufView> view;          ///< buffer view of r, p, v, q, *

  const size_t entry_size;           ///< size of each entry (in bytes)
  const size_t max_step;             ///< max number of steps can be stored

  size_t max_episode;                ///< max number of stored episodes allowed for each slot (0 for not checking)
  float priority_exponent;           ///< exponent of priority term (default 0.0)
  float mix_lambda;                  ///< mixture factor for computing multi-step return
  unsigned rollout_len;              ///< rollout length
  bool do_padding;                   ///< padding before first entry
  float priority_decay;              ///< decay factor for sample priority
  float traceback_threshold;         ///< traceback stopping threshold
  std::vector<float> discount_factor;///< discount factor for calculate R with rewards
  std::vector<float> reward_coeff;   ///< reward coefficient

  std::string uuid;                  ///< uuid for current instance

protected:
  std::deque<MemorySlot> slots;      ///< memory slots
  PrtTree slot_prt;                  ///< Priority tree for sampling a slot
  std::mutex rem_mutex;              ///< mutex for the replay memory
  std::condition_variable not_empty; ///< condition variable for not empty
public:
  qlib::RNG * rng;                   ///< random number generator
  float ma_sqa;                      ///< moving average estimation of squared advantage

  size_t total_episodes;             ///< total number of episodes
  size_t total_steps;                ///< total number of steps

public:
  /**
   * Construct a new replay memory
   * @param m_step      max number of steps in a memory slot
   * @param n_slot      number of slots in this replay memory
   */
  ReplayMemory(const std::deque<BufView>& vw,
      size_t m_step,
      size_t n_slot,
      qlib::RNG * prt_rng):
    view{vw},
    entry_size{DataEntry::nbytes(this)},
    max_step{m_step},
    max_episode{0},
    priority_exponent{0.0},
    mix_lambda{1.0},
    rollout_len{1},
    do_padding{false},
    priority_decay{1.0},
    traceback_threshold{1e-3},
    slots{},
    slot_prt{prt_rng, static_cast<int>(n_slot)},
    rng{prt_rng},
    ma_sqa{1.0},
    total_episodes{0},
    total_steps{0}
  {
    for(size_t i=0; i<n_slot; i++) {
      slots.emplace_back(i, entry_size, max_step, rng);
    }
    //
    check();
    discount_factor.resize(reward_buf().size());
    for(auto&& each : discount_factor)
      each = 1.0f;
    reward_coeff.resize(reward_buf().size());
    for(auto&& each : reward_coeff)
      each = 1.0f;
    // Get uuid
    char buf[64];
    snprintf(buf, 64, "RM::IP:%s,PID:%u,Ptr:%p", get_host_ip("8.8.8.8", 53).c_str(), getpid(), this);
    uuid = std::string(buf);
  }

  void check() const {
    qassert(prob_buf().format_   == "f");
    qassert(reward_buf().format_ == "f");
    qassert(value_buf().format_  == "f");
    qassert(qvest_buf().format_  == "f");
    if(reward_buf().ndim() > 1)
      qlog_error("reward (%s) should be a single number or a 1-dim array.\n",
          reward_buf().str().c_str());
    if(value_buf().size() and value_buf().shape_ != reward_buf().shape_)
      qlog_error("value shape (%s) should match reward shape (%s) or be zero.\n",
          value_buf().str().c_str(), reward_buf().str().c_str());
    if(qvest_buf().size() and qvest_buf().shape_ != reward_buf().shape_)
      qlog_error("qvest shape (%s) should match reward shape (%s) or be zero.\n",
          qvest_buf().str().c_str(), reward_buf().str().c_str());
    qassert(reward_buf().is_c_contiguous());
    qassert(prob_buf().is_c_contiguous());
    qassert(value_buf().is_c_contiguous());
    qassert(qvest_buf().is_c_contiguous());
  }

  const BufView& reward_buf() const { return view[0]; }
  const BufView& prob_buf()   const { return view[1]; }
  const BufView& value_buf()  const { return view[2]; }
  const BufView& qvest_buf()  const { return view[3]; }

  void print_info(FILE * f = stderr) const
  {
    fprintf(f, "uuid:          %s\n",  uuid.c_str());
    for(unsigned i=0; i<view.size(); i++)
      fprintf(f, "view[%u]:      %s\n",  i, view[i].str().c_str());
    fprintf(f, "max_step:      %lu\n", max_step);
    fprintf(f, "num_slot:      %lu\n", num_slot());
    fprintf(f, "max_episode:   %lu\n", max_episode);
    fprintf(f, "priority_e:    %lf\n", priority_exponent);
    fprintf(f, "mix_lambda:    %lf\n", mix_lambda);
    fprintf(f, "rollout_len:   %u\n",  rollout_len);
    fprintf(f, "do_padding:    %d\n",  do_padding);
    fprintf(f, "priority_decay:%lf\n", priority_decay);
    fprintf(f, "tb_threshold:  %lf\n", traceback_threshold);
    fprintf(f, "entry::nbytes  %lu\n", DataEntry::nbytes(this));
    fprintf(f, "discount_f:    [");
    for(unsigned i=0; i<discount_factor.size(); i++)
      fprintf(f, "%lf,", discount_factor[i]);
    fprintf(f, "]\n");
    fprintf(f, "reward_coeff:  [");
    for(unsigned i=0; i<reward_coeff.size(); i++)
      fprintf(f, "%lf,", reward_coeff[i]);
    fprintf(f, "]\n");
  }

  size_t num_slot() const { return slots.size(); }

public:
  /**
   * Add sequential data into the replay memory.
   */
  bool add_data(uint32_t slot_index, void * raw_data, uint32_t start_step, uint32_t n_step, bool is_episode_end) {
    qassert(slot_index < slots.size());
    if(slots[slot_index].add_data(this, raw_data, start_step, n_step, is_episode_end)) { // success
      not_empty.notify_one();
      return true;
    } else
      return false;
  }

  /**
   * Discard current episode in the memory slot
   */
  void discard_data(uint32_t slot_index) {
    slots[slot_index].discard_data(this);
  }

  /**
   * Clear slot
   */
  void clear(uint32_t slot_index) {
    slots[slot_index].clear(this);
  }

  /**
   * Get batch_size * rollout data samples. Currently this is sampling with replacement.
   * TODO(qing) should we use average priority over rollout for sampling?
   */
  void get_data(void * raw_data, float * weight, uint32_t batch_size) {
    qassert(rollout_len <= max_step);
    for(uint32_t batch_idx=0; batch_idx<batch_size; batch_idx++) {
      std::unique_lock<std::mutex> lock(rem_mutex);
      if(slot_prt.get_weight_sum() <= batch_size * EPS) { // EPS?
        qlog_info("get_data(): Waiting for data.\n");
        not_empty.wait(lock, [this](){ return slot_prt.get_weight_sum() > 0; });
      }
      // Sample the memory slot.
      uint32_t slot_index = slot_prt.sample_index();
      // NOTE: If we want to sample the old/slows slot less often, we could decay the slot's priority here.
      //       e.g. slot_prt.set_weight(slot_index, 0.99 * slot_prt[slot_index].get_weight() )
      //       Also, in sampling without replacement mode, the priority of slot_index is also decreased.
      //       e.g. priority_decay = 0
      lock.unlock();
      if(true) {
        std::lock_guard<std::mutex> guard(slots[slot_index].slot_mutex);
        qlog_debug("get_data(): total_weight_sum: %le, weight_sum: %le\n",
            slot_prt.get_weight_sum(), slots[slot_index].prt.get_weight_sum());
        uint32_t last_idx = slots[slot_index].prt.sample_index();
        weight[batch_idx] = slots[slot_index].prt.get_weight(last_idx); // rollout's weight determined by the last entry
        if(priority_decay != 1.0)
          slots[slot_index].prt.set_weight(last_idx, priority_decay * weight[batch_idx]);
        long len = rollout_len - 1;
        if(do_padding)
          len = std::min<long>(rollout_len - 1, slots[slot_index].get_offset_in_episode(last_idx));
        uint32_t entry_idx = (last_idx - len + max_step) % max_step;
        len += 1;
        char * dst = static_cast<char*>(raw_data) + batch_idx * rollout_len * entry_size;
        void * src = slots[slot_index][entry_idx].data();
        // Padding with first state
        for(int i=0; i<rollout_len - len; i++) {
          memcpy(dst, src, entry_size);
          dst += entry_size;
        }
        // Copy to raw_data
        if(entry_idx + len > static_cast<long>(max_step)) {
          long write_len = max_step - entry_idx;
          memcpy(dst, src, write_len * entry_size);
          dst += write_len * entry_size;
          src = slots[slot_index][0].data();
          memcpy(dst, src, (len - write_len) * entry_size);
        }
        else
          memcpy(dst, src, len * entry_size);
        qlog_debug("w[%u]: %e\n", batch_idx, weight[batch_idx]);
      }
      if(priority_decay != 1.0) {
        lock.lock();
        slot_prt.set_weight(slot_index, slots[slot_index].prt.get_weight_sum());
      }
    }
  }

public:
  enum Mode {
    Conn = 0,
    Bind = 1,
  };

};

/**
 * Default data type for ReplayMemory 
 */
typedef ReplayMemory RM;

