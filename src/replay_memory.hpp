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

#define VERSION (20181108ul)

#define N_VIEW (5)

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
    void * data() { return (void*)(this); }
    BufView bundle(const ReplayMemory * p) { return get(0, p); }
    BufView reward(const ReplayMemory * p) { return get(1, p); }
    BufView prob (const ReplayMemory * p)  { return get(2, p); }
    BufView value(const ReplayMemory * p)  { return get(3, p); }
    BufView qvest(const ReplayMemory * p)  { return get(4, p); }

    static size_t nbytes(const ReplayMemory * p) {
      DataEntry * dummy = (DataEntry*)nullptr;
      auto last = dummy->get(N_VIEW-1, p);
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
    std::queue<Episode> episode;       ///< Episodes
    PrtTree prt;                       ///< Priority tree for sampling
    int stage;                         ///< init -> 0 -> new -> 10 -> close -> 0
    const unsigned self_index;         ///< slot_index in ReplayMemory
    std::mutex slot_mutex;             ///< mutex for this memory slot
    long new_offset;                   ///< offset for new episode
    long new_length;                   ///< current length of new episode
    long cur_step;                     ///< current length since episode beginning
    bool not_first;                    ///< current episode is not the first ever

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
      stage{0},
      self_index{index},
      new_offset{0},
      new_length{max_step},
      cur_step{0},
      not_first{false}
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
     * Get length for a new episode
     */
    long get_new_length() const {
      long length = data.capacity();
      if(episode.size() > 0)
        length = (episode.front().offset - episode.back().offset - episode.back().length) % data.capacity();
      return length;
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
      episode.pop();
    }

    /**
     * Update value in an episode
     */
    void update_value(const ReplayMemory * prm, long off, long len, bool is_finished) {
      assert(prm->reward_buf().size() != 0);
      assert(prm->value_buf().size() != 0);
      assert(prm->qvest_buf().size() != 0);
      const auto& gamma = prm->discount_factor;
      // Fill qvest
      for(int i=len-1; i>=0; i--) {
        long post = (off + i + 1) % data.capacity();
        long prev = (off + i) % data.capacity();
        auto post_value  = data[post].value(prm).as_array<float>();
        auto prev_reward = data[prev].reward(prm).as_array<float>();
        auto prev_qvest  = data[prev].qvest(prm).as_array<float>();
        auto post_qvest  = data[post].qvest(prm).as_array<float>();
        for(int j=0; j<(int)prev_reward.size(); j++) { // OPTIMIZE:
          // TD-Lambda
          if(i == len-1) // last
            prev_qvest[j] = (is_finished ? prev_reward[j] : data[prev].value(prm).as_array<float>()[j]);
          else
            prev_qvest[j] = prev_reward[j] + prm->mix_lambda * gamma[j] * post_qvest[j] +
              (1-prm->mix_lambda) * gamma[j] * post_value[j];
        }
      }
    }
    void update_value(const ReplayMemory * prm, const Episode& epi) {
      return update_value(prm, epi.offset, epi.length, true);
    }

    /**
     * Update weight in an episode.
     */
    void update_weight(ReplayMemory * prm, long off, long len) {
      assert(prm->reward_buf().size() != 0);
      assert(prm->value_buf().size() != 0);
      assert(prm->qvest_buf().size() != 0);
      double local_diff = 0;
      double local_ma = prm->ma_sqa;
      float c = sqrt(prm->ma_sqa/2);
      for(int i=0; i<len; i++) {
        long idx = (off + i) % data.capacity();
        auto prev_value  = data[idx].value(prm).as_array<float>();
        auto prev_qvest  = data[idx].qvest(prm).as_array<float>();
        // R is computed with TD-lambda, while V is the original value in prediction
        float priority = 0;
        for(int i=0; i<(int)prev_qvest.size(); i++)
          priority += prm->reward_coeff[i] * (prev_qvest[i] - prev_value[i]);
        // priority is (R-V), now update ma_sqa
        local_diff += (priority * priority - local_ma);
        // calculate real priority
        priority = pow(fabs(priority/c), prm->priority_exponent);
        prt.set_weight(idx, priority);
      }
      prm->ma_sqa += 1e-8 * local_diff;
      prm->ma_sqa = std::max<float>(prm->ma_sqa, EPS);
    }
    void update_weight(ReplayMemory * prm, const Episode& epi) {
      return update_weight(prm, epi.offset, epi.length);
    }

    void clear_priority(long offset, long len) {
      // TODO: Optimize this
      qassert(len >= 0);
      for(int i=0; i<len; i++) {
        long idx = (offset + i) % data.capacity();
        prt.set_weight(idx, 0);
      }
    }

    void update_global_weight(ReplayMemory * prm) {
      std::lock_guard<std::mutex> guard(prm->rem_mutex);
      prm->slot_prt.set_weight(self_index, prt.get_weight_sum());
    }

    void add_data(ReplayMemory * prm, void * raw_data, uint32_t start_step, uint32_t n_step, bool is_episode_end) {
      std::lock_guard<std::mutex> guard(slot_mutex);
      // Check stage
      if(start_step == 0) {
        if(stage != 0) {
          // Clear data
          qlog_warning("Discard unfinished episode (offset:%ld, len:%ld) in slot %u.\n", new_offset, new_length, self_index);
          clear_priority(new_offset, new_length);
        }
        new_offset = get_new_offset();
        new_length = 0;
        cur_step = 0;
        stage = 10;
      }
      if(start_step != cur_step or stage != 10) {
        qlog_warning("Check failed. Usually this is caused by lost messages (maybe push_buf_size is too short?).");
        qlog_info("start_step: %u, cur_step: %ld, is_end: %d, stage: %d, offset: %ld, length: %ld\n",
            start_step, cur_step, is_episode_end, stage, new_offset, new_length);
      }
      qassert(stage == 10);
      qassert(start_step == cur_step);
      // Check space
      while(!episode.empty() and (new_length + n_step) > get_new_length()) {
        //qlog_info("new_length: %ld, get_new_length: %ld\n", new_length, get_new_length());
        long len = new_length + n_step - get_new_length();
        auto& front = episode.front();
        if(len < front.length) {
          front.offset += len;
          front.length -= len;
        }
        else
          episode.pop();
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
      new_length = std::min<long>(new_length + n_step, data.capacity());
      new_offset = (idx - new_length + data.capacity()) % data.capacity();
      cur_step += n_step;
      prm->total_steps += n_step;
      if(is_episode_end) {
        qassert(stage == 10);
        episode.emplace(new_offset, new_length);
        while(prm->max_episode > 0 and episode.size() > prm->max_episode)
          remove_oldest(prm);
        prm->total_episodes += 1;
        stage = 0;
      }
      // Update value and weight
      update_value(prm, new_offset, new_length, is_episode_end);
      update_weight(prm, new_offset, new_length);
      if(not not_first) // Leave first prm->rollout_len-1 entry's weight as zero
        clear_priority(new_offset, prm->rollout_len - 1 + new_length - cur_step);
      clear_priority(idx, prm->rollout_len - 1);
      qlog_debug("new_offset: %ld, new_length: %ld, idx: %ld\n", new_offset, new_length, idx);
      qlog_debug("prt.get_weight_sum(): %le\n", prt.get_weight_sum());
      not_first |= is_episode_end; 
      update_global_weight(prm);
    }

  };

public:
  const BufView view[N_VIEW];        ///< buffer view of b, r, p, v, q

  const size_t entry_size;           ///< size of each entry (in bytes)
  const size_t max_step;             ///< max number of steps can be stored

  size_t max_episode;                ///< max number of stored episodes allowed for each slot (0 for not checking)
  float priority_exponent;           ///< exponent of priority term (default 0.0)
  float mix_lambda;                  ///< mixture factor for computing multi-step return
  unsigned rollout_len;              ///< rollout length
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
  ReplayMemory(const BufView * vw,
      size_t m_step,
      size_t n_slot,
      qlib::RNG * prt_rng):
    view{
      BufView(nullptr, vw[0].itemsize_, vw[0].format_, vw[0].shape_, vw[0].stride_),
      BufView(nullptr, vw[1].itemsize_, vw[1].format_, vw[1].shape_, vw[1].stride_),
      BufView(nullptr, vw[2].itemsize_, vw[2].format_, vw[2].shape_, vw[2].stride_),
      BufView(nullptr, vw[3].itemsize_, vw[3].format_, vw[3].shape_, vw[3].stride_),
      BufView(nullptr, vw[4].itemsize_, vw[4].format_, vw[4].shape_, vw[4].stride_),
    },
    entry_size{DataEntry::nbytes(this)},
    max_step{m_step},
    max_episode{0},
    priority_exponent{0.0},
    mix_lambda{1.0},
    rollout_len{1},
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
    qassert(bundle_buf().is_c_contiguous());
    qassert(reward_buf().is_c_contiguous());
    qassert(prob_buf().is_c_contiguous());
    qassert(value_buf().is_c_contiguous());
    qassert(qvest_buf().is_c_contiguous());
  }

  const BufView& bundle_buf() const { return view[0]; }
  const BufView& reward_buf() const { return view[1]; }
  const BufView& prob_buf()   const { return view[2]; }
  const BufView& value_buf()  const { return view[3]; }
  const BufView& qvest_buf()  const { return view[4]; }

  void print_info(FILE * f = stderr) const
  {
    fprintf(f, "uuid:          %s\n",  uuid.c_str());
    fprintf(f, "bundle_buf:    %s\n",  bundle_buf().str().c_str());
    fprintf(f, "reward_buf:    %s\n",  reward_buf().str().c_str());
    fprintf(f, "prob_buf:      %s\n",  prob_buf().str().c_str());
    fprintf(f, "value_buf:     %s\n",  value_buf().str().c_str());
    fprintf(f, "qvest_buf:     %s\n",  qvest_buf().str().c_str());
    fprintf(f, "max_step:      %lu\n", max_step);
    fprintf(f, "num_slot:      %lu\n", num_slot());
    fprintf(f, "max_episode:   %lu\n", max_episode);
    fprintf(f, "priority_e:    %lf\n", priority_exponent);
    fprintf(f, "mix_lambda:    %lf\n", mix_lambda);
    fprintf(f, "rollout_len:   %u\n",  rollout_len);
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
  void add_data(uint32_t slot_index, void * raw_data, uint32_t start_step, uint32_t n_step, bool is_episode_end) {
    qassert(slot_index < slots.size());
    slots[slot_index].add_data(this, raw_data, start_step, n_step, is_episode_end);
    not_empty.notify_one();
  }

  /**
   * Get batch_size * rollout data samples. Currently this is sampling with replacement.
   * TODO(qing) sampling without replacement.
   * TODO(qing) should we use average priority over rollout for sampling?
   */
  void get_data(void * raw_data, float * weight, uint32_t batch_size) {
    qassert(rollout_len <= max_step);
    for(uint32_t batch_idx=0; batch_idx<batch_size; batch_idx++) {
      std::unique_lock<std::mutex> lock(rem_mutex);
      if(slot_prt.get_weight_sum() <= 0) {
        qlog_info("get_data(): Waiting for data.\n");
        not_empty.wait(lock, [this](){ return slot_prt.get_weight_sum() > 0; });
      }
      // Sample the first entry
      uint32_t slot_index = slot_prt.sample_index();
      lock.unlock();
      if(true) {
        std::lock_guard<std::mutex> guard(slots[slot_index].slot_mutex);
        qlog_debug("get_data(): total_weight_sum: %le, weight_sum: %le\n",
            slot_prt.get_weight_sum(), slots[slot_index].prt.get_weight_sum());
        uint32_t entry_idx = slots[slot_index].prt.sample_index();
        weight[batch_idx] = slots[slot_index].prt.get_weight(entry_idx);
        entry_idx = (entry_idx - rollout_len + 1 + max_step) % max_step; // rollout's weight is the last entry's
        // Copy to raw_data
        char * dst = static_cast<char*>(raw_data) + batch_idx * rollout_len * entry_size;
        void * src = slots[slot_index][entry_idx].data();
        if(entry_idx + rollout_len > max_step) {
          long len = max_step - entry_idx;
          memcpy(dst, src, len * entry_size);
          dst += len * entry_size;
          src = slots[slot_index][0].data();
          memcpy(dst, src, (rollout_len - len) * entry_size);
        }
        else
          memcpy(dst, src, rollout_len * entry_size);
        qlog_debug("w[%u]: %e\n", batch_idx, weight[batch_idx]);
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

