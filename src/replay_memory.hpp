#pragma once
#include <cstdint>
#include <atomic>
#include <mutex>
#include "qlog.hpp"
#include "array_view.hpp"
#include "vector.hpp"
#include "qrand.hpp"
#include <signal.h>
#include "utils.hpp" // non_copyable
#include "prt_tree.hpp"

template<typename state_t, typename action_t, typename reward_t>
class ReplayMemory
{
public:
  typedef float prob_t;       ///< probabilities are saved in float
  typedef reward_t value_t;   ///< values should have the same type as reward

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

protected:
  typedef DataEntry T;

public:
  /**
   * Message protocal
   */
  class Message : public non_copyable
  {
  public:
    static const int Success = 20;
    static const int AddEpisode = 21;
    static const int GetSizes = 22;

    int type;
    int length;  // length of episode (in number of entries)
    DataEntry entry;
  };

  static int reqbuf_size()  { return sizeof(Message); }
  static int repbuf_size()  { return sizeof(Message) + sizeof(ReplayMemory); }
  static int pushbuf_size() { return sizeof(Message); }

  /**
   * An Episode in replay memory
   */
  class Episode
  {
  public:
    // Const ptr to parent replay memory
    const ReplayMemory * rem;
    // increment
    std::atomic<uint64_t> inc;
    // Actual data (owns the memory)
    Vector<T> data;
    // Priority tree for weighted sampling
    PrtTree prt;

    Episode(const ReplayMemory& replay_memory)
      : rem{&replay_memory}, inc{0}, data{rem->entry_size}, prt{rem->rng, static_cast<int>(rem->epi_max_len)} {}
    Episode(Episode&& src) // for stl
      : rem{src.rem}, inc{src.inc.load()}, data{src.data}, prt{src.prt} {}

    void clear() { data.clear(); }

    size_t size() const { return data.size(); }

    /**
     * Update value when episode is finished
     * 1. Update r as discounted future reward
     * 2. Preserve v
     */
    void update_value()
    {
      if(rem->value_size==0 or data.size()==0)
        return;
      assert(rem->value_size == rem->reward_size);
      int i=data.size()-1;
      i--;
      const auto& gamma = rem->discount_factor;
      for( ; i>=0; i--) {
        auto post_reward = data[i+1].reward(rem);
        auto post_value = data[i+1].value(rem);
        auto prev_reward = data[i].reward(rem);
        for(int j=0; j<prev_reward.size(); j++) { // OPTIMIZE:
          prev_reward[j] += rem->td_lambda * gamma * post_reward[j] + (1-rem->td_lambda) * gamma * post_value[j];
        }
      }
    }

  };

  /**
   * State transition:
   *
   * new_episode():       (0) -> (2)
   * close_episode():     (2) -> (4)
   * update_priority():   (0) -> (1) -> (0)
   */
  #define IS_FILLED(inc)    ((inc)%4==0)
  #define IS_UPDATING(inc)  ((inc)%4==1)
  #define IS_FILLING(inc)   ((inc)%4==2)
  #define IS_SAME_EPISODE(load_a, load_b) ((load_a)/2==(load_b)/2)

public:
  const size_t state_size;           ///< num of state
  const size_t action_size;          ///< num of action
  const size_t reward_size;          ///< num of reward
  const size_t prob_size;            ///< num of base probability
  const size_t value_size;           ///< num of discounted future reward sum

  const size_t entry_size;           ///< size of each entry (in bytes)
  const size_t max_episode;          ///< Max episode
  const size_t epi_max_len;          ///< Max length for an episode (used by Episode::prt)

  float discount_factor;             ///< discount factor for calculate R with rewards
  float priority_exponent;           ///< exponent of priority term (default 0.5)
  float td_lambda;                   ///< mixture factor for TD-lambda
  int frame_stack;                   ///< Number of frames stacked for each state (default 1)
  int multi_step;                    ///< Number of steps between prev and next (default 1)

  std::vector<Episode> episode;      ///< Episodes
  std::mutex episode_mutex;          ///< mutex for new/close an episode
  PrtTree prt_epi;                   ///< Priority tree for sampling episodes
  std::mutex prt_epi_mutex;          ///< mutex for new/close an episode
  size_t tail_idx;                   ///< Index for inserting (See new_episode())

  std::atomic<size_t> total_episodes;///< counter of total episodes
  std::atomic<size_t> total_steps;   ///< counter of total steps

  qlib::RNG * rng;                   ///< Random Number Generator


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
      size_t max_epi,
      size_t episode_max_length,
      qlib::RNG * prt_rng) :
    state_size{s_size},
    action_size{a_size},
    reward_size{r_size},
    prob_size{p_size},
    value_size{v_size},
    entry_size{T::nbytes(this)},
    max_episode{max_epi},
    epi_max_len{episode_max_length},
    priority_exponent{0.5},
    frame_stack{1},
    multi_step{1},
    prt_epi{prt_rng, static_cast<int>(max_epi)},
    tail_idx{0},
    total_episodes{0},
    total_steps{0},
    rng{prt_rng}
  {
    episode.reserve(max_episode);
  }

  void print_info(FILE * f = stderr) const
  {
    fprintf(f, "state_size:  %lu\n", state_size);
    fprintf(f, "action_size: %lu\n", action_size);
    fprintf(f, "reward_size: %lu\n", reward_size);
    fprintf(f, "prob_size:   %lu\n", prob_size);
    fprintf(f, "value_size:  %lu\n", value_size);
    fprintf(f, "max_episode: %lu\n", max_episode);
    fprintf(f, "epi_max_len: %lu\n", epi_max_len);
    fprintf(f, "discount_f:  %lf\n", discount_factor);
    fprintf(f, "priority_e:  %lf\n", priority_exponent);
    fprintf(f, "td_lambda:   %lf\n", td_lambda);
    fprintf(f, "frame_stack: %d\n",  frame_stack);
    fprintf(f, "multi_step:  %d\n",  multi_step);
  }

  size_t num_episode() const { return episode.size(); }

  /**
   * Get index of a new episode
   * @return index of an episode just opened
   */
  size_t new_episode()
  {
    for(int attempt=0; attempt<2; attempt++) {
      if(episode.size() < max_episode) {
        // Use a new episode
        std::lock_guard<std::mutex> guard(episode_mutex);
        if(episode.size() >= max_episode)
          continue;
        episode.emplace_back(*this);
        size_t idx = episode.size()-1;
        qassert(IS_FILLED(episode[idx].inc));
        episode[idx].inc += 2; // filled -> filling
        //qlog_info("New epi[%lu] = %lu\n", idx, episode[idx].inc.load());
        return idx;
      } else {
        // TODO: For DEBUG
        std::vector<std::tuple<size_t, uint64_t, bool>> rets;
        // Reuse an old episode: 
        for(int i=0; i<16; i++) { // TODO: number of attempts
          tail_idx = (tail_idx + 1) % episode.size();
          size_t idx = tail_idx;
          //size_t idx = prt_epi.sample_reversely();
          uint64_t loaded = episode[idx].inc;
          bool ret = false;
          if(IS_FILLED(loaded))
            ret = atomic_compare_exchange_strong(&episode[idx].inc, &loaded, loaded+2);
          rets.push_back(std::make_tuple(idx, loaded, ret));
          if(IS_FILLED(loaded) and ret) {
            if(true) { // TODO: whether to set zero
              std::lock_guard<std::mutex> guard(prt_epi_mutex);
              prt_epi.set_weight(idx, 0.0);
            }
            episode[idx].clear();
            episode[idx].prt.clear();
            //qlog_info("Renew epi[%lu] = %lu\n", idx, episode[idx].inc.load());
            return idx;
          }
        }
        for(const auto& each : rets) {
          fprintf(stderr, "rets: %lu %lu %d\n", std::get<0>(each), std::get<1>(each), std::get<2>(each));
        }
        qthrow("Cannot found available episode slot.");
      }
    }
    qthrow("Cannot happen!");
  }

  /**
   * Close an episode (by setting its flag to `Filled`)
   * @param epi_idx  index of the episode (should be opened)
   */
  void close_episode(size_t epi_idx, bool do_update_value = true, bool do_update_weight = true)
  {
    qassert(epi_idx < episode.size());
    uint64_t loaded = episode[epi_idx].inc;
    qassert(IS_FILLING(loaded));
    if(do_update_value)
      episode[epi_idx].update_value();
    if(do_update_weight) {
      // Update prt and prt_epi
      assert(frame_stack >= 1 and multi_step >= 1);
      for(int i=0; i<(frame_stack-1); i++)
        episode[epi_idx].prt.set_weight_without_update(i, 0.0);
      for(int i=0; i<multi_step; i++)
        episode[epi_idx].prt.set_weight_without_update(episode[epi_idx].size()-1-i, 0.0);
      // Update priority
      //float gg = pow(discount_factor, multi_step);
      for(int i=(frame_stack-1); i<episode[epi_idx].size()-multi_step; i++) {
        auto& prev = episode[epi_idx].data[i];
        //auto& next = episode[epi_idx].data[i+multi_step];
        // R is computed with TD-lambda, while V is original value in prediction
        // TODO: We only consider the first dimension of reward here
        float priority = (prev.reward(this)[0] - prev.value(this)[0]);
        //float priority = (prev.reward(this)[0] - gg*next.reward(this)[0]) - prev.value(this)[0] + gg*next.value(this)[0];
        priority = pow(fabs(priority), priority_exponent);
        episode[epi_idx].prt.set_weight_without_update(i, episode[epi_idx].prt.get_weight(i) * priority);
      }
      episode[epi_idx].prt.update_all();
      //qlog_warning("length: %lu, weight_sum: %lf\n", episode[epi_idx].size(), episode[epi_idx].prt.get_weight_sum());
    }
    if(true) {
      std::lock_guard<std::mutex> guard(prt_epi_mutex);
      prt_epi.set_weight(epi_idx, episode[epi_idx].prt.get_weight_sum());
    }
    //qlog_info("Closing epi[%lu] = %lu\n", epi_idx, loaded);
    qassert(atomic_compare_exchange_strong(&episode[epi_idx].inc, &loaded, loaded+2));
    //qlog_info("Closed epi[%lu]: %lu\n", epi_idx, episode[epi_idx].inc.load());
  }

  /**
   * Clear all `Filled` data
   * Filling episodes will remain valid
   * TODO: Currently not used
   */
  void clear()
  {
    for(auto&& each : episode) {
      uint64_t loaded = each.inc;
      if(not IS_FILLED(loaded))
        continue;
      if(not atomic_compare_exchange_strong(&each.inc, &loaded, loaded+2)) // failed
        continue;
      loaded += 2;
      // now filling
      each.clear();
      qassert(atomic_compare_exchange_strong(&each.inc, &loaded, loaded+2));
    }
    if(true) {
      std::lock_guard<std::mutex> guard(prt_epi_mutex);
      prt_epi.clear();
    }
  }

  /**
   * Memcpy the content of src to the back of episode[epi_idx].
   * This function is safe to be called in parallel, when epi_idx are different for these calls
   * Not thread-safe with identical epi_idx
   *
   * @param epi_idx  index of the episode (should be opened)
   * @param src      pointer to the entry
   */
  void memcpy_back(size_t epi_idx, const T * src)
  {
    qassert(epi_idx < episode.size());
    qassert(IS_FILLING(episode[epi_idx].inc));
    episode[epi_idx].data.memcpy_back(src);
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
  void add_entry(size_t epi_idx,
      const state_t  * p_s,
      const action_t * p_a,
      const reward_t * p_r,
      const prob_t   * p_p,
      const value_t  * p_v,
      float weight)
  {
    qassert(epi_idx < episode.size());
    qassert(IS_FILLING(episode[epi_idx].inc));
    episode[epi_idx].data.memcpy_back(nullptr); // push_back an empty entry, which will be filled later
    int entry_idx = episode[epi_idx].data.size() - 1; // last one
    auto& entry = episode[epi_idx].data[entry_idx]; 
    entry.from_memory(this, 0, p_s, p_a, p_r, p_p, p_v);
    episode[epi_idx].prt.set_weight_without_update(entry_idx, weight);
    qassert(episode[epi_idx].prt.get_weight(entry_idx) == weight);
  }

  /**
   * Get a batch of samples/transitions
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
      state_t  * prev_s,
      action_t * prev_a,
      reward_t * prev_r,
      prob_t   * prev_p,
      value_t  * prev_v,
      state_t  * next_s,
      action_t * next_a,
      reward_t * next_r,
      prob_t   * next_p,
      value_t  * next_v,
      int * epi_idx_arr,
      int * entry_idx_arr,
      uint64_t * epi_inc_arr,
      float * entry_weight_arr)
  {
    if(episode.size() == 0) {
      qlog_warning("get_batch() failed as the ReplayMemory is empty.\n");
      return false;
    }
    for(size_t i=0; i<batch_size; i++) {
      int epi_idx = prt_epi.sample_index();
      // choose an episode
      uint64_t loaded = std::atomic_load_explicit(&episode[epi_idx].inc, std::memory_order_relaxed);
      std::atomic_thread_fence(std::memory_order_acquire);
      if(prt_epi.get_weight(epi_idx) == 0.0) {
        qlog_warning("get_batch() failed.\n"
            "This may due to a large frame_stack(%d) or multi_step(%d) used comparing to episodes' lengths (this:%lu).\n",
            frame_stack, multi_step, episode[epi_idx].size());
        qlog_info("epi_idx: %d, epi_weight: %lf\n", epi_idx, prt_epi.get_weight(epi_idx));
        return false;
      }
      // choose an example
      int pre_idx = episode[epi_idx].prt.sample_index();
      if(pre_idx < (frame_stack-1) or pre_idx + multi_step >= episode[epi_idx].size()) {// prt has been altered, redo this sample
        qlog_warning("pre_idx: %d, weight: %lf, frame_stack: %d, multi_step: %d, episode[%d].size(): %lu\n",
            pre_idx, episode[epi_idx].prt.get_weight(pre_idx), frame_stack, multi_step, epi_idx, episode[epi_idx].size());
        //episode[epi_idx].prt.debug_print(pre_idx);
        i--;
        continue;
      }
      assert(pre_idx >= (frame_stack-1) and pre_idx + multi_step < episode[epi_idx].size());
      // add to batch
      for(int j=frame_stack-1; j>=0; j--) {
        auto& prev_entry = episode[epi_idx].data[pre_idx - j];
        prev_entry.to_memory(this, frame_stack*i + (frame_stack-1-j), prev_s, nullptr, nullptr, nullptr, nullptr);
        if(j==0)
          prev_entry.to_memory(this, i, nullptr, prev_a, prev_r, prev_p, prev_v);
      }
      for(int j=frame_stack-1; j>=0; j--) {
        auto& next_entry = episode[epi_idx].data[pre_idx - j + multi_step];
        next_entry.to_memory(this, frame_stack*i + (frame_stack-1-j), next_s, nullptr, nullptr, nullptr, nullptr);
        if(j==0)
          next_entry.to_memory(this, i, nullptr, next_a, next_r, next_p, next_v);
      }
      if(epi_idx_arr)
        epi_idx_arr[i] = epi_idx;
      if(entry_idx_arr)
        entry_idx_arr[i] = pre_idx;
      if(epi_inc_arr)
        epi_inc_arr[i] = loaded;
      if(entry_weight_arr)
        entry_weight_arr[i] = episode[epi_idx].prt.get_weight(pre_idx);
      uint64_t another_load = std::atomic_load_explicit(&episode[epi_idx].inc, std::memory_order_relaxed);
      std::atomic_thread_fence(std::memory_order_acquire);
      if(not IS_SAME_EPISODE(loaded,another_load)) {// memory has been altered, redo this sample
        qlog_warning("memory has been altered, redo this sample.\n");
        i--;
      }
    }
    return true;
  }

  /**
   * Update sample weight
   */
  void update_weight(size_t batch_size,
      const int * epi_idx_arr,
      const int * entry_idx_arr,
      const uint64_t * epi_inc_arr,
      const float * entry_weight_arr)
  {
    for(int i=0; i<batch_size; i++) {
      int epi_idx = epi_idx_arr[i];
      uint64_t loaded = episode[epi_idx].inc;
      if(not IS_FILLED(loaded))
        continue;
      bool ret = atomic_compare_exchange_strong(&episode[epi_idx].inc, &loaded, loaded+1);
      if(not ret)
        continue;
      loaded += 1;
      if(IS_SAME_EPISODE(loaded, epi_inc_arr[i])) {
        episode[epi_idx].prt.set_weight(entry_idx_arr[i], entry_weight_arr[i]);
        if(true) {
          std::lock_guard<std::mutex> guard(prt_epi_mutex);
          prt_epi.set_weight(epi_idx, episode[epi_idx].prt.get_weight_sum());
        }
      }
      qassert(atomic_compare_exchange_strong(&episode[epi_idx].inc, &loaded, loaded-1));
    }
  }

  enum Mode {
    Conn = 0,
    Bind = 1,
  };

  /**
   * Responsible for receiving an episode and adding it to this replay memory
   */
  static void pull_thread_main(
      ReplayMemory * prm,
      void * ctx,
      const char * endpoint,
      Mode mode)
  {
    void * soc = zmq_socket(ctx, ZMQ_PULL); qassert(soc);
    if(mode == Bind)
      ZMQ_CALL(zmq_bind(soc, endpoint));
    else if(mode == Conn)
      ZMQ_CALL(zmq_connect(soc, endpoint));
    int buf_size = prm->pushbuf_size();
    char * buf = (char*)malloc(buf_size); qassert(buf);
    Message * args = reinterpret_cast<Message*>(buf);
    int size;
    while(true) { // TODO
      ZMQ_CALL(size = zmq_recv(soc, buf, buf_size, 0)); qassert(size <= buf_size);
      if(args->type == Message::AddEpisode) {
        size_t epi_idx = prm->new_episode();
        prm->episode[epi_idx].data.reserve(args->length);
        size_t expected_size = prm->entry_size * args->length;
        ZMQ_CALL(size = zmq_recv(soc, prm->episode[epi_idx].data.data(), expected_size, 0));
        qassert(size == expected_size);
        prm->episode[epi_idx].data.set_size(args->length);
        // Receive PRT
        ZMQ_CALL(size = zmq_recv(soc, prm->episode[epi_idx].prt.w_.data(), 2*prm->episode[epi_idx].prt.size_*sizeof(float), 0));
        prm->close_episode(epi_idx, false, false);
        //qlog_info("Received episode[%lu] of length %lu, weight: %lf\n",
        //    epi_idx, prm->episode[epi_idx].size(), prm->prt_epi.get_weight(epi_idx));
        //prm->episode[epi_idx].prt.debug_print(prm->episode[epi_idx].prt.sample_index()); // for debug
        // Update counters
        prm->total_episodes += 1;
        prm->total_steps += args->length;
      }
      else
        qthrow("Unknown args->type");
    }
    // never
    free(buf);
    zmq_close(soc);
  }

  /**
   * Responsible for answering the request of GetSizes.
   */
  static void rep_thread_main(
      ReplayMemory * prm,
      void * ctx,
      const char * endpoint,
      Mode mode)
  {
    void * soc = zmq_socket(ctx, ZMQ_REP); qassert(soc);
    if(mode == Bind)
      ZMQ_CALL(zmq_bind(soc, endpoint));
    else if(mode == Conn)
      ZMQ_CALL(zmq_connect(soc, endpoint));
    int reqbuf_size = prm->reqbuf_size();
    char * reqbuf = (char*)malloc(reqbuf_size); qassert(reqbuf);
    int repbuf_size = prm->repbuf_size();
    char * repbuf = (char*)malloc(repbuf_size); qassert(repbuf);
    Message * args = reinterpret_cast<Message*>(reqbuf);
    Message * rets = reinterpret_cast<Message*>(repbuf);
    int size;
    while(true) { // TODO
      ZMQ_CALL(size = zmq_recv(soc, reqbuf, reqbuf_size, 0)); qassert(size <= reqbuf_size);
      if(args->type == Message::GetSizes) {
        // return sizes
        rets->type = Message::Success;
        ReplayMemory * p = reinterpret_cast<ReplayMemory *>(&rets->entry);
        // memcpy. Only primary objects are valid.
        memcpy(p, prm, sizeof(ReplayMemory));
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
   * Responsible for connecting multiple frontends and backends
   */
  static void proxy_main(
      void * ctx,
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

};

