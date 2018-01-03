#pragma once
#include <cstdint>
#include <atomic>
#include <mutex>
#include "qlog.hpp"
#include "array_view.hpp"
#include "vector.hpp"
#include "rng.hpp"
#include <signal.h>
#include "utils.hpp" // non_copyable

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
        const value_t  * p_v) {
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
        value_t  * p_v) {
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
  static int repbuf_size()  { return sizeof(Message) + 5*sizeof(size_t); }
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

    Episode(const ReplayMemory& replay_memory)
      : rem{&replay_memory}, inc{0}, data{rem->entry_size} {}
    Episode(Episode&& src) // for stl
      : rem{src.rem}, inc{src.inc.load()}, data{src.data} {}

    void clear() { data.clear(); }

    size_t size() const { return data.size(); }

    /**
     * Update value when episode is finished
     */
    void update_value() {
      if(rem->value_size==0 or data.size()==0)
        return;
      assert(rem->value_size == rem->reward_size);
      int i=data.size()-1;
      if(true) { // last entry
        auto prev_value = data[i].value(rem);
        auto prev_reward = data[i].reward(rem);
        for(int j=0; j<prev_value.size(); j++) { // OPTIMIZE:
          prev_value[j] = prev_reward[j];
        }
      }
      i--;
      const auto& gamma = rem->discount_factor;
      for( ; i>=0; i--) {
        auto prev_value = data[i].value(rem);
        auto post_value = data[i+1].value(rem);
        auto prev_reward = data[i].reward(rem);
        for(int j=0; j<prev_value.size(); j++) { // OPTIMIZE:
          prev_value[j] = prev_reward[j] + gamma * post_value[j];
        }
      }
    }

  };

  #define IS_FILLING(inc) ((inc)%2==1)
  #define IS_FILLED(inc)  ((inc)%2==0)

public:
  const size_t state_size;           ///< num of state
  const size_t action_size;          ///< num of action
  const size_t reward_size;          ///< num of reward
  const size_t prob_size;            ///< num of base probability
  const size_t value_size;           ///< num of discounted future reward sum

  const size_t entry_size;           ///< size of each entry (in bytes)
  const size_t max_episode;          ///< Max episode
  std::vector<Episode> episode;      ///< Episodes
  std::mutex episode_mutex;          ///< mutex for new/close an episode

  float discount_factor;             ///< discount factor for calculate R with rewards

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
      size_t max_epi) :
    state_size{s_size},
    action_size{a_size},
    reward_size{r_size},
    prob_size{p_size},
    value_size{v_size},
    entry_size{T::nbytes(this)},
    max_episode{max_epi}
  {
    episode.reserve(max_episode);
  }

  void print_info() const {
    printf("state_size:  %lu\n", state_size);
    printf("action_size: %lu\n", action_size);
    printf("reward_size: %lu\n", reward_size);
    printf("prob_size:   %lu\n", prob_size);
    printf("value_size:  %lu\n", value_size);
    printf("max_episode: %lu\n", max_episode);
  }

  size_t num_episode() const { return episode.size(); }

  /**
   * Get index of a new episode
   * @return index of an episode just opened
   */
  size_t new_episode() {
    for(int attempt=0; attempt<2; attempt++) {
      if(episode.size() < max_episode) {
        // Use a new episode
        std::lock_guard<std::mutex> guard(episode_mutex);
        if(episode.size() >= max_episode)
          continue;
        episode.emplace_back(*this);
        size_t idx = episode.size()-1;
        episode[idx].inc++; // filled -> filling
        return idx;
      } else {
        // Reuse an old episode
        std::uniform_int_distribution<size_t> dis(0, episode.size()-1);
        for(int i=0; i<128; i++) { // TODO: 128 attempts
          size_t idx = dis(g_rng);
          uint64_t loaded = episode[idx].inc;
          if(IS_FILLED(loaded)) {
            if(not atomic_compare_exchange_strong(&episode[idx].inc, &loaded, loaded+1))
              continue;
            episode[idx].clear();
            return idx;
          }
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
  void close_episode(size_t epi_idx) {
    qassert(epi_idx < episode.size());
    uint64_t loaded = episode[epi_idx].inc;
    qassert(IS_FILLING(loaded));
    episode[epi_idx].update_value();
    qassert(atomic_compare_exchange_strong(&episode[epi_idx].inc, &loaded, loaded+1));
  }

  /**
   * Clear all `Filled` data
   * Filling episodes will remain valid
   */
  void clear() {
    for(auto&& each : episode) {
      uint64_t loaded = each.inc;
      if(IS_FILLING(loaded))
        continue;
      if(not atomic_compare_exchange_strong(&each.inc, &loaded, loaded+1)) // failed
        continue;
      loaded ++;
      // now filling
      each.clear();
      qassert(atomic_compare_exchange_strong(&each.inc, &loaded, loaded+1));
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
  void memcpy_back(size_t epi_idx, const T * src) {
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
   */
  void add_entry(size_t epi_idx,
      const state_t  * p_s,
      const action_t * p_a,
      const reward_t * p_r,
      const prob_t   * p_p) {
    qassert(epi_idx < episode.size());
    qassert(IS_FILLING(episode[epi_idx].inc));
    episode[epi_idx].data.memcpy_back(nullptr); // push_back an empty entry, which will be filled later
    auto& entry = episode[epi_idx].data[ episode[epi_idx].data.size()-1 ]; // last one
    entry.from_memory(this, 0, p_s, p_a, p_r, p_p, nullptr);
  }

  /**
   * Get a batch of samples/transitions
   * Assuming the memory layout for data (state, action ..) as
   *  data[batch_size][data_size]
   *
   * @param batch_size             batch_size
   * @param finished_episodes_only sample only from finished episodes, if set true (default true)
   * @param interval               interval between prev state and next state (normally 1)
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
      bool finished_episodes_only = true, unsigned interval = 1) {
    if(episode.size() == 0) {
      qlog_warning("get_batch() failed as the ReplayMemory is empty.\n");
      return false;
    }
    for(size_t i=0; i<batch_size; i++) {
      int epi_idx = g_rng() % episode.size();
      // choose an episode
      int attempt = 0;
      uint64_t loaded;
      for( ; attempt<episode.size(); attempt++) {
        loaded = episode[epi_idx].inc;
        if(episode[epi_idx].size() > interval // we need prev state and next state
            and (!finished_episodes_only or IS_FILLED(loaded)))
          break;
        else
          epi_idx = (epi_idx + 1) % episode.size();
      }
      if(attempt==episode.size()) {
        qlog_warning("get_batch() failed.\n"
            "This may due to a large interval(%u) used comparing to episodes' lengths (this:%lu).\n",
            interval, episode[epi_idx].size());
        if(finished_episodes_only and IS_FILLING(episode[epi_idx].inc))
          qlog_warning("Or due to all episodes are currently filling, as finished_episodes_only is set to '%d'\n",
              (int)finished_episodes_only);
        return false;
      }
      // choose an example
      int pre_idx = g_rng() % (episode[epi_idx].size() - interval);
      // add to batch
      auto& prev_entry = episode[epi_idx].data[pre_idx];
      prev_entry.to_memory(this, i, prev_s, prev_a, prev_r, prev_p, prev_v);
      auto& next_entry = episode[epi_idx].data[pre_idx + interval];
      next_entry.to_memory(this, i, next_s, next_a, next_r, next_p, next_v);
      if(loaded != episode[epi_idx].inc) // memory has been altered, redo this sample
        i--;
    }
    return true;
  }

  enum Mode {
    Conn = 0,
    Bind = 1,
  };

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
        prm->close_episode(epi_idx);
      }
      else
        qthrow("Unknown args->type");
    }
    // never
    free(buf);
    zmq_close(soc);
  }

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
        size_t * p = reinterpret_cast<size_t *>(&rets->entry);
        p[0] = prm->state_size;
        p[1] = prm->action_size;
        p[2] = prm->reward_size;
        p[3] = prm->prob_size;
        p[4] = prm->value_size;
        ZMQ_CALL(zmq_send(soc, repbuf, sizeof(Message) + 5*sizeof(size_t), 0));
      }
      else
        qthrow("Unknown args->type");
    }
    // never
    free(reqbuf);
    free(repbuf);
    zmq_close(soc);
  }

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

