#pragma once
#include <cstdint>
#include <cassert>
#include <mutex>
#include "array_view.hpp"
#include "vector.hpp"
#include "rng.hpp"
#include "utils.hpp" // non_copyable

/**
 * Episode Flag
 */
enum Flag : uint8_t {
  Empty = 0,
  Filling = 1,
  Filled = 2,
};

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
    static size_t bytesize(const ReplayMemory * p) {
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
private:
  typedef DataEntry T;

public:
  /**
   * Message protocal
   */
  class Message : public non_copyable
  {
  public:
    static const int Success = 10;
    static const int CloseAndNew = 11;
    static const int AddEntry = 12;
    static const int GetSizes = 13;

    int type;
    int epi_idx;
    DataEntry entry;
  };

  static int req_size() { return sizeof(Message); }
  static int rep_size() { return sizeof(Message) + 5*sizeof(size_t); }
  int push_size() const { return sizeof(Message) + T::bytesize(this); }

  /**
   * An Episode in replay memory
   */
  class Episode
  {
  public:
    // Const ptr to parent replay memory
    const ReplayMemory * rem;
    // empty, filling or filled (terminated)
    Flag flag;
    // Actual data (owns the memory)
    Vector<T> data;

    Episode(const ReplayMemory& replay_memory)
      : rem{&replay_memory}, flag{Empty}, data{rem->entry_size} {}

    void clear() {
      data.clear();
    }

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
    entry_size{T::bytesize(this)},
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
        episode[idx].flag = Filling;
        return idx;
      } else {
        // Reuse an old episode
        std::uniform_int_distribution<size_t> dis(0, episode.size()-1);
        for(int i=0; i<128; i++) { // TODO: 128 attempts
          size_t idx = dis(g_rng);
          if(episode[idx].flag!=Filling) {
            std::lock_guard<std::mutex> guard(episode_mutex);
            if(episode[idx].flag==Filling) // OPTIMIZE: use a CAS op
              continue;
            episode[idx].flag = Filling;
            episode[idx].clear();
            return idx;
          }
        }
        assert(false and "Cannot found available episode slot.");
      }
    }
    assert(false and "Cannot happen!");
  }

  /**
   * Close an episode (by setting its flag to `Filled`)
   * @param epi_idx  index of the episode (should be opened)
   */
  void close_episode(size_t epi_idx) {
    assert(epi_idx < episode.size());
    assert(episode[epi_idx].flag == Filling);
    episode[epi_idx].update_value();
    episode[epi_idx].flag = Filled;
  }

  /**
   * Memcpy the content of src to the back of episode[epi_idx].
   * This function is safe to be called in parallel, when epi_idx are different for these calls
   *
   * @param epi_idx  index of the episode (should be opened)
   * @param src      pointer to the entry
   */
  void memcpy_back(size_t epi_idx, const T * src) {
    assert(epi_idx < episode.size());
    assert(episode[epi_idx].flag == Filling);
    episode[epi_idx].data.memcpy_back(src);
  }

  /**
   * Add an entry to an existing episode
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
      const prob_t   * p_p,
      const value_t  * p_v) {
    assert(epi_idx < episode.size());
    assert(episode[epi_idx].flag == Filling);
    episode[epi_idx].data.memcpy_back(nullptr); // push_back an empty entry, which will be filled later
    auto& entry = episode[epi_idx].data[ episode[epi_idx].data.size()-1 ]; // last one
    entry.from_memory(this, 0, p_s, p_a, p_r, p_p, p_v);
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
    assert(episode.size() > 0);
    if(finished_episodes_only)
      episode_mutex.lock();
    for(size_t i=0; i<batch_size; i++) {
      int epi_idx = g_rng() % episode.size();
      // choose an episode
      int attempt = 0;
      for( ; attempt<episode.size(); attempt++) {
        if(episode[epi_idx].size() > interval // we need prev state and next state
            and (!finished_episodes_only or episode[epi_idx].flag == Filled))
          break;
        else
          epi_idx = (epi_idx + 1) % episode.size();
      }
      if(attempt==episode.size()) {
        if(finished_episodes_only)
          episode_mutex.unlock();
        return false;
      }
      // choose an example
      int pre_idx = g_rng() % (episode[epi_idx].size() - interval);
      // add to batch
      auto& prev_entry = episode[epi_idx].data[pre_idx];
      prev_entry.to_memory(this, i, prev_s, prev_a, prev_r, prev_p, prev_v);
      auto& next_entry = episode[epi_idx].data[pre_idx + interval];
      next_entry.to_memory(this, i, next_s, next_a, next_r, next_p, next_v);
    }
    if(finished_episodes_only)
      episode_mutex.unlock();
    return true;
  }

  /**
   * Process a message request
   * This is used in server-client protocal
   */
  int process(char * inbuf, int size, char * oubuf, int oubuf_size) {
    // TODO: add const
    Message * args = reinterpret_cast<Message*>(inbuf);
    Message * rets = reinterpret_cast<Message*>(oubuf); // may be nullptr
    if(args->type == Message::CloseAndNew) {
      assert(rets);
      // close last episode and get a new episode
      int epi_idx = args->epi_idx;
      if(epi_idx >= 0)
        close_episode(epi_idx);
      epi_idx = new_episode();
      rets->type = Message::Success;
      rets->epi_idx = epi_idx;
      return sizeof(Message);
    }
    if(args->type == Message::AddEntry) {
      // add an entry
      int epi_idx = args->epi_idx;
      memcpy_back(epi_idx, &args->entry);
      return 0;
    }
    if(args->type == Message::GetSizes) {
      assert(rets);
      // return sizes
      rets->type = Message::Success;
      rets->epi_idx = -1;
      size_t * p = reinterpret_cast<size_t *>(&rets->entry);
      p[0] = state_size;
      p[1] = action_size;
      p[2] = reward_size;
      p[3] = prob_size;
      p[4] = value_size;
      return sizeof(Message) + 5*sizeof(size_t);
    }
    fprintf(stderr, "Unknown message type: %d\n", args->type);
    assert(false);
  }

};

