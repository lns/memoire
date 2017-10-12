#pragma once
#include <cstdint>
#include <cassert>
#include "vector.hpp"
#include "rng.hpp"

/**
 * Episode Flag
 */
enum Flag : uint8_t {
  Empty = 0,
  Filling = 1,
  Filled = 2,
};

// declarations
template<typename T>
class Episode;
template<typename T>
class ReplayMemory;

/**
 * An Episode in replay memory
 */
template<typename T>
class Episode
{
public:
  // Const ptr to parent replay memory
  const ReplayMemory<T> * rem;
  // empty, filling or filled (terminated)
  Flag flag;
  // Actual data
  Vector<T> data;

  Episode(const ReplayMemory<T>& replay_memory)
    : rem{&replay_memory}, flag{Empty}, data{rem->entry_size} {}

  void reset() {
    flag = Empty;
    data.clear();
  }

  size_t size() const { return data.size(); }

  /**
   * Update value when episode is finished
   */
  void update_value() {
    if(T::value_size==0)
      return;
    assert(T::value_size == T::reward_size);
    int i=data.size()-1;
    if(true) { // last entry
      auto prev_value = data[i].value();
      auto prev_reward = data[i].reward();
      for(int j=0; j<prev_value.size(); j++) { // OPTIMIZE:
        prev_value[j] = prev_reward[j];
      }
    }
    const auto& gamma = rem->discount_factor;
    for( ; i>=0; i--) {
      auto prev_value = data[i].value();
      auto post_value = data[i+1].value();
      auto prev_reward = data[i].reward();
      for(int j=0; j<prev_value.size(); j++) { // OPTIMIZE:
        prev_value[j] = prev_reward[j] + gamma * post_value[j];
      }
    }
  }

};

template<class T>
class DataBatch {
public:
  Vector<T> prev;
  Vector<T> next;

  DataBatch(size_t entry_size) : prev{entry_size}, next{entry_size} {}
};

template<class T>
class ReplayMemory
{
public:
  const size_t entry_size;           // size of each entry (in bytes)
  const size_t max_episode;          // Max episode
  std::vector<Episode<T>> episode;   // Episodes

  float discount_factor;             // discount factor for calculate R with rewards

  /**
   * Construct a new replay memory
   * @param e_size   size of a single entry (in bytes)
   * @param max_epi  max number of episodes kept in the memory
   */
  ReplayMemory(size_t e_size, size_t max_epi) : entry_size{e_size}, max_episode{max_epi}
  {
    episode.reserve(max_episode);
  }

  /**
   * Get index of a new episode
   * @return index of an episode just opened
   */
  size_t new_episode() {
    if(episode.size() < max_episode) {
      // Use a new episode
      episode.emplace_back(*this);
      size_t idx = episode.size()-1;
      episode[idx].flag = Filling;
      return idx;
    } else {
      // Reuse an old episode
      std::uniform_int_distribution<size_t> dis(0, episode.size()-1);
      for(int i=0; i<128; i++) {
        size_t idx = dis(g_rng);
        if(episode[idx].flag!=Filling) {
          episode[idx].reset();
          episode[idx].flag = Filling;
          return idx;
        }
      }
      assert(false and "Cannot found available episode slot.");
    }
  }

  /**
   * Close an episode (by setting its flag to `Filled`)
   * @param epi_idx  index of the episode (should be opened)
   */
  void close_episode(size_t epi_idx) {
    assert(epi_idx < episode.size());
    assert(episode[epi_idx].flag == Filling);
    episode[epi_idx].flag = Filled;
    episode[epi_idx].update_value();
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
   * Get a batch of samples/transitions
   *
   * @param batch_size             batch_size
   * @param batch                  batch instance for storing output samples
   * @param finished_episodes_only sample only from finished episodes, if set true (default true)
   * @param interval               interval between prev state and next state (normally 1)
   *
   * @return true iff success
   */
  bool get_batch(size_t batch_size, DataBatch<T>& batch,
      bool finished_episodes_only = true, unsigned interval = 1) const {
    assert(episode.size() > 0);
    assert(batch.prev.entry_size == entry_size);
    batch.prev.clear();
    batch.next.clear();
    batch.prev.reserve(batch_size);
    batch.next.reserve(batch_size);
    for(size_t i=0; i<batch_size; i++) {
      int epi_idx = g_rng() % episode.size();
      // choose an episode
      int attempt = 0;
      for( ; attempt<episode.size(); attempt++) {
        if(episode[epi_idx].size() > interval) // we need prev state and next state
          break;
        else
          epi_idx = (epi_idx + 1) % episode.size();
      }
      if(attempt==episode.size())
        return false;
      // choose an example
      int pre_idx = g_rng() % (episode[epi_idx].size() - interval);
      // add to batch
      batch.prev.memcpy_back(&episode[epi_idx].data[pre_idx]);
      batch.next.memcpy_back(&episode[epi_idx].data[pre_idx + interval]);
    }
    return true;
  }

};

