#pragma once
#include <cstdint>
#include <cassert>
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
template<class T>
class Episode;
template<class T>
class ReplayMemory;

/**
 * An Episode in replay memory
 */
template<class T>
class Episode
{
public:
  // Const ptr to parent replay memory
  const ReplayMemory<T> * rem;
  // empty, filling or filled (terminated)
  Flag flag;
  // Actual data
  std::vector<T> data;

  Episode(const ReplayMemory<T>& replay_memory) : rem{&replay_memory}, flag{Empty} {}

  void reset() {
    flag = Empty;
    data.clear();
  }

};

template<class T>
class DataBatch {
  std::vector<T> prev;
  std::vector<T> next;
};

template<class T>
class ReplayMemory
{
public:
  // Max episode
  const size_t max_episode;
  // Episodes
  std::vector<Episode<T>> episode;

  ReplayMemory(const size_t max_epi) : max_episode{max_epi} { episode.reserve(max_episode); }

  /**
   * Get index of a new episode
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

  void close_episode(size_t epi_idx) {
    assert(epi_idx < episode.size());
    assert(episode[epi_idx].flag == Filling);
    episode[epi_idx].flag = Filled;
  }

  template<typename... Args>
  void emplace(size_t epi_idx, Args&&... args) {
    assert(epi_idx < episode.size());
    assert(episode[epi_idx].flag == Filling);
    episode[epi_idx].emplace_back(std::forward<Args>(args)...);
  }

  void get_batch(size_t batch_size, DataBatch<T>& batch) const {
    batch.prev.clear();
    batch.next.clear();
    batch.prev.reserve(batch_size);
    batch.next.reserve(batch_size);
    for(size_t i=0; i<batch_size; i++) {
      // choose an episode
      int epi_idx = g_rng() % episode.size();
      assert(episode[epi_idx].size() > 0);
      // choose an example
      int pre_idx = g_rng() % (episode[epi_idx].size() - 1);
      // add to batch
      batch.prev.push_back(episode[epi_idx].data[pre_idx]);
      batch.next.push_back(episode[epi_idx].data[pre_idx+1]);
    }
  }

};

