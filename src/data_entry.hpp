#pragma once

#include <cstring>  // size_t
#include "array.hpp"

/**
 * Memory structure of an entry. Does not own the memory.
 * (Actually, we should not have member variables, except for static ones)
 */
template<typename state_t, typename action_t, typename reward_t>
class DataEntry
{
public:
  typedef float prob_t;       // probabilities are saved in float
  typedef reward_t value_t;   // values should have the same type as reward

  static size_t state_size;   ///< num of state
  static size_t action_size;  ///< num of action
  static size_t reward_size;  ///< num of reward
  static size_t prob_size;    ///< num of base probability
  static size_t value_size;   ///< num of discounted future reward sum

  static size_t bytesize() {
    return state_size * sizeof(state_t)
      + action_size * sizeof(action_t)
      + reward_size * sizeof(reward_t)
      + prob_size * sizeof(prob_t)
      + value_size * sizeof(value_t);
  }

  Array<state_t> state() {
    char * head = reinterpret_cast<char*>(this);
    size_t offset = 0;
    return Array<state_t>(head + offset, state_size);
  }

  Array<action_t> action() {
    char * head = reinterpret_cast<char*>(this);
    size_t offset = state_size * sizeof(state_t);
    return Array<action_t>(head + offset, action_size);
  }

  Array<reward_t> reward() {
    char * head = reinterpret_cast<char*>(this);
    size_t offset = state_size * sizeof(state_t)
      + action_size * sizeof(action_t);
    return Array<reward_t>(head + offset, reward_size);
  }

  Array<prob_t> prob() {
    char * head = reinterpret_cast<char*>(this);
    size_t offset = state_size * sizeof(state_t)
      + action_size * sizeof(action_t)
      + reward_size * sizeof(reward_t);
    return Array<prob_t>(head + offset, prob_size);
  }

  Array<value_t> value() {
    char * head = reinterpret_cast<char*>(this);
    size_t offset = state_size * sizeof(state_t)
      + action_size * sizeof(action_t)
      + reward_size * sizeof(reward_t)
      + prob_size * sizeof(prob_t);
    return Array<value_t>(head + offset, value_size);
  }

};

