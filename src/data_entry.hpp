#pragma once

#include <memory>
#include <array>

template<typename state_t, size_t state_N,
  typename action_t, size_t action_N,
  typename reward_t, size_t reward_N>
class DataEntry
{
public:
  std::array<state_t,  state_N>  state;
  std::array<action_t, action_N> action;
  std::array<reward_t, reward_N> reward;
};

