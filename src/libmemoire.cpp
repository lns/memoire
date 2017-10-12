#include "data_entry.hpp"
#include "replay_memory.hpp"
#include "backward.hpp"
#include <cstdio>

typedef DataEntry<float, float, float> Entry;

template<> size_t Entry::state_size = 0;
template<> size_t Entry::action_size = 0;
template<> size_t Entry::reward_size = 0;
template<> size_t Entry::prob_size = 0;
template<> size_t Entry::value_size = 0;

namespace backward {
  backward::SignalHandling sh;
}

static void print_logo() __attribute__((constructor));

void print_logo() {
  fprintf(stderr, " Memoire, ver 17.10.12.\n");
}

