#include "replay_memory.hpp"
#include "backward.hpp"
#include <cstdio>

namespace backward {
  backward::SignalHandling sh;
}

static void print_logo() __attribute__((constructor));

void print_logo() {
  fprintf(stderr, " Memoire, ver 17.10.12.\n");
}

