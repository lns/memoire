#include "data_entry.hpp"
#include "replay_memory.hpp"
#include "mem.hpp"
#include "array.hpp"

int main(int argc, char* argv[]) {

  typedef DataEntry<float, 32, int, 1, float, 1> data_t;

  data_t entry;

  ReplayMemory<data_t> rem{65536ul*65536ul};

  for(size_t i=0; i<32768ul; i++) {
    int epi_idx = rem.new_episode();
    rem.close_episode(epi_idx);
  }

  Mem m1{128};
  Mem m2;
  
  Array<float> arr(4);

  m2 = std::move(m1);
  printf("%lu\n",m1.size());
  printf("%lu\n",m2.size());

  float a = 123.0;
  for(int i=0; i<10000; i++) {
    arr.memcpy_back(&a);
  }
  printf("%lu\n",arr.size());
  printf("%lu\n",arr.capacity());
  printf("%f\n",arr[0]);
  return 0;
}

