#include "data_entry.hpp"
#include "replay_memory.hpp"
#include "mem.hpp"
#include "vector.hpp"

template<> size_t DataEntry<float, int, float, float, float>::state_size = 14;
template<> size_t DataEntry<float, int, float, float, float>::action_size = 1;
template<> size_t DataEntry<float, int, float, float, float>::reward_size = 1;
template<> size_t DataEntry<float, int, float, float, float>::prob_size = 0;
template<> size_t DataEntry<float, int, float, float, float>::value_size = 0;

int main(int argc, char* argv[]) {

  typedef DataEntry<float,int,float,float,float> Entry;
  printf("Entry::bytesize(): %lu\n", Entry::bytesize());
  printf("sizeof(Entry): %lu\n", sizeof(Entry));

  ReplayMemory<Entry> rem{Entry::bytesize(), 32768};
  DataBatch<Entry> batch{Entry::bytesize()};

  Entry * pe = reinterpret_cast<Entry*>(malloc(Entry::bytesize()));
  for(size_t i=0; i<32768ul; i++) {
    int epi_idx = rem.new_episode();
    for(int step=0; step<1024; step++) {
      pe->state()[0] = i;
      pe->state()[13] = i;
      pe->action()[0] = step;
      pe->reward()[0] = 1.0/(1+i);
      rem.memcpy_back(epi_idx, pe);
    }
    rem.close_episode(epi_idx);
  }
  free(pe);

  for(size_t i=0; i<10ul; i++) {
    assert(true==rem.get_batch(32, batch, true));
    printf("%f %d %f\n",
        batch.prev[0].state()[0],
        batch.prev[0].action()[0],
        batch.prev[0].reward()[0]);
    printf("%f %d %f\n",
        batch.next[0].state()[0],
        batch.next[0].action()[0],
        batch.next[0].reward()[0]);
  }
  for(size_t i=0; i<32768ul; i++) {
    assert(true==rem.get_batch(32, batch, false));
  }

  Mem m1{128};
  Mem m2;
  
  Vector<float> arr(sizeof(float));

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

