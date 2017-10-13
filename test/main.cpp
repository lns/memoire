#include "replay_memory.hpp"
#include "mem.hpp"
#include "vector.hpp"
#include "backward.hpp"

namespace backward {
  backward::SignalHandling sh;
}

int main(int argc, char* argv[]) {

  typedef ReplayMemory<float,int,float> RM;
  RM rem{14,1,1,0,1, 32768};
  rem.discount_factor = 1.0;// 0.99;


  if(true) {
    // Add entries locally
    float s[14];
    int a[1];
    float r[1];
    float v[1];

    for(size_t i=0; i<32768ul; i++) {
      int epi_idx = rem.new_episode();
      for(int step=0; step<1024; step++) {
        s[0] = i;
        s[13] = i;
        a[0] = step;
        r[0] = 1.0;
        rem.add_entry(epi_idx, s, a, r, nullptr, v);
      }
      rem.close_episode(epi_idx);
    }
  }

  if(true) {
    // Get batch data
    #define BATCH_SIZE 32
    float prev_s[BATCH_SIZE][14];
    int   prev_a[BATCH_SIZE][1];
    float prev_r[BATCH_SIZE][1];
    float prev_v[BATCH_SIZE][1];
    float next_s[BATCH_SIZE][14];
    int   next_a[BATCH_SIZE][1];
    float next_r[BATCH_SIZE][1];
    float next_v[BATCH_SIZE][1];

    printf("prev: %p %p %p %p\n", prev_s[0], prev_a[0], prev_r[0], prev_v[0]);
    printf("next: %p %p %p %p\n", next_s[0], next_a[0], next_r[0], next_v[0]);
    for(size_t i=0; i<10ul; i++) {
      assert(true==rem.get_batch(BATCH_SIZE,
            prev_s[0], prev_a[0], prev_r[0], nullptr, prev_v[0],
            next_s[0], next_a[0], next_r[0], nullptr, next_v[0],
            true, 1));
      printf("%f %d %f %f\n",
          prev_s[0][13], prev_a[0][0], prev_r[0][0], prev_v[0][0]);
      printf("%f %d %f %f\n",
          next_s[0][13], next_a[0][0], next_r[0][0], next_v[0][0]);
    }
    for(size_t i=0; i<32768ul; i++) {
      assert(true==rem.get_batch(BATCH_SIZE,
            prev_s[0], prev_a[0], prev_r[0], nullptr, prev_v[0],
            next_s[0], next_a[0], next_r[0], nullptr, next_v[0],
            true, 1));
    }
    #undef BATCH_SIZE
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

