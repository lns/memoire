#include "replay_memory.hpp"
#include "mem.hpp"
#include "vector.hpp"
#include "backward.hpp"
#include <thread>

namespace backward {
  backward::SignalHandling sh;
}

#define S_SIZE 145
#define A_SIZE 7
#define R_SIZE 3

template<class T>
void add_entries(T& rem) {
  // Add entries locally
  float s[S_SIZE];
  int   a[A_SIZE];
  float r[R_SIZE];

  for(size_t i=0; i<1000ul; i++) {
    int epi_idx = rem.new_episode();
    for(int step=0; step<5000; step++) {
      s[0] = i;
      s[S_SIZE-1] = i;
      a[0] = step;
      r[0] = 1.0;
      rem.add_entry(epi_idx, s, a, r, nullptr);
    }
    rem.close_episode(epi_idx);
  }
}

int main(int argc, char* argv[]) {

  typedef ReplayMemory<float,int,float> RM;
  RM rem{S_SIZE,A_SIZE,R_SIZE,0,R_SIZE, 1024};
  rem.discount_factor = 1.0;// 0.99;

  if(true) {
    #define N_THREAD 1
    std::thread * pt[N_THREAD];
    for(int i=0; i<N_THREAD; i++) {
      pt[i] = new std::thread(add_entries<RM>, std::ref(rem));
    }
    for(int i=0; i<N_THREAD; i++) {
      pt[i]->join();
      delete pt[i];
    }
  }
  
  exit(0);

  if(true) {
    // Get batch data
    #define BATCH_SIZE 256
    float prev_s[BATCH_SIZE][S_SIZE];
    int   prev_a[BATCH_SIZE][A_SIZE];
    float prev_r[BATCH_SIZE][R_SIZE];
    float prev_v[BATCH_SIZE][R_SIZE];
    float next_s[BATCH_SIZE][S_SIZE];
    int   next_a[BATCH_SIZE][A_SIZE];
    float next_r[BATCH_SIZE][R_SIZE];
    float next_v[BATCH_SIZE][R_SIZE];

    printf("prev: %p %p %p %p\n", prev_s[0], prev_a[0], prev_r[0], prev_v[0]);
    printf("next: %p %p %p %p\n", next_s[0], next_a[0], next_r[0], next_v[0]);
    for(size_t i=0; i<10ul; i++) {
      assert(true==rem.get_batch(BATCH_SIZE,
            prev_s[0], prev_a[0], prev_r[0], nullptr, prev_v[0],
            next_s[0], next_a[0], next_r[0], nullptr, next_v[0],
            true, 1000));
      printf("%f %d %f %f\n",
          prev_s[0][S_SIZE-1], prev_a[0][0], prev_r[0][0], prev_v[0][0]);
      printf("%f %d %f %f\n",
          next_s[0][S_SIZE-1], next_a[0][0], next_r[0][0], next_v[0][0]);
    }
    for(size_t i=0; i<4096ul; i++) {
      assert(true==rem.get_batch(BATCH_SIZE,
            prev_s[0], prev_a[0], prev_r[0], nullptr, prev_v[0],
            next_s[0], next_a[0], next_r[0], nullptr, next_v[0],
            true, 1000));
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

