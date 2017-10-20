#include "replay_memory.hpp"
#include "backward.hpp"
#include <thread>
#include "client.hpp"
#include "server.hpp"
#include "qlog.hpp"

namespace backward {
  backward::SignalHandling sh;
}

#define S_SIZE 123
#define A_SIZE 1
#define R_SIZE 1

int main(int argc, char* argv[])
{
  qlog_set_print_time(true);

  typedef ReplayMemory<float,int,float> RM;
  typedef ReplayMemoryClient<RM> RMC;
  typedef ReplayMemoryServer<RM> RMS;

  RM rem{S_SIZE,A_SIZE,R_SIZE,0,R_SIZE, 8192};
  rem.discount_factor = 1.0;// 0.99;

  RMS server(&rem);

  std::vector<std::thread> threads;
  //threads.emplace_back([&server](){ server.rep_proxy_main("tcp://*:5561", "inproc://rep_workers"); });
  threads.emplace_back([&server](){
      server.pull_proxy_main("tcp://*:5562", "inproc://pull_workers");
    });

  threads.emplace_back([&server](){
      server.rep_worker_main("tcp://*:5561", RM::Bind);
    });
  threads.emplace_back([&server](){
      server.pull_worker_main("inproc://pull_workers", RM::Conn);
    });

  //sleep(1);
 
  RMC client{"tcp://localhost:5561", "tcp://localhost:5562", 1};

  qlog_info("Begin adding entries.\n");
  if(true) {
    float s[S_SIZE];
    int   a[A_SIZE];
    float r[R_SIZE];

    int epi_idx = -1;
    for(int iter=0; iter<1000; iter++) {
      epi_idx = client.prm->new_episode();
      for(int step=0; step<5000; step++) {
        s[0] = iter;
        s[S_SIZE-1] = iter;
        a[0] = step;
        a[A_SIZE-1] = step;
        r[0] = 1;
        r[R_SIZE-1] = 1;
        client.prm->add_entry(epi_idx, s, a, r, nullptr);
      }
      client.prm->close_episode(epi_idx);
      client.push_episode_to_remote(epi_idx);
    }
  }
  qlog_info("Finished adding entries.\n");

  //sleep(1);

  /*
  while(true) {
    if(rem.episode.size() == 1000)
      exit(0);
    else {
      //printf("%lu\n",rem.episode.size());
      qlib::sleep(0.0);
      std::this_thread::yield();
    }
  }
  */

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
            true, 1));
      printf("%f %d %f %f\n",
          prev_s[0][S_SIZE-1], prev_a[0][0], prev_r[0][0], prev_v[0][0]);
      printf("%f %d %f %f\n",
          next_s[0][S_SIZE-1], next_a[0][0], next_r[0][0], next_v[0][0]);
    }
    for(size_t i=0; i<4096ul; i++) {
      assert(true==rem.get_batch(BATCH_SIZE,
            prev_s[0], prev_a[0], prev_r[0], nullptr, prev_v[0],
            next_s[0], next_a[0], next_r[0], nullptr, next_v[0],
            true, 1));
    }
    #undef BATCH_SIZE
  }
  qlog_info("All finished.\n");

  for(auto&& each : threads)
    each.join();

  return 0;
}

