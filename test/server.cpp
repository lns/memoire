#include "replay_memory.hpp"
#include "backward.hpp"
#include <thread>
#include "mtserver.hpp"
#include "client.hpp"
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
  typedef ReplayMemoryClient<float,int,float> RMC;
  RM rem{S_SIZE,A_SIZE,R_SIZE,0,R_SIZE, 64};
  rem.discount_factor = 1.0;// 0.99;

  worker_func_t worker_func = [&rem](char * inbuf, int inbuf_size, char * oubuf, int oubuf_size) -> int {
    return rem.process(inbuf, inbuf_size, oubuf, oubuf_size);
  };

  int inbuf_size = RM::DataEntry::bytesize(&rem) + sizeof(RM::Message); // OPTIMIZE: DIVUP
  int oubuf_size = sizeof(RM::Message) + 5*sizeof(size_t);

  void * ctx = zmq_ctx_new();
#if 1
  std::thread server(server_main, worker_func, ctx, inbuf_size, oubuf_size,
      "tcp://*:5561", "inproc://workers", "inproc://workers", 7);
#else
  server_main(worker_func, ctx, inbuf_size, oubuf_size,
      "tcp://*:5561", "inproc://workers", "inproc://workers", 7);
#endif
  
  RMC client{ctx, "tcp://localhost:5561", inbuf_size, oubuf_size};

  qlog_info("Begin adding entries.\n");
  if(true) {
    float s[S_SIZE];
    int   a[A_SIZE];
    float r[R_SIZE];
    float v[R_SIZE];

    int epi_idx = -1;
    for(int iter=0; iter<256; iter++) {
      epi_idx = client.close_and_new(epi_idx);
      for(int step=0; step<64; step++) {
        s[0] = iter;
        s[S_SIZE-1] = iter;
        a[0] = step;
        a[A_SIZE-1] = step;
        r[0] = 1;
        r[R_SIZE-1] = 1;
        client.add_entry(epi_idx, s, a, r, nullptr, nullptr);
      }
    }
  }
  qlog_info("Finished adding entries.\n");

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

  server.join();

  zmq_ctx_destroy(ctx);
  return 0;
}

