#pragma once

#include "utils.hpp"
#include <vector>
#include <thread>
#include <cstdlib>
#include <functional>

/**
 * Worker Function
 *
 * Parse request buffer, do the real work, and return results in responed buffer
 *
 * @param reqbuf         request buffer
 * @param reqbuf_size    size of request
 * @param repbuf         responde buffer
 * @param repbuf_size    max size of responed buffer
 * @return size         size of responed data
 */
typedef std::function<int(char *, int, char *, int)> worker_func_t;

void worker_main(
    worker_func_t worker_func,
    int reqbuf_size,
    int repbuf_size,
    void * ctx,
    const char * endpoint,
    int id)
{
  void * soc = zmq_socket(ctx, ZMQ_REP);
  ZMQ_CALL(zmq_connect(soc, endpoint));
  // prepare
  char * reqbuf = (char*)malloc(reqbuf_size);
  char * repbuf = (char*)malloc(repbuf_size);
  int size;
  while(true) {
    ZMQ_CALL(size = zmq_recv(soc, reqbuf, reqbuf_size, 0));
    // Real work is done here.
    // arguments in reqbuf are parsed and processed,
    // and the outputs are written to the repbuf.
    size = worker_func(reqbuf, size, repbuf, repbuf_size);
    assert(0 <= size and size <= repbuf_size);
    ZMQ_CALL(zmq_send(soc, repbuf, size, 0));
  }
  // never get here
  zmq_close(soc);
}

void server_main(
    worker_func_t worker_func,
    void * ctx,
    int reqbuf_size,
    int repbuf_size,
    const char * router_endpoint,
    const char * dealer_endpoint,
    const char * worker_endpoint,
    int n_threads)
{
  // frontend
  void * router = zmq_socket(ctx, ZMQ_ROUTER);
  ZMQ_CALL(zmq_bind(router, router_endpoint));
  // backend
  void * dealer = zmq_socket(ctx, ZMQ_DEALER);
  ZMQ_CALL(zmq_bind(dealer, dealer_endpoint));
  // start worker threads
  std::vector<std::thread> thread;
  for(int id=0; id<n_threads; id++)
    thread.emplace_back(worker_main, worker_func, reqbuf_size, repbuf_size, ctx, worker_endpoint, id);
  // main loop
  ZMQ_CALL(zmq_proxy(router, dealer, nullptr));
  // never get here
  zmq_close(router);
  zmq_close(dealer);
}

