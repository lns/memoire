#pragma once

#include "utils.hpp"
#include "replay_memory.hpp"
#include <future>

template<class RM>
class ReplayMemoryClient : public non_copyable {
public:
  typedef typename RM::Message Message;

  RM * prm;
  void * rrsoc; // for request-reply
  void * ppsoc; // for push-pull
  int reqbuf_size;
  char * reqbuf;
  int repbuf_size;
  char * repbuf;
  int pushbuf_size;
  char * pushbuf;
  Message *req, *rep, *push;

  ReplayMemoryClient(
      void * ctx,
      const char * req_endpoint,
      const char * push_endpoint,
      size_t rem_max_capacity,
      int req_size = -1,
      int rep_size = -1,
      int push_size = -1) : prm(nullptr)
  {
    rrsoc = zmq_socket(ctx, ZMQ_REQ);
    ZMQ_CALL(zmq_connect(rrsoc, req_endpoint));
    ppsoc = zmq_socket(ctx, ZMQ_PUSH);
    ZMQ_CALL(zmq_connect(ppsoc, push_endpoint));  
    qlog_info("Client CONN: %s()\n",__func__);
    // Make bufs
    reqbuf_size  = MAX(reqbuf_size,  RM::reqbuf_size());
    reqbuf = (char*)malloc(reqbuf_size);
    req  = reinterpret_cast<Message*>(reqbuf);
    repbuf_size  = MAX(repbuf_size,  RM::repbuf_size());
    repbuf = (char*)malloc(repbuf_size);
    rep  = reinterpret_cast<Message*>(repbuf);
    pushbuf_size = MAX(pushbuf_size, RM::pushbuf_size());
    pushbuf = (char*)malloc(pushbuf_size);
    push = reinterpret_cast<Message*>(pushbuf);
    // Get sizes
    sync_sizes(rem_max_capacity);
    prm->print_info();
  }

  ~ReplayMemoryClient() {
    zmq_close(rrsoc);
    zmq_close(ppsoc);
    free(reqbuf);
    free(repbuf);
    free(pushbuf);
    delete prm;
  }

  void push_episode_to_remote(int epi_idx) {
    push->type = Message::AddEpisode;
    push->length = prm->episode[epi_idx].size();
    ZMQ_CALL(zmq_send(ppsoc, pushbuf, sizeof(Message), ZMQ_SNDMORE));
    size_t send_size = prm->entry_size * push->length;
    ZMQ_CALL(zmq_send(ppsoc, prm->episode[epi_idx].data.data(), send_size, 0));
  }

protected:
  void sync_sizes(size_t max_capacity) {
    if(prm)
      delete prm;
    req->type = Message::GetSizes;
    ZMQ_CALL(zmq_send(rrsoc, reqbuf, sizeof(Message), 0));
    ZMQ_CALL(zmq_recv(rrsoc, repbuf, repbuf_size, 0));
    assert(rep->type == Message::Success);
    // Get sizes
    size_t * p = reinterpret_cast<size_t*>(&rep->entry);
    prm = new RM{p[0], p[1], p[2], p[3], p[4], max_capacity};
  }

};

