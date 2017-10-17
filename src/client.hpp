#pragma once

#include "utils.hpp"
#include "replay_memory.hpp"
#include <future>

template<typename state_t, typename action_t, typename reward_t>
class ReplayMemoryClient : public non_copyable {
public:
  typedef ReplayMemory<state_t,action_t,reward_t> RM;
  typedef typename RM::Message Message;
  typedef float prob_t;
  typedef reward_t value_t;

  RM * rem;
  void * rrsoc; // for request-reply
  void * ppsoc; // for push-pull
  int reqbuf_size;
  char * reqbuf;
  int repbuf_size;
  char * repbuf;
  int pushbuf_size;
  char * pushbuf;
  int send_size; // cached
  Message *req, *rep, *push;

  ReplayMemoryClient(
      void * ctx,
      const char * req_endpoint,
      const char * push_endpoint,
      int req_size = -1,
      int rep_size = -1,
      int push_size = -1) : rem(nullptr)
  {
    rrsoc = zmq_socket(ctx, ZMQ_REQ);
    ZMQ_CALL(zmq_connect(rrsoc, req_endpoint));
    ppsoc = zmq_socket(ctx, ZMQ_PUSH);
    ZMQ_CALL(zmq_connect(ppsoc, push_endpoint));  
    qlog_info("Client CONN: %s()\n",__func__);
    // Make bufs
    reqbuf_size  = MAX(reqbuf_size,  RM::req_size());
    reqbuf = (char*)malloc(reqbuf_size);
    req  = reinterpret_cast<Message*>(reqbuf);
    repbuf_size  = MAX(repbuf_size,  RM::rep_size());
    repbuf = (char*)malloc(repbuf_size);
    rep  = reinterpret_cast<Message*>(repbuf);
    // Get sizes
    sync_sizes();
    rem->print_info();
    // Make push buf
    pushbuf_size = MAX(pushbuf_size, rem->push_size());
    pushbuf = (char*)malloc(pushbuf_size);
    push = reinterpret_cast<Message*>(pushbuf);
    assert(send_size <= pushbuf_size);
  }

  ~ReplayMemoryClient() {
    zmq_close(rrsoc);
    free(reqbuf);
    free(repbuf);
    delete rem;
  }

  void sync_sizes() {
    if(rem)
      delete rem;
    req->type = Message::GetSizes;
    ZMQ_CALL(zmq_send(rrsoc, reqbuf, sizeof(Message), 0));
    qlog_info("Client REQ: %s()\n",__func__);
    ZMQ_CALL(zmq_recv(rrsoc, repbuf, repbuf_size, 0));
    qlog_info("Client REP: %s()\n",__func__);
    assert(rep->type == Message::Success);
    // Get sizes
    size_t * p = reinterpret_cast<size_t*>(&rep->entry);
    rem = new RM{p[0], p[1], p[2], p[3], p[4], 0};
    // Set send size
    send_size = sizeof(Message) + RM::DataEntry::bytesize(rem);
  }

  int close_and_new(int old_epi_idx)
  {
    req->type = Message::CloseAndNew;
    req->epi_idx = old_epi_idx; // set old_epi_idx = -1 to omit closing
    ZMQ_CALL(zmq_send(rrsoc, reqbuf, sizeof(Message), 0));
    qlog_info("Client REQ: %s()\n",__func__);
    ZMQ_CALL(zmq_recv(rrsoc, repbuf, repbuf_size, 0));
    qlog_info("Client REP: %s()\n",__func__);
    assert(rep->type == Message::Success);
    return rep->epi_idx;
  }

  void add_entry(int epi_idx,
      const state_t  * p_s,
      const action_t * p_a,
      const reward_t * p_r,
      const prob_t   * p_p,
      const value_t  * p_v)
  {
    push->type = Message::AddEntry;
    push->epi_idx = epi_idx;
    push->entry.from_memory(rem, 0, p_s, p_a, p_r, p_p, p_v);
    ZMQ_CALL(zmq_send(ppsoc, pushbuf, send_size, 0)); //ZMQ_NOBLOCK));
    qlog_info("Client PUSH: %s()\n",__func__);
  }

};

