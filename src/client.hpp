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
  int reqbuf_size;
  char * reqbuf;
  int repbuf_size;
  char * repbuf;
  int send_size;

  ReplayMemoryClient(
      void * ctx,
      const char * rrendpoint,
      int req_size,
      int rep_size)
  {
    rrsoc = zmq_socket(ctx, ZMQ_REQ);
    ZMQ_CALL(zmq_connect(rrsoc, rrendpoint));
    // Make bufs
    reqbuf_size = req_size;
    reqbuf = (char*)malloc(reqbuf_size);
    repbuf_size = rep_size;
    repbuf = (char*)malloc(repbuf_size);
    // Get sizes
    int size;
    Message * req = reinterpret_cast<Message*>(reqbuf);
    req->type = Message::GetSizes;
    ZMQ_CALL(zmq_send(rrsoc, reqbuf, sizeof(Message), 0));
    ZMQ_CALL(size = zmq_recv(rrsoc, repbuf, repbuf_size, 0));
    Message * rep = reinterpret_cast<Message*>(repbuf);
    assert(rep->type == Message::Success);
    size_t * p = reinterpret_cast<size_t*>(&rep->entry);
    rem = new RM{p[0], p[1], p[2], p[3], p[4], 0};
    send_size = sizeof(Message) + RM::DataEntry::bytesize(rem);
    assert(send_size <= reqbuf_size);
  }

  ~ReplayMemoryClient() {
    zmq_close(rrsoc);
    free(reqbuf);
    free(repbuf);
    delete rem;
  }

  int close_and_new(int old_epi_idx)
  {
    Message * req = reinterpret_cast<Message*>(reqbuf);
    req->type = Message::CloseAndNew;
    req->epi_idx = old_epi_idx; // set old_epi_idx = -1 to omit closing
    int size;
    ZMQ_CALL(zmq_send(rrsoc, reqbuf, sizeof(Message), 0));
    ZMQ_CALL(size = zmq_recv(rrsoc, repbuf, repbuf_size, 0));
    Message * rep = reinterpret_cast<Message*>(repbuf);
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
    Message * msg = reinterpret_cast<Message*>(reqbuf);
    msg->type = Message::AddEntry;
    msg->epi_idx = epi_idx;
    msg->entry.from_memory(rem, 0, p_s, p_a, p_r, p_p, p_v);
    ZMQ_CALL(zmq_send(rrsoc, reqbuf, send_size, ZMQ_NOBLOCK));
    int size;
    ZMQ_CALL(size = zmq_recv(rrsoc, repbuf, repbuf_size, 0));
    Message * rep = reinterpret_cast<Message*>(repbuf);
    assert(rep->type == Message::Success);
  }

};

