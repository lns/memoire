#pragma once

#include "utils.hpp"
#include "replay_memory.hpp"
#include "qlog.hpp"
#include "msg.pb.h"
#include "py_serial.hpp"
#include "zmq_base.hpp"

template<class RM>
class ReplayMemoryClient : public ZMQBase {
public:
  std::string x_descr_pickle;
  uint32_t remote_slot_index;
  uint32_t entry_size;
  std::vector<BufView> view;

  std::string req_endpoint;
  std::string push_endpoint;
  std::string uuid;

  uint32_t start_step;

public:
  /**
   * Initialization
   *
   * endpoint can be set to nullptr or "" to disable this protocal
   */
  ReplayMemoryClient(
      const std::string req_ep,
      const std::string push_ep,
      const std::string input_uuid)
    : remote_slot_index{~0u},
      req_endpoint{req_ep},
      push_endpoint{push_ep},
      uuid{input_uuid},
      start_step{0}
  {}

  ~ReplayMemoryClient() {}

  void get_info() {
    thread_local void * soc = nullptr;
    thread_local std::string reqbuf;
    thread_local std::string repbuf;
    if(not soc) {
      soc = new_zmq_socket(ZMQ_REQ);
      ZMQ_CALL(zmq_connect(soc, req_endpoint.c_str()));
      reqbuf.resize(1024, '\0');
      repbuf.resize(1024, '\0'); // TODO(qing): adjust default size
    }
    proto::Msg req;
    req.set_type(proto::REQ_GET_INFO);
    req.set_version(req.version());
    req.set_sender(uuid);
    req.SerializeToString(&reqbuf);
    do {
      int size;
      ZMQ_CALL(zmq_send(soc, reqbuf.data(), reqbuf.size(), 0));
      qlog_debug("Send msg.. %s\n", req.DebugString().c_str()); 
      ZMQ_CALL(size = zmq_recv(soc, &repbuf[0], repbuf.size(), 0));
      if(not (size <= (int)repbuf.size())) { // resize and wait for next
        repbuf.resize(size);
        qlog_warning("Resize repbuf to %lu. Resending.. \n", repbuf.size());
        continue;
      }
    } while(false);
    proto::Msg rep;
    rep.ParseFromString(repbuf);
    qlog_debug("Received msg.. %s\n", rep.DebugString().c_str()); 
    qassert(rep.version() == req.version());
    qassert(rep.type() == proto::REP_GET_INFO);
    // Get info
    const proto::RepGetInfo& info = rep.rep_get_info();
    x_descr_pickle = info.x_descr_pickle();
    remote_slot_index = info.slot_index();
    entry_size = info.entry_size();
    qassert(info.view_size() == N_VIEW);
    view.resize(info.view_size());
    for(int i=0; i<info.view_size(); i++)
      view[i].from_pb(&info.view(i));
  }

  void push_data(void * data, uint32_t n_step, bool is_episode_end) {
    thread_local void * soc = nullptr;
    thread_local std::string pushbuf;
    if(not soc) {
      soc = new_zmq_socket(ZMQ_PUSH);
      ZMQ_CALL(zmq_connect(soc, push_endpoint.c_str()));
      pushbuf.resize(1024, '\0');
    }
    proto::Msg push;
    push.set_type(proto::PUSH_DATA);
    push.set_version(push.version()); // TODO(qing): check this
    push.set_sender(uuid);
    auto * d = push.mutable_push_data();
    d->set_is_episode_end(is_episode_end);
    d->set_start_step(start_step);
    d->set_n_step(n_step);
    d->set_slot_index(remote_slot_index);
    d->set_data(data, n_step * entry_size);
    push.SerializeToString(&pushbuf);
    qlog_debug("Send msg.. %s\n", push.DebugString().c_str()); 
    ZMQ_CALL(zmq_send(soc, pushbuf.data(), pushbuf.size(), 0));
    if(is_episode_end)
      start_step = 0;
    else
      start_step += n_step;
  }

  void py_serialize_entry_to_mem(py::tuple entry, void * data) const {
    if(x_descr_pickle == "")
      qlog_error("x_descr_pickle not initialied. Please call get_info() first.\n");
    qassert(view.size() == 5);
    char * head = static_cast<char*>(data);
    // x
    static py::object descr = pickle_loads(py::bytes(x_descr_pickle)); 
    head += descr_serialize_to_mem(py::list(entry)[py::slice(0,-3,1)], descr, head);
    if(true) { // r
      BufView v = AS_BV(entry[entry.size()-3]);
      qassert(v.is_consistent_with(view[1]) or "Shape of r mismatch");
      v.to_memory(head);
      head += v.nbytes();
    }
    if(true) { // p
      BufView v = AS_BV(entry[entry.size()-2]);
      qassert(v.is_consistent_with(view[2]) or "Shape of p mismatch");
      v.to_memory(head);
      head += v.nbytes();
    }
    if(true) { // v
      BufView v = AS_BV(entry[entry.size()-1]);
      qassert(v.is_consistent_with(view[3]) or "Shape of v mismatch");
      v.to_memory(head);
      head += v.nbytes();
    }
    if(true) { // q
      BufView v = AS_BV(entry[entry.size()-1]);
      qassert(v.is_consistent_with(view[4]) or "Shape of q mismatch");
      v.to_memory(head); // reuse data from v
      head += v.nbytes();
    }
    qassert(head == static_cast<char*>(data) + entry_size);
  }

  void py_push_data(py::list data, bool is_episode_end) {
    uint32_t n_step = data.size();
    Mem mem(entry_size * n_step);
    // Serialize data to memory
    for(unsigned i=0; i<n_step; i++) {
      char * head = (char*)mem.data() + i * entry_size;
      py_serialize_entry_to_mem(data[i], head);
    }
    // Send data to remote
    if(true) {
      py::gil_scoped_release release;
      push_data(mem.data(), n_step, is_episode_end);
    }
  }

};

typedef ReplayMemoryClient<RM> RMC;

