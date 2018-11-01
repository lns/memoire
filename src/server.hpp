#pragma once

#include "utils.hpp"
#include <vector>
#include <thread>
#include <cstdlib>
#include <functional>
#include "replay_memory.hpp"
#include "hexdump.hpp"
#include <arpa/inet.h>
#include "py_serial.hpp"
#include "zmq_base.hpp"

template<class RM>
class ReplayMemoryServer : public ZMQBase {
public:
  RM rem;
  const std::string x_descr_pickle;
  std::unordered_map<std::string, uint32_t> m;

  ReplayMemoryServer(const BufView * vw, size_t max_step, size_t n_slot, 
      qlib::RNG * prt_rng, std::string input_descr_pickle)
    : rem{vw, max_step, n_slot, prt_rng},
      x_descr_pickle{input_descr_pickle}
  {}

  ~ReplayMemoryServer() {}

protected:
  uint32_t get_slot_index(std::string client_uuid)
  {
    auto iter = m.find(client_uuid);
    uint32_t slot_index;
    if(iter == m.end()) { // Not found
      slot_index = m.size();
      m[client_uuid] = slot_index;
    } else {
      slot_index = m[client_uuid];
    }
    if(slot_index >= rem.num_slot())
      qlog_error("Insufficient number of slots (%lu) for these actors.", rem.num_slot());
    return slot_index;
  }

public:
  void print_info(FILE * f = stderr) const
  {
    rem.print_info(f);
  }
 
  /**
   * Responsible for answering ReqGetInfo
   */
  void rep_worker_main(const char * endpoint, typename RM::Mode mode)
  {
    void * soc = new_zmq_socket(ZMQ_REP);
    std::string reqbuf(1024, '\0'), repbuf(1024, '\0'); // TODO(qing): adjust default size
    if(mode == RM::Bind)
      ZMQ_CALL(zmq_bind(soc, endpoint));
    else if(mode == RM::Conn)
      ZMQ_CALL(zmq_connect(soc, endpoint));
    int size;
    while(true) {
      ZMQ_CALL(size = zmq_recv(soc, &reqbuf[0], reqbuf.size(), 0));
      if(not (size <= (int)reqbuf.size())) { // resize and wait for next
        reqbuf.resize(size);
        qlog_warning("Resize reqbuf to %lu. Waiting for next request.. \n", reqbuf.size());
        continue;
      }
      proto::Msg req, rep;
      req.ParseFromString(reqbuf);
      qlog_debug("Received msg: %s\n", req.DebugString().c_str());
      qassert(req.version() == rep.version());
      rep.set_version(req.version());
      if(req.type() == proto::REQ_GET_INFO) {
        rep.set_type(proto::REP_GET_INFO);
        auto * p = rep.mutable_rep_get_info();
        p->set_x_descr_pickle(x_descr_pickle);
        p->set_slot_index(get_slot_index(req.sender()));
        p->set_entry_size(rem.entry_size);
        p->clear_view();
        for(int i=0; i<N_VIEW; i++)
          rem.view[i].to_pb(p->add_view()); 
      }
      else
        qthrow("Unknown args->type");
      rep.SerializeToString(&repbuf);
      ZMQ_CALL(zmq_send(soc, repbuf.data(), repbuf.size(), 0));
      qlog_debug("Send msg: %s\n", rep.DebugString().c_str());
    }
  }

  /**
   * Responsible for receiving PushData
   */
  void pull_worker_main(const char * endpoint, typename RM::Mode mode)
  {
    void * soc = new_zmq_socket(ZMQ_PULL);
    std::string buf(1024, '\0'); // TODO(qing): adjust default size
    if(mode == RM::Bind)
      ZMQ_CALL(zmq_bind(soc, endpoint));
    else if(mode == RM::Conn)
      ZMQ_CALL(zmq_connect(soc, endpoint));
    int size;
    while(true) {
      ZMQ_CALL(size = zmq_recv(soc, &buf[0], buf.size(), 0));
      if(not (size <= (int)buf.size())) { // resize and wait for next
        buf.resize(size);
        qlog_warning("Resize buf to %lu. Waiting for next push.. \n", buf.size());
        continue;
      }
      proto::Msg msg;
      msg.ParseFromString(buf);
      qlog_debug("Received msg: %s\n", msg.DebugString().c_str());
      qassert(msg.version() == msg.version());
      if(msg.type() == proto::PUSH_DATA) {
        auto * p = msg.mutable_push_data();
        rem.add_data(p->slot_index(), (void*)p->data().data(), p->start_step(), p->n_step(), p->is_episode_end());
      }
      else
        qthrow("Unknown args->type");
    }
  }

  /**
   * Unserialize an entry from memory
   */
  py::tuple py_unserialize_from_mem(void * data)
  {
    static py::object descr = pickle_loads(x_descr_pickle);
    static size_t x_nbytes = get_descr_nbytes(descr);
    // Unserialize from data: x
    char * head = static_cast<char*>(data);
    py::object x = descr_unserialize_from_mem(descr, head);
    head += x_nbytes;
    py::list entry = py::list(x);
    for(int i=1; i<5; i++) { // r, p, v, q
      const BufView& v = rem.view[i];
      py::array a(py::dtype(v.format_), v.shape_, v.stride_, nullptr);
      memcpy(a.mutable_data(), head, v.nbytes());
      head += v.nbytes();
      entry.append(a);
    }
    return py::tuple(entry);
  }

  /**
   * Get data and weight: (data, weight)
   */
  py::tuple py_get_data(uint32_t batch_size, uint32_t rollout_length)
  {
    Mem mem(batch_size * rollout_length * rem.entry_size);
    pyarr_float w({batch_size,});
    if(true) {
      py::gil_scoped_release release;
      rem.get_data(mem.data(), w.mutable_data(), batch_size, rollout_length);
    }
    py::list ret;
    for(uint32_t batch_idx=0; batch_idx<batch_size; batch_idx++) {
      py::list rollout;
      for(uint32_t step=0; step<rollout_length; step++) {
        char * head = (char*)mem.data() + (batch_idx * rollout_length + step) * rem.entry_size;
        py::tuple entry = py_unserialize_from_mem(head);
        rollout.append(entry);
      }
      ret.append(rollout);
    }
    return py::make_tuple(ret, w);
  }

};

typedef ReplayMemoryServer<RM> RMS;

