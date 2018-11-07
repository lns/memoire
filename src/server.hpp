#pragma once

#include "utils.hpp"
#include <vector>
#include <thread>
#include <cstdlib>
#include <iostream>
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

  std::string pub_endpoint;
  int pub_hwm;
  int rep_hwm;
  int pull_hwm;

  ReplayMemoryServer(const BufView * vw, size_t max_step, size_t n_slot, 
      qlib::RNG * prt_rng, std::string input_descr_pickle)
    : rem{vw, max_step, n_slot, prt_rng},
      x_descr_pickle{input_descr_pickle}
  {
    pub_hwm = rep_hwm = pull_hwm = 32;
  }

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
    ZMQ_CALL(zmq_setsockopt(soc, ZMQ_RCVHWM, &rep_hwm, sizeof(rep_hwm)));
    ZMQ_CALL(zmq_setsockopt(soc, ZMQ_SNDHWM, &rep_hwm, sizeof(rep_hwm)));
    std::string reqbuf(1024, '\0'), repbuf(1024, '\0'); // TODO(qing): adjust default size
    if(mode == RM::Bind)
      ZMQ_CALL(zmq_bind(soc, endpoint));
    else if(mode == RM::Conn)
      ZMQ_CALL(zmq_connect(soc, endpoint));
    int size;
    while(true) {
      size = zmq_recv(soc, &reqbuf[0], reqbuf.size(), 0);
      if(size == -1) { // receive failed.
        int e = zmq_errno();
        qlog_warning("[ERRNO %d] '%s'.", e, zmq_strerror(e));
        return;
      }
      if(not (size <= (int)reqbuf.size())) { // resize and wait for next
        qlog_warning("Resize reqbuf from %lu to %d. Waiting for next request.\n", reqbuf.size(), size);
        reqbuf.resize(size);
        continue;
      }
      proto::Msg req, rep;
      qlog_debug("Received msg of size(%d)\n", size);
      if(not req.ParseFromArray(reqbuf.data(), size)) {
        std::ofstream out("reqbuf.msg", std::ios_base::binary | std::ios_base::trunc);
        out << std::string(reqbuf.data(), size);
        qlog_warning("ParseFromArray(%p, %d) failed. Saved to 'reqbuf.msg'.\n", reqbuf.data(), size);
        std::this_thread::sleep_for(std::chrono::seconds(1));
        ZMQ_CALL(zmq_send(soc, repbuf.data(), 0, 0)); // reply with an empty message
        continue;
      }
      qlog_debug("req: '%s'\n", req.DebugString().c_str());
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
      qlog_debug("Send msg of size(%lu): '%s'\n", reqbuf.size(), rep.DebugString().c_str());
    }
  }

  /**
   * Responsible for receiving PushData
   */
  void pull_worker_main(const char * endpoint, typename RM::Mode mode)
  {
    void * soc = new_zmq_socket(ZMQ_PULL);
    ZMQ_CALL(zmq_setsockopt(soc, ZMQ_RCVHWM, &pull_hwm, sizeof(pull_hwm)));
    ZMQ_CALL(zmq_setsockopt(soc, ZMQ_SNDHWM, &pull_hwm, sizeof(pull_hwm)));
    std::string buf(1024, '\0'); // TODO(qing): adjust default size
    if(mode == RM::Bind)
      ZMQ_CALL(zmq_bind(soc, endpoint));
    else if(mode == RM::Conn)
      ZMQ_CALL(zmq_connect(soc, endpoint));
    int size;
    while(true) {
      size = zmq_recv(soc, &buf[0], buf.size(), 0);
      if(size == -1) { // receive failed.
        int e = zmq_errno();
        qlog_warning("[ERRNO %d] '%s'.", e, zmq_strerror(e));
        return;
      }
      if(not (size <= (int)buf.size())) { // resize and wait for next
        qlog_warning("Resize buf from %lu to %d. Waiting for next push.\n", buf.size(), size);
        buf.resize(size);
        continue;
      }
      proto::Msg msg;
      if(not msg.ParseFromArray(buf.data(), size)) {
        std::ofstream out("pullbuf.msg", std::ios_base::binary | std::ios_base::trunc);
        out << std::string(buf.data(), size);
        qlog_warning("ParseFromArray(%p, %d) failed. Saved to 'pullbuf.msg'.\n", buf.data(), size);
        std::this_thread::sleep_for(std::chrono::seconds(1));
        continue;
      }
      qlog_debug("Received msg of size(%lu): '%s'\n", buf.size(), msg.DebugString().c_str());
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
   * Publish a byte string to clients
   */
  void pub_bytes(const std::string& topic, const std::string& message) {
    thread_local void * soc = nullptr;
    if(not soc) {
      if(pub_endpoint == "") {
        qlog_warning("To use %s(), please set server.pub_endpoint firstly\n", __func__);
        return;
      }
      soc = new_zmq_socket(ZMQ_PUB);
      ZMQ_CALL(zmq_setsockopt(soc, ZMQ_SNDHWM, &pub_hwm, sizeof(pub_hwm)));
      ZMQ_CALL(zmq_setsockopt(soc, ZMQ_RCVHWM, &pub_hwm, sizeof(pub_hwm)));
      #ifdef EPGM_EXPERIMENT
      int multicast_hops = 255; // default 1
      ZMQ_CALL(zmq_setsockopt(soc, ZMQ_MULTICAST_HOPS, &multicast_hops, sizeof(multicast_hops)));
      int rate = 1048576; // 1Gb
      ZMQ_CALL(zmq_setsockopt(soc, ZMQ_RATE, &rate, sizeof(rate)));
      int recovery_ivl = 200; // 200ms
      ZMQ_CALL(zmq_setsockopt(soc, ZMQ_RECOVERY_IVL, &recovery_ivl, sizeof(recovery_ivl)));
      #endif
      ZMQ_CALL(zmq_bind(soc, pub_endpoint.c_str()));
    }
    qlog_debug("Publish msg of size(%lu) in topic '%s'.\n", message.size(), topic.c_str());
    ZMQ_CALL(zmq_send(soc, topic.data(), topic.size(), ZMQ_SNDMORE));
    ZMQ_CALL(zmq_send(soc, message.data(), message.size(), 0));
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
  py::tuple py_get_data(uint32_t batch_size)
  {
    Mem mem(batch_size * rem.rollout_len * rem.entry_size);
    pyarr_float w({batch_size,});
    if(true) {
      py::gil_scoped_release release;
      rem.get_data(mem.data(), w.mutable_data(), batch_size);
    }
    py::list ret;
    for(uint32_t batch_idx=0; batch_idx<batch_size; batch_idx++) {
      py::list rollout;
      for(uint32_t step=0; step<rem.rollout_len; step++) {
        char * head = (char*)mem.data() + (batch_idx * rem.rollout_len + step) * rem.entry_size;
        py::tuple entry = py_unserialize_from_mem(head);
        rollout.append(entry);
      }
      ret.append(rollout);
    }
    return py::make_tuple(ret, w);
  }

};

typedef ReplayMemoryServer<RM> RMS;

