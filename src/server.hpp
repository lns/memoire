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
#include "proxy.hpp"
#include "msg.pb.h"
#include "py_serial.hpp"

template<class RM>
class ReplayMemoryServer : public Proxy {
public:
  RM rem;
  proto::Descriptor desc;
  std::unordered_map<std::string, uint32_t> m;

  std::string pub_endpoint;
  int pub_hwm;
  int rep_hwm;
  int pull_hwm;
  size_t pull_buf_size;

  std::string logfile_path;
  FILE * logfile;
  std::mutex logfile_mutex;

  ReplayMemoryServer(const BufView * vw, size_t max_step, size_t n_slot, 
      qlib::RNG * prt_rng, std::string desc_serial)
    : rem{vw, max_step, n_slot, prt_rng},
      logfile{nullptr}
  {
    pub_hwm = 4;
    rep_hwm = pull_hwm = 1024;
    pull_buf_size = 1048768;
    //
    qassert(desc.ParseFromString(desc_serial));
  }

  ~ReplayMemoryServer() {
    if(logfile)
      fclose(logfile);
    logfile = nullptr;
  }

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
  void print_info() const
  {
    rem.print_info(stderr);
  }

  void set_logfile(const std::string path, const std::string mode) {
    std::lock_guard<std::mutex> guard(logfile_mutex);
    if(logfile)
      fclose(logfile);
    logfile = fopen(path.c_str(), mode.c_str());
    if(not logfile)
      qlog_error("fopen(%s,%s) failed.\n", path.c_str(), mode.c_str());
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
        *p->mutable_desc() = desc;
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
    std::string buf(pull_buf_size, '\0');
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
        qlog_warning("Pushed data lost due to insufficient push_buf_size."
            "Resize buf from %lu to %d. Waiting for next push.\n", buf.size(), size);
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
        qassert(msg.has_push_data());
        const auto m = msg.push_data();
        rem.add_data(m.slot_index(), (void*)m.data().data(), m.start_step(), m.n_step(), m.is_episode_end());
      }
      else if(msg.type() == proto::PUSH_LOG) {
        if(logfile) {
          std::lock_guard<std::mutex> guard(logfile_mutex);
          qassert(msg.has_push_log());
          const auto m = msg.push_log();
          fwrite((void*)m.log().data(), 1, m.log().size(), logfile);
          fflush(logfile);
        }
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
   * Get data and weight: (data, weight)
   */
  py::tuple py_get_data(uint32_t batch_size)
  {
    pyarr_float w({batch_size,});
    py::list ret;
    std::vector<BufView> v;
    // Add bundle
    for(unsigned i=0; i<desc.view_size(); i++) {
      BufView t(desc.view(i));
      t.shape_.insert(t.shape_.begin(), static_cast<ssize_t>(rem.rollout_len));
      t.shape_.insert(t.shape_.begin(), static_cast<ssize_t>(batch_size));
      t.make_c_stride();
      ret.append(t.new_array());
      v.emplace_back(py::cast<py::buffer>(ret[i]));
    }
    // Add r,p,v,q
    for(int i=1; i<N_VIEW; i++) {
      BufView t(rem.view[i]);
      t.shape_.insert(t.shape_.begin(), static_cast<ssize_t>(rem.rollout_len));
      t.shape_.insert(t.shape_.begin(), static_cast<ssize_t>(batch_size));
      t.make_c_stride();
      ret.append(t.new_array());
      v.emplace_back(py::cast<py::buffer>(ret[i]));
    }
    if(true) {
      py::gil_scoped_release release;
      // Fill data
      Mem mem(batch_size * rem.rollout_len * rem.entry_size);
      rem.get_data(mem.data(), w.mutable_data(), batch_size);
      size_t offset = 0;
      char * head = static_cast<char*>(mem.data());
      for(unsigned b=0; b<batch_size; b++) {
        for(unsigned r=0; r<rem.rollout_len; r++) {
          for(unsigned i=0; i<ret.size(); i++) {
            offset += v[i][b][r].from_memory(head + offset); // performance issue of operator[] ?
          }
        }
      }
      qassert(offset == batch_size * rem.rollout_len * rem.entry_size);
    }
    return py::make_tuple(py::tuple(ret), w);
  }

};

typedef ReplayMemoryServer<RM> RMS;

