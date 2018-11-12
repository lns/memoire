#pragma once

#include <iostream>
#include "utils.hpp"
#include "replay_memory.hpp"
#include "qlog.hpp"
#include "msg.pb.h"
#include "py_serial.hpp"
#include "zmq_base.hpp"

template<class RM>
class ReplayMemoryClient : public ZMQBase {
public:
  const std::string uuid;
  std::string sub_endpoint;
  std::string req_endpoint;
  std::string push_endpoint;
  int sub_hwm;
  int req_hwm;
  int push_hwm;
  size_t sub_size;              ///< default size of buffer for sub

  proto::RepGetInfo info;
  uint32_t start_step;

public:
  /**
   * Initialization
   *
   * endpoint can be set to nullptr or "" to disable this protocal
   */
  ReplayMemoryClient(
      const std::string input_uuid)
    : uuid{input_uuid},
      start_step{0}
  {
    sub_hwm = req_hwm = 4;
    push_hwm = 64;
    sub_size = 1024;
  }

  ~ReplayMemoryClient() {}

  void get_info() {
    thread_local void * soc = nullptr;
    thread_local std::string reqbuf(1024, '\0');
    thread_local std::string repbuf(1024, '\0');
    if(not soc) {
      if(req_endpoint == "") {
        qlog_warning("To use %s(), please set client.req_endpoint firstly.\n", __func__);
        return;
      }
      soc = new_zmq_socket(ZMQ_REQ);
      ZMQ_CALL(zmq_setsockopt(soc, ZMQ_SNDHWM, &req_hwm, sizeof(req_hwm)));
      ZMQ_CALL(zmq_setsockopt(soc, ZMQ_RCVHWM, &req_hwm, sizeof(req_hwm)));
      ZMQ_CALL(zmq_connect(soc, req_endpoint.c_str()));
    }
    proto::Msg req;
    req.set_type(proto::REQ_GET_INFO);
    req.set_version(req.version());
    req.set_sender(uuid);
    req.SerializeToString(&reqbuf);
    proto::Msg rep;
    do {
      int size;
      ZMQ_CALL(zmq_send(soc, reqbuf.data(), reqbuf.size(), 0));
      qlog_debug("Send msg of size(%lu): '%s'\n", reqbuf.size(), req.DebugString().c_str()); 
      ZMQ_CALL(size = zmq_recv(soc, &repbuf[0], repbuf.size(), 0));
      if(not (size <= (int)repbuf.size())) { // resize and wait for next
        qlog_warning("Resize repbuf from %lu to %d. Resending req.\n", repbuf.size(), size);
        repbuf.resize(size);
        continue;
      }
      qlog_debug("Received msg of size(%d)\n", size);
      if(not rep.ParseFromArray(repbuf.data(), size)) {
        std::ofstream out("repbuf.msg", std::ios_base::binary | std::ios_base::trunc);
        out << std::string(repbuf.data(), size);
        qlog_warning("ParseFromArray(%p, %d) failed. Saved to 'repbuf.msg'.\n", repbuf.data(), size);
        std::this_thread::sleep_for(std::chrono::seconds(1));
        continue;
      }
      qlog_debug("rep: '%s'\n", rep.DebugString().c_str()); 
    } while(false);
    qassert(rep.version() == req.version());
    qassert(rep.type() == proto::REP_GET_INFO);
    // Get info
    info = rep.rep_get_info();
  }

  void push_data(void * data, uint32_t n_step, bool is_episode_end) {
    thread_local void * soc = nullptr;
    thread_local std::string pushbuf;
    if(not soc) {
      if(push_endpoint == "") {
        qlog_warning("To use %s(), please set client.push_endpoint firstly.\n", __func__);
        return;
      }
      soc = new_zmq_socket(ZMQ_PUSH);
      ZMQ_CALL(zmq_setsockopt(soc, ZMQ_SNDHWM, &push_hwm, sizeof(push_hwm)));
      ZMQ_CALL(zmq_setsockopt(soc, ZMQ_RCVHWM, &push_hwm, sizeof(push_hwm))); // not used
      ZMQ_CALL(zmq_connect(soc, push_endpoint.c_str()));
      pushbuf.resize(1024, '\0');
    }
    proto::Msg push;
    push.set_type(proto::PUSH_DATA);
    push.set_version(push.version());
    push.set_sender(uuid);
    auto * d = push.mutable_push_data();
    d->set_is_episode_end(is_episode_end);
    d->set_start_step(start_step);
    d->set_n_step(n_step);
    d->set_slot_index(info.slot_index());
    d->set_data(data, n_step * info.entry_size());
    push.SerializeToString(&pushbuf);
    qlog_debug("Send msg of size(%lu): '%s'\n", pushbuf.size(), push.DebugString().c_str()); 
    ZMQ_CALL(zmq_send(soc, pushbuf.data(), pushbuf.size(), 0));
    if(is_episode_end)
      start_step = 0;
    else
      start_step += n_step;
  }

  void push_log(const std::string& msg) {
    thread_local void * soc = nullptr;
    thread_local std::string pushbuf;
    if(not soc) {
      if(push_endpoint == "") {
        qlog_warning("To use %s(), please set client.push_endpoint firstly.\n", __func__);
        return;
      }
      soc = new_zmq_socket(ZMQ_PUSH);
      ZMQ_CALL(zmq_setsockopt(soc, ZMQ_SNDHWM, &push_hwm, sizeof(push_hwm)));
      ZMQ_CALL(zmq_setsockopt(soc, ZMQ_RCVHWM, &push_hwm, sizeof(push_hwm))); // not used
      ZMQ_CALL(zmq_connect(soc, push_endpoint.c_str()));
      pushbuf.resize(1024, '\0');
    }
    proto::Msg push;
    push.set_type(proto::PUSH_LOG);
    push.set_version(push.version());
    push.set_sender(uuid);
    auto * d = push.mutable_push_log();
    d->set_log(msg);
    push.SerializeToString(&pushbuf);
    qlog_debug("Send msg of size(%lu): '%s'\n", pushbuf.size(), push.DebugString().c_str()); 
    ZMQ_CALL(zmq_send(soc, pushbuf.data(), pushbuf.size(), 0));
  }

  /**
   * Blocked Receive of Bytestring
   */
  std::string sub_bytes(std::string topic) {
    thread_local void * soc = nullptr;
    thread_local Mem tpcbuf;
    thread_local Mem subbuf;
    thread_local std::string last_topic("");
    if(not soc) {
      if(sub_endpoint == "") {
        qlog_warning("To use %s(), please set client.sub_endpoint firstly.\n", __func__);
        return "";
      }
      soc = new_zmq_socket(ZMQ_SUB);
      ZMQ_CALL(zmq_setsockopt(soc, ZMQ_SNDHWM, &sub_hwm, sizeof(sub_hwm)));
      ZMQ_CALL(zmq_setsockopt(soc, ZMQ_RCVHWM, &sub_hwm, sizeof(sub_hwm)));
      #ifdef EPGM_EXPERIMENT
      int multicast_hops = 255; // default 1
      ZMQ_CALL(zmq_setsockopt(soc, ZMQ_MULTICAST_HOPS, &multicast_hops, sizeof(multicast_hops)));
      int rate = 1048576; // 1Gb
      ZMQ_CALL(zmq_setsockopt(soc, ZMQ_RATE, &rate, sizeof(rate)));
      int recovery_ivl = 200; // 200ms
      ZMQ_CALL(zmq_setsockopt(soc, ZMQ_RECOVERY_IVL, &recovery_ivl, sizeof(recovery_ivl)));
      #endif
      ZMQ_CALL(zmq_connect(soc, sub_endpoint.c_str()));
      tpcbuf.resize(256);
      subbuf.resize(sub_size);
    }
    if(topic.size() >= tpcbuf.size())
      qlog_error("topic: '%s' is too long.\n", topic.c_str());
    if(last_topic != "")
      ZMQ_CALL(zmq_setsockopt(soc, ZMQ_UNSUBSCRIBE, last_topic.c_str(), last_topic.size()));
    last_topic = topic;
    if(topic != "")
      ZMQ_CALL(zmq_setsockopt(soc, ZMQ_SUBSCRIBE, topic.c_str(), topic.size()));
    int size;
    while(true) {
      memset(tpcbuf.data(), 0, tpcbuf.size());
      memset(subbuf.data(), 0, subbuf.size());
      // Recv topic
      ZMQ_CALL(size = zmq_recv(soc, tpcbuf.data(), tpcbuf.size(), 0)); qassert(size <= (int)tpcbuf.size());
      // Recv Message
      ZMQ_CALL(size = zmq_recv(soc, subbuf.data(), subbuf.size(), 0));
      qlog_debug("Received msg of size(%d) in topic '%s'.\n", size, (char*)tpcbuf.data());
      // Check topic
      if(strcmp((const char*)tpcbuf.data(), topic.data())) { // topic mismatch, this should not happen
        qlog_error("topic mismatch: '%s' != '%s'\n", (const char *)tpcbuf.data(), topic.c_str());
      }
      // Check msg size
      if(not (size <= (int)subbuf.size())) { // resize and wait for next
        qlog_warning("Resize subbuf from %lu to %d and wait for the next msg.\n", subbuf.size(), size);
        subbuf.resize(size);
        continue;
      }
      else
        return std::string((char*)subbuf.data(), size);
    }
  }

  py::bytes py_sub_bytes(std::string topic) {
    std::string ret;
    if(true) {
      py::gil_scoped_release release;
      ret = sub_bytes(topic);
    }
    return py::bytes(ret);
  }

  void py_serialize_entry_to_mem(py::tuple entry, void * data) {
    qassert(info.view_size() == entry.size() + 1);
    char * head = static_cast<char*>(data);
    if(true) { // r
      BufView v(entry[entry.size()-3]);
      qassert(v.is_consistent_with(info.view(0)) or "Shape of r mismatch");
      head += v.to_memory(head);
    }
    if(true) { // p
      BufView v(entry[entry.size()-2]);
      qassert(v.is_consistent_with(info.view(1)) or "Shape of p mismatch");
      head += v.to_memory(head);
    }
    if(true) { // v
      BufView v(entry[entry.size()-1]);
      qassert(v.is_consistent_with(info.view(2)) or "Shape of v mismatch");
      head += v.to_memory(head);
    }
    if(true) { // q
      BufView v(entry[entry.size()-1]);
      qassert(v.is_consistent_with(info.view(3)) or "Shape of q mismatch");
      head += v.to_memory(head); // reuse data from v
    }
    // rest
    for(unsigned i=4; i<info.view_size(); i++) {
      BufView v(entry[i-4]);
      qassert(v.is_consistent_with(info.view(i)));
      head += v.to_memory(head);
    }
    qassert(head == static_cast<char*>(data) + info.entry_size());
  }

  void py_push_data(py::list data, bool is_episode_end) {
    uint32_t n_step = data.size();
    Mem mem(info.entry_size() * n_step);
    // Serialize data to memory
    for(unsigned i=0; i<n_step; i++) {
      char * head = (char*)mem.data() + i * info.entry_size();
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

