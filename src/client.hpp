#pragma once

#include <iostream>
#include <memory>
#include "utils.hpp"
#include "replay_memory.hpp"
#include "qlog.hpp"
#include "msg.pb.h"
#include "zmq_base.hpp"
#include "bounded_vector.hpp"

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
  size_t sub_buf_size;              ///< default size of buffer for sub
  size_t push_length;

  proto::RepGetInfo info;

protected:
  typedef BoundedVector<bool, typename RM::DataEntry> Queue;
  std::unique_ptr<Queue> queue;

public:
  /**
   * Initialization
   *
   * endpoint can be set to nullptr or "" to disable this protocal
   */
  ReplayMemoryClient(
      const std::string input_uuid)
    : uuid{input_uuid}
  {
    sub_hwm = req_hwm = 4;
    push_hwm = 256;
    sub_buf_size = 1024;
    push_length = 32;
  }

  ~ReplayMemoryClient() {}

  size_t get_push_length() const {
    return push_length;
  }

  void set_push_length(size_t new_length) {
    push_length = new_length;
    queue = std::unique_ptr<Queue>(new Queue(info.entry_size(), 4*push_length));
  }

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
    qassert(req.SerializeToString(&reqbuf));
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
    set_push_length(push_length); // resize queue
  }

  void push_log(const std::string& msg) {
    thread_local void * soc = nullptr;
    thread_local std::string pushbuf;
    thread_local Mem repbuf;
    if(not soc) {
      if(push_endpoint == "") {
        qlog_warning("To use %s(), please set client.push_endpoint firstly.\n", __func__);
        return;
      }
      soc = new_zmq_socket(ZMQ_REQ);
      ZMQ_CALL(zmq_setsockopt(soc, ZMQ_SNDHWM, &push_hwm, sizeof(push_hwm)));
      ZMQ_CALL(zmq_setsockopt(soc, ZMQ_RCVHWM, &push_hwm, sizeof(push_hwm))); // not used
      ZMQ_CALL(zmq_connect(soc, push_endpoint.c_str()));
      pushbuf.resize(1024, '\0');
      repbuf.resize(1024);
    }
    proto::Msg push;
    push.set_type(proto::REQ_PUSH_LOG);
    push.set_version(push.version());
    push.set_sender(uuid);
    auto * d = push.mutable_req_push_log();
    d->set_log(msg);
    qassert(push.SerializeToString(&pushbuf));
    qlog_debug("Send msg of size(%lu): '%s'\n", pushbuf.size(), push.DebugString().c_str()); 
    ZMQ_CALL(zmq_send(soc, pushbuf.data(), pushbuf.size(), 0));
    int size;
    qlog_debug("Waiting for response ..\n");
    ZMQ_CALL(size = zmq_recv(soc, repbuf.data(), repbuf.size(), 0));
    qlog_debug("Received msg of size(%d).\n", size);
    // TODO: check returned 'succ'
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
      subbuf.resize(sub_buf_size);
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

  void py_add_entry(py::tuple entry, bool is_term) {
    thread_local Mem mem(info.entry_size());
    std::vector<BufView> view;
    view.emplace_back(entry[entry.size()-3]); // r
    view.emplace_back(entry[entry.size()-2]); // p
    view.emplace_back(entry[entry.size()-1]); // v
    view.emplace_back(entry[entry.size()-1]); // q
    for(unsigned i=4; i<info.view_size(); i++)
      view.emplace_back(entry[i-4]);
    if(true) {
      py::gil_scoped_release release;
      // Check & Copy
      char * head = reinterpret_cast<char*>(mem.data());
      for(unsigned i=0; i<view.size(); i++) {
        qassert(view[i].is_consistent_with(info.view(i)));
        head += view[i].to_memory(head);
      }
      queue->put(is_term, reinterpret_cast<typename RM::DataEntry*>(mem.data()));
    }
  }

  void push_worker_main() {
    // This mainloop should be executed by only one thread.
    thread_local void * soc = nullptr;
    thread_local std::string pushbuf;
    thread_local Mem repbuf;
    thread_local uint32_t start_step = 0;
    if(not soc) {
      if(push_endpoint == "") {
        qlog_warning("To use %s(), please set client.push_endpoint firstly.\n", __func__);
        return;
      }
      soc = new_zmq_socket(ZMQ_REQ);
      ZMQ_CALL(zmq_setsockopt(soc, ZMQ_SNDHWM, &push_hwm, sizeof(push_hwm)));
      ZMQ_CALL(zmq_setsockopt(soc, ZMQ_RCVHWM, &push_hwm, sizeof(push_hwm))); // not used
      ZMQ_CALL(zmq_connect(soc, push_endpoint.c_str()));
      pushbuf.resize(1024, '\0');
      repbuf.resize(1024);
    }
    while(true) {
      proto::Msg push;
      push.set_type(proto::REQ_PUSH_DATA);
      push.set_version(push.version());
      push.set_sender(uuid);
      auto * d = push.mutable_req_push_data();
      d->set_start_step(start_step);
      d->set_n_step(push_length);
      d->set_slot_index(info.slot_index());
      d->clear_term();
      // Get push_length entries
      Mem mem(info.entry_size() * push_length);
      size_t count = 0;
      char * head = reinterpret_cast<char*>(mem.data());
      while(count < push_length) {
        bool term;
        queue->get(term, reinterpret_cast<typename RM::DataEntry*>(head + count * info.entry_size()));
        d->add_term(term);
        count += 1;
        start_step += 1;
        if(term) {
          qlog_info("Episode length: %u\n", start_step);
          start_step = 0;
        }
      }
      qassert(count == push_length and (size_t)d->term_size() == push_length);
      d->set_data(mem.data(), count * info.entry_size());
      qassert(push.SerializeToString(&pushbuf));
      qlog_debug("Send msg of size(%lu): '%s'\n", pushbuf.size(), push.DebugString().c_str()); 
      ZMQ_CALL(zmq_send(soc, pushbuf.data(), pushbuf.size(), 0));
      int size;
      qlog_debug("Waiting for response ..\n");
      ZMQ_CALL(size = zmq_recv(soc, repbuf.data(), repbuf.size(), 0));
      qlog_debug("Received msg of size(%d).\n", size);
      // TODO: check returned 'succ'
    }
  }

};

typedef ReplayMemoryClient<RM> RMC;

