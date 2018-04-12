#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstdio>
#include "replay_memory.hpp"
#include "server.hpp"
#include "client.hpp"

namespace py = pybind11;
using namespace py::literals;

namespace backward {
  backward::SignalHandling sh;
}

static void print_logo() __attribute__((constructor));

void print_logo() {
  fprintf(stderr, " Memoire, ver 18.04.12, built on %s.\n",__DATE__);
}

typedef py::array_t<float, py::array::c_style> pyarr_float;
typedef py::array_t<int, py::array::c_style> pyarr_int;
typedef py::array_t<uint64_t, py::array::c_style> pyarr_ulong;
typedef py::array_t<uint8_t, py::array::c_style> pyarr_char;

typedef ReplayMemory<uint8_t, float, float> RM;
typedef ReplayMemoryClient<RM> RMC;
typedef ReplayMemoryServer<RM> RMS;

static qlib::LCG64 lcg64;

PYBIND11_MODULE(memoire /* module name */, m) {
  m.doc() = "Memoire"; // TODO

  py::class_<RM>(m, "ReplayMemory")
    .def_readwrite("discount_factor", &RM::discount_factor)
    .def_readwrite("priority_exponent", &RM::priority_exponent)
    .def_readwrite("td_lambda", &RM::lambda)
    .def_readwrite("frame_stack", &RM::frame_stack)
    .def_readwrite("multi_step", &RM::multi_step)
    .def_readonly("state_size", &RM::state_size)
    .def_readonly("action_size", &RM::action_size)
    .def_readonly("reward_size", &RM::reward_size)
    .def_readonly("prob_size", &RM::prob_size)
    .def_readonly("value_size", &RM::value_size)
    .def_readonly("entry_size", &RM::entry_size)
    .def_readonly("max_episode", &RM::max_episode)
    .def_readonly("epi_max_len", &RM::epi_max_len)
    .def(py::init([](size_t ss, size_t as, size_t rs, size_t ps, size_t vs, size_t me, size_t eml) {
            return std::unique_ptr<RM>(new RM(ss,as,rs,ps,vs,me,eml,&lcg64));
          }),
        "state_size"_a,
        "action_size"_a,
        "reward_size"_a,
        "prob_size"_a,
        "value_size"_a,
        "max_episode"_a,
        "episode_max_length"_a)
    .def("print_info", [](RM& rem) { rem.print_info(); })
    .def("num_episode", &RM::num_episode)
    .def("new_episode", &RM::new_episode)
    .def("close_episode", &RM::close_episode, "epi_idx"_a, "do_update_value"_a = true, "do_update_wieght"_a = true)
    .def("clear", &RM::clear)
    .def("add_entry", [](RM& rem,
          size_t epi_idx,
          pyarr_char s,
          pyarr_float a,
          pyarr_float r,
          pyarr_float p,
          pyarr_float v,
          float weight) {
        qassert(s.size() == (long)rem.state_size);
        qassert(a.size() == (long)rem.action_size);
        qassert(r.size() == (long)rem.reward_size);
        qassert(p.size() == (long)rem.prob_size);
        qassert(v.size() == (long)rem.value_size);
        rem.add_entry(epi_idx, s.data(), a.data(), r.data(), p.data(), v.data(), weight);
        }, "epi_idx"_a, "state"_a, "action"_a, "reward"_a, "prob"_a, "value"_a, "weight"_a)
    .def("get_batch", [](RM& rem,
          size_t batch_size) {
        size_t stack_size = rem.frame_stack;
        pyarr_char prev_s({batch_size, stack_size, rem.state_size});
        pyarr_float prev_a({batch_size, rem.action_size});
        pyarr_float prev_r({batch_size, rem.reward_size});
        pyarr_float prev_p({batch_size, rem.prob_size});
        pyarr_float prev_v({batch_size, rem.value_size});
        pyarr_char next_s({batch_size, stack_size, rem.state_size});
        pyarr_float next_a({batch_size, rem.action_size});
        pyarr_float next_r({batch_size, rem.reward_size});
        pyarr_float next_p({batch_size, rem.prob_size});
        pyarr_float next_v({batch_size, rem.value_size});
        pyarr_int epi_idx_arr({batch_size,});
        pyarr_int entry_idx_arr({batch_size,});
        pyarr_ulong epi_inc_arr({batch_size,});
        pyarr_float entry_weight_arr({batch_size,});
        bool ret = rem.get_batch(batch_size,
          prev_s.mutable_data(),
          prev_a.mutable_data(),
          prev_r.mutable_data(),
          prev_p.mutable_data(),
          prev_v.mutable_data(),
          next_s.mutable_data(),
          next_a.mutable_data(),
          next_r.mutable_data(),
          next_p.mutable_data(),
          next_v.mutable_data(),
          epi_idx_arr.mutable_data(),
          entry_idx_arr.mutable_data(),
          epi_inc_arr.mutable_data(),
          entry_weight_arr.mutable_data());
        if(not ret)
          throw std::runtime_error("get_batch() failed.");
        auto prev = std::make_tuple(prev_s, prev_a, prev_r, prev_p, prev_v);
        auto next = std::make_tuple(next_s, next_a, next_r, next_p, next_v);
        auto info = std::make_tuple(epi_idx_arr, entry_idx_arr, epi_inc_arr, entry_weight_arr);
        return std::make_tuple(prev,next,info);
      },"batch_size"_a)
    .def("update_weight", [](RM& rem,
          size_t batch_size,
          pyarr_int epi_idx_arr,
          pyarr_int entry_idx_arr,
          pyarr_ulong epi_inc_arr,
          pyarr_float entry_weight_arr) {
        rem.update_weight(batch_size,
          epi_idx_arr.data(),
          entry_idx_arr.data(),
          epi_inc_arr.data(),
          entry_weight_arr.data());
      },"batch_size"_a, "epi_idx_arr"_a, "entry_idx_arr"_a, "epi_inc_arr"_a, "entry_weight_arr"_a);

  py::enum_<typename RM::Mode>(m, "Mode", py::arithmetic())
    .value("Conn", RM::Conn)
    .value("Bind", RM::Bind)
    .export_values();

  py::class_<RMC>(m, "ReplayMemoryClient")
    .def(py::init<const char*, const char*, size_t>(), "req_endpoint"_a, "push_endpoint"_a, "rem_max_capacity"_a)
    .def_readonly("prm", &RMC::prm)
    .def("push_episode_to_remote", [](RMC& c, int epi_idx) {
        py::gil_scoped_release release;
        c.push_episode_to_remote(epi_idx);
        }, "epi_idx"_a);

  py::class_<RMS>(m, "ReplayMemoryServer")
    .def(py::init<RM*>(), "replay_memory"_a)
    .def("rep_worker_main", [](RMS& s, const char * ep, typename RM::Mode m) {
        py::gil_scoped_release release;
        s.rep_worker_main(ep,m);
        }, "endpoint"_a, "mode"_a)
    .def("pull_worker_main", [](RMS& s, const char * ep, typename RM::Mode m) {
        py::gil_scoped_release release;
        s.pull_worker_main(ep,m);
        }, "endpoint"_a, "mode"_a)
    .def("rep_proxy_main", [](RMS& s, const char * f, const char * b) {
        py::gil_scoped_release release;
        s.rep_proxy_main(f,b);
        }, "front_ep"_a, "back_ep"_a)
    .def("pull_proxy_main", [](RMS& s, const char * f, const char * b) {
        py::gil_scoped_release release;
        s.pull_proxy_main(f,b);
        }, "front_ep"_a, "back_ep"_a);

}

