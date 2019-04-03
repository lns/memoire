#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdio>
#include "replay_memory.hpp"
#include "client.hpp"
#include "server.hpp"
#include "proxy.hpp"
#include "py_serial.hpp"

namespace py = pybind11;
using namespace py::literals;

#ifdef PRINT_DEBUG
namespace backward {
  backward::SignalHandling sh;
}
#endif

static void print_logo() __attribute__((constructor));

void print_logo() {
  fprintf(stderr, " Memoire v4, ver %lu, built on %s.\n", VERSION, __DATE__);
  qlog_set_print_time(true);
}

typedef py::array_t<float, py::array::c_style> pyarr_float;
typedef py::array_t<uint8_t, py::array::c_style> pyarr_uint8;

typedef BufView BV;

static qlib::LCG64 lcg64;

PYBIND11_MODULE(memoire /* module name */, m) {
  m.doc() = "Memoire, a distributed prioritized replay memory";

  m.def("get_host_ip", &get_host_ip);
  m.def("get_pid", &getpid);

  py::class_<BV>(m, "BufView")
    .def_readonly("ptr", &BV::ptr_)
    .def_readonly("itemsize", &BV::itemsize_)
    .def_readonly("format", &BV::format_)
    .def_readonly("shape", &BV::shape_)
    .def_readonly("stride", &BV::stride_)
    .def_property_readonly("dtype",  [](BV& self) {
        return py::dtype(self.format_);
      })
    .def(py::init([](py::buffer b) {
        return BufView(b);
      }), "buffer"_a)
    .def("__str__", &BV::str)
    .def("as_array", [](BV& self) {
        return py::array(py::dtype(self.format_), self.shape_, self.stride_, self.ptr_);
      })
    ;

  py::enum_<typename RM::Mode>(m, "Mode", py::arithmetic())
    .value("Conn", RM::Conn)
    .value("Bind", RM::Bind)
    .export_values()
    ;

  py::class_<RM>(m, "ReplayMemory")
    .def_property_readonly("reward_buf", &RM::reward_buf)
    .def_property_readonly("prob_buf",   &RM::prob_buf)
    .def_property_readonly("value_buf",  &RM::value_buf)
    .def_property_readonly("qvest_buf",  &RM::qvest_buf)
    .def_property_readonly("num_slot",   &RM::num_slot)
    .def_readonly("view", &RM::view)
    .def_readonly("entry_size", &RM::entry_size)
    .def_readonly("max_step", &RM::max_step)
    .def_readonly("uuid", &RM::uuid)
    .def_readonly("ma_sqa", &RM::ma_sqa)
    .def_readonly("total_episodes", &RM::total_episodes)
    .def_readonly("total_steps", &RM::total_steps)
    .def_readwrite("max_episode", &RM::max_episode)
    .def_readwrite("priority_exponent", &RM::priority_exponent)
    .def_readwrite("mix_lambda", &RM::mix_lambda)
    .def_readwrite("rollout_len", &RM::rollout_len)
    .def_readwrite("do_padding", &RM::do_padding)
    .def_readwrite("priority_decay", &RM::priority_decay)
    .def_readwrite("traceback_threshold", &RM::traceback_threshold)
    .def_readwrite("discount_factor", &RM::discount_factor)
    .def_readwrite("reward_coeff", &RM::reward_coeff)
    ;

  py::class_<Proxy>(m, "Proxy")
    .def(py::init<>())
    .def("rep_proxy_main",   &Proxy::rep_proxy_main,   py::call_guard<py::gil_scoped_release>())
    .def("pull_proxy_main",  &Proxy::pull_proxy_main,  py::call_guard<py::gil_scoped_release>())
    .def("pub_proxy_main",   &Proxy::pub_proxy_main,   py::call_guard<py::gil_scoped_release>())
    ;
 
  py::class_<RMC>(m, "ReplayMemoryClient")
    .def(py::init<const std::string&>(), "uuid"_a)
    .def_property_readonly("slot_index", [](RMC& rmc) { return rmc.info.slot_index(); })
    .def_property_readonly("entry_size", [](RMC& rmc) { return rmc.info.entry_size(); })
    .def_property_readonly("view_size",  [](RMC& rmc) { return rmc.info.view_size(); })
    .def_property_readonly("template", [](RMC& rmc) {
        py::list ret;
        for(int i=4; i<rmc.info.view_size(); i++)
          ret.append(BufView(rmc.info.view(i)).new_array());
        for(int i=0; i<3; i++)
          ret.append(BufView(rmc.info.view(i)).new_array());
        return py::tuple(ret);
    })   
    .def_property("push_length", &RMC::get_push_length, &RMC::set_push_length)
    .def_readwrite("sub_endpoint", &RMC::sub_endpoint)
    .def_readwrite("req_endpoint", &RMC::req_endpoint)
    .def_readwrite("push_endpoint", &RMC::push_endpoint)
    .def_readwrite("sub_hwm", &RMC::sub_hwm)
    .def_readwrite("req_hwm", &RMC::req_hwm)
    .def_readwrite("push_hwm", &RMC::push_hwm)
    .def_readwrite("sub_buf_size", &RMC::sub_buf_size)
    .def_readonly("uuid", &RMC::uuid)
    .def("view", [](RMC& rmc, int i) { return BufView(rmc.info.view(i)); })
    .def("close",            &RMC::close,              py::call_guard<py::gil_scoped_release>())
    .def("get_info",         &RMC::get_info,           py::call_guard<py::gil_scoped_release>())
    .def("add_entry",        &RMC::py_add_entry)
    .def("sub_bytes",        &RMC::py_sub_bytes)
    .def("push_log",         &RMC::push_log,           py::call_guard<py::gil_scoped_release>())
    .def("push_worker_main", &RMC::push_worker_main,   py::call_guard<py::gil_scoped_release>())
    ;

  py::class_<RMS>(m, "ReplayMemoryServer")
    .def_readonly("rem", &RMS::rem)
    .def_readwrite("pub_endpoint", &RMS::pub_endpoint)
    .def_readwrite("pub_hwm", &RMS::pub_hwm)
    .def_readwrite("rep_hwm", &RMS::rep_hwm)
    .def_readwrite("pull_hwm", &RMS::pull_hwm)
    .def_readwrite("pull_buf_size", &RMS::pull_buf_size)
    .def(py::init([](py::tuple entry, size_t max_step, size_t n_slot) {
        // We require entry[-3] is reward, entry[-2] is prob, and entry[-1] is value.
        std::deque<BufView> view;
        view.emplace_back(entry[entry.size()-3]); // r
        view.emplace_back(entry[entry.size()-2]); // p
        view.emplace_back(entry[entry.size()-1]); // v
        view.emplace_back(entry[entry.size()-1]); // q
        for(int i=0; i<static_cast<int>(entry.size())-3; i++)
          view.emplace_back(entry[i]);
        return(std::unique_ptr<RMS>(new RMS(view,max_step,n_slot,&lcg64)));
      }), "entry"_a, "max_step"_a, "n_slot"_a)
    .def("close",            &RMS::close,              py::call_guard<py::gil_scoped_release>())
    .def("get_data",         &RMS::py_get_data)
    .def("pub_bytes",        &RMS::pub_bytes,          py::call_guard<py::gil_scoped_release>())
    .def("set_logfile",      &RMS::set_logfile,        py::call_guard<py::gil_scoped_release>())
    .def("print_info",       &RMS::print_info,         py::call_guard<py::gil_scoped_release>())
    .def("rep_worker_main",  &RMS::rep_worker_main,    py::call_guard<py::gil_scoped_release>())
    .def("pull_worker_main", &RMS::pull_worker_main,   py::call_guard<py::gil_scoped_release>())
    // inherited from Proxy, to share zmq_ctx
    .def("rep_proxy_main",   &Proxy::rep_proxy_main,   py::call_guard<py::gil_scoped_release>())
    .def("pull_proxy_main",  &Proxy::pull_proxy_main,  py::call_guard<py::gil_scoped_release>())
    .def("pub_proxy_main",   &Proxy::pub_proxy_main,   py::call_guard<py::gil_scoped_release>())
    ;

}

