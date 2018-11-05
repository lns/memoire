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

namespace backward {
  backward::SignalHandling sh;
}

static void print_logo() __attribute__((constructor));

void print_logo() {
  fprintf(stderr, " Memoire v3, ver %lu, built on %s.\n", VERSION, __DATE__);
}

typedef py::array_t<float, py::array::c_style> pyarr_float;
typedef py::array_t<uint8_t, py::array::c_style> pyarr_uint8;

typedef BufView BV;

static qlib::LCG64 lcg64;

PYBIND11_MODULE(memoire /* module name */, m) {
  m.doc() = "Memoire, a distributed prioritized replay memory";

  m.def("get_descr", &get_descr);
  m.def("get_descr_nbytes", &get_descr_nbytes);
  m.def("descr_serialize", &descr_serialize);
  m.def("descr_unserialize", &descr_unserialize);
  m.def("pickle_dumps", &pickle_dumps);
  m.def("pickle_loads", &pickle_loads);

  m.def("get_host_ip", &get_host_ip);
  m.def("get_pid", &getpid);

  py::class_<BV>(m, "BufView")
    .def_readonly("ptr", &BV::ptr_)
    .def_readonly("itemsize", &BV::itemsize_)
    .def_readonly("format", &BV::format_)
    .def_readonly("shape", &BV::shape_)
    .def_readonly("stride", &BV::stride_)
    .def(py::init([](py::buffer b) {
        return AS_BV(b);
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
    .def_property_readonly("bundle_buf", &RM::bundle_buf)
    .def_property_readonly("reward_buf", &RM::reward_buf)
    .def_property_readonly("prob_buf",   &RM::prob_buf)
    .def_property_readonly("value_buf",  &RM::value_buf)
    .def_property_readonly("qvest_buf",  &RM::qvest_buf)
    .def_property_readonly("num_slot",   &RM::num_slot)
    .def_readonly("entry_size", &RM::entry_size)
    .def_readonly("max_step", &RM::max_step)
    .def_readonly("uuid", &RM::uuid)
    .def_readonly("ma_sqa", &RM::ma_sqa)
    .def_readonly("incre_episode", &RM::incre_episode)
    .def_readonly("incre_step", &RM::incre_step)
    .def_readwrite("max_episode", &RM::max_episode)
    .def_readwrite("priority_exponent", &RM::priority_exponent)
    .def_readwrite("mix_lambda", &RM::mix_lambda)
    .def_readwrite("rollout_len", &RM::rollout_len)
    .def_readwrite("step_discount", &RM::step_discount)
    .def_readwrite("discount_factor", &RM::discount_factor)
    .def_readwrite("reward_coeff", &RM::reward_coeff)
    ;
 
  py::class_<RMC>(m, "ReplayMemoryClient")
    .def(py::init<const std::string&, const std::string&, const std::string&>(),
        "uuid"_a, "req_endpoint"_a, "push_endpoint"_a)
    .def_readonly("x_descr_pickle", &RMC::x_descr_pickle)
    .def_readonly("remote_slot_index", &RMC::remote_slot_index)
    .def_readonly("entry_size", &RMC::entry_size)
    .def_readonly("view", &RMC::view)
    .def_readonly("req_endpoint", &RMC::req_endpoint)
    .def_readonly("push_endpoint", &RMC::push_endpoint)
    .def_readonly("uuid", &RMC::uuid)
    .def("close", &RMC::close, py::call_guard<py::gil_scoped_release>())
    .def("get_info", &RMC::get_info, py::call_guard<py::gil_scoped_release>())
    .def("push_data", &RMC::py_push_data)
    ;

  py::class_<RMS>(m, "ReplayMemoryServer")
    .def_readonly("rem", &RMS::rem)
    .def(py::init([](py::tuple entry, size_t max_step, size_t n_slot) {
        // We require entry[-3] is reward, entry[-2] is prob, and entry[-1] is value.
        py::list x = py::list(entry)[py::slice(0,-3,1)];
        py::object descr = get_descr(x);
        pyarr_uint8 bundle = descr_serialize(x, descr);
        std::string x_descr_pickle = pickle_dumps(descr);
        BV view[N_VIEW];
        view[0] = AS_BV(bundle);
        view[1] = AS_BV(entry[entry.size()-3]); // r
        view[2] = AS_BV(entry[entry.size()-2]); // p
        view[3] = AS_BV(entry[entry.size()-1]); // v
        view[4] = AS_BV(entry[entry.size()-1]); // q
        return(std::unique_ptr<RMS>(new RMS(view,max_step,n_slot,&lcg64,x_descr_pickle)));
      }), "entry"_a, "max_step"_a, "n_slot"_a)
    .def("close", &RMS::close, py::call_guard<py::gil_scoped_release>())
    .def("get_data", &RMS::py_get_data)
    .def("print_info", [](RMS& rms) { rms.print_info(stderr); }, py::call_guard<py::gil_scoped_release>())
    .def("rep_worker_main",  &RMS::rep_worker_main,    py::call_guard<py::gil_scoped_release>())
    .def("pull_worker_main", &RMS::pull_worker_main,   py::call_guard<py::gil_scoped_release>())
    ;

  py::class_<Proxy>(m, "Proxy")
    .def(py::init<>())
    .def("rep_proxy_main",   &Proxy::rep_proxy_main,   py::call_guard<py::gil_scoped_release>())
    .def("pull_proxy_main",  &Proxy::pull_proxy_main,  py::call_guard<py::gil_scoped_release>())
    .def("pub_proxy_main",   &Proxy::pub_proxy_main,   py::call_guard<py::gil_scoped_release>())
    ;

}

