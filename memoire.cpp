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
    .def_readwrite("pre_skip", &RM::pre_skip)
    .def_readwrite("post_skip", &RM::post_skip)
    .def_readwrite("step_discount", &RM::step_discount)
    .def_readwrite("discount_factor", &RM::discount_factor)
    .def_readwrite("reward_coeff", &RM::reward_coeff)
    ;
 
  py::class_<RMC>(m, "ReplayMemoryClient")
    .def(py::init<const std::string&, const std::string&, const std::string&>(),
        "req_endpoint"_a, "push_endpoint"_a, "uuid"_a)
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
    .def(py::init([](py::tuple entry, size_t max_step, size_t n_slot, std::string uuid) {
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
        return(std::unique_ptr<RMS>(new RMS(view,max_step,n_slot,&lcg64,uuid,x_descr_pickle)));
      }), "entry"_a, "max_step"_a, "n_slot"_a, "uuid"_a)
    .def("close", &RMS::close, py::call_guard<py::gil_scoped_release>())
    .def("get_data", &RMS::py_get_data)
    .def("print_info", [](RMS& rms) { rms.print_info(stderr); }, py::call_guard<py::gil_scoped_release>())
    .def("rep_worker_main",  &RMS::rep_worker_main,    py::call_guard<py::gil_scoped_release>())
    .def("pull_worker_main", &RMS::pull_worker_main,   py::call_guard<py::gil_scoped_release>())
    ;

  py::class_<Proxy>(m, "Proxy")
    .def("rep_proxy_main",   &Proxy::rep_proxy_main,   py::call_guard<py::gil_scoped_release>())
    .def("pull_proxy_main",  &Proxy::pull_proxy_main,  py::call_guard<py::gil_scoped_release>())
    .def("pub_proxy_main",   &Proxy::pub_proxy_main,   py::call_guard<py::gil_scoped_release>())
    ;

/*
  py::class_<RM>(m, "ReplayMemory")
    .def_property_readonly("state_buf",  &RM::state_buf)
    .def_property_readonly("action_buf", &RM::action_buf)
    .def_property_readonly("reward_buf", &RM::reward_buf)
    .def_property_readonly("prob_buf",   &RM::prob_buf)
    .def_property_readonly("value_buf",  &RM::value_buf)
    .def_property_readonly("qvest_buf",  &RM::qvest_buf)
    .def_property_readonly("info_buf",   &RM::info_buf)
    .def_readonly("entry_size", &RM::entry_size)
    .def_readonly("max_step", &RM::max_step)
    .def_readonly("uuid", &RM::uuid)
    .def_readonly("new_offset", &RM::new_offset)
    .def_readonly("new_length", &RM::new_length)
    .def_readonly("incre_episode", &RM::incre_episode)
    .def_readonly("incre_step", &RM::incre_step)
    .def_readonly("cur_step", &RM::cur_step)
    .def_readwrite("max_episode", &RM::max_episode)
    .def_readwrite("priority_exponent", &RM::priority_exponent)
    .def_readwrite("mix_lambda", &RM::mix_lambda)
    .def_readwrite("frame_stack", &RM::frame_stack)
    .def_readwrite("multi_step", &RM::multi_step)
    .def_readwrite("cache_size", &RM::cache_size)
    .def_readwrite("reuse_cache", &RM::reuse_cache)
    .def_readwrite("autosave_step", &RM::autosave_step)
    .def_readwrite("replace_data", &RM::replace_data)
    .def_readwrite("step_discount", &RM::step_discount)
    .def_property("discount_factor",
      [](RM& rm) {
        py::list l;
        for(int i=0; i<std::min<int>(rm.reward_buf().size(), MAX_RWD_DIM); i++)
          l.append(rm.discount_factor[i]);
        return l;
      },
      [](RM& rm, py::list l) {
        for(int i=0; i<std::min<int>(MAX_RWD_DIM, l.size()); i++)
          rm.discount_factor[i] = py::cast<float>(l[i]);
      })
    .def_property("reward_coeff",
      [](RM& rm) {
        py::list l;
        for(int i=0; i<std::min<int>(rm.reward_buf().size(), MAX_RWD_DIM); i++)
          l.append(rm.reward_coeff[i]);
        return l;
      },
      [](RM& rm, py::list l) {
        for(int i=0; i<std::min<int>(MAX_RWD_DIM, l.size()); i++)
          rm.reward_coeff[i] = py::cast<float>(l[i]);
      })
    .def_property("cache_flags",
      [](RM& rm) {
        py::list l;
        for(int i=0; i<2*N_VIEW; i++)
          l.append(rm.cache_flags[i]);
        return l;
      },
      [](RM& rm, py::list l) {
        for(int i=0; i<2*N_VIEW and i<(int)l.size(); i++)
          rm.cache_flags[i] = py::cast<uint8_t>(l[i]);
      })
    .def(py::init([](py::buffer s, py::buffer a, py::buffer r, py::buffer p,
            py::buffer v, py::buffer q, py::buffer i, size_t capa) {
          BV view[N_VIEW];
          view[0] = AS_BV(s);
          view[1] = AS_BV(a);
          view[2] = AS_BV(r);
          view[3] = AS_BV(p);
          view[4] = AS_BV(v);
          view[5] = AS_BV(q);
          view[6] = AS_BV(i);
          return std::unique_ptr<RM>(new RM(view,capa,&lcg64));
          }),
        "state_buf"_a,
        "action_buf"_a,
        "reward_buf"_a,
        "prob_buf"_a,
        "value_buf"_a,
        "qvest_buf"_a,
        "info_buf"_a,
        "max_step"_a)
    .def("print_info", [](RM& rem) { rem.print_info(); })
    .def("num_episode", &RM::num_episode)
    .def("new_episode", &RM::new_episode)
    .def("close_episode", &RM::close_episode, "episodic_weight_multiplier"_a = 1.0,
        "do_update_value"_a = true, "do_update_wieght"_a = true)
    .def("add_entry", [](RM& rem,
          py::buffer s,
          py::buffer a,
          py::buffer r,
          py::buffer p,
          py::buffer v,
          py::buffer i) {
        rem.add_entry(AS_BV(s), AS_BV(a), AS_BV(r), AS_BV(p), AS_BV(v), AS_BV(i));
        }, "state"_a, "action"_a, "reward"_a, "prob"_a, "value"_a, "info"_a)
    .def("get_entry_buf", [](RM& rem) {
        std::vector<py::array> entry;
        for(int i=0; i<N_VIEW; i++) {
          const auto& v = rem.view[i];
          if(i == 5) // skip qvest
            continue;
          entry.emplace_back(py::dtype(v.format_), v.shape_, v.stride_, nullptr);
        }
        return entry;
      })
    ;

  py::class_<RMS>(m, "ReplayMemoryServer")
    .def_readonly("rem", &RMS::rem) // Actually, the content of rem is readwrite
    .def_readonly("total_caches", &RMS::total_caches)
    .def_readonly("total_episodes", &RMS::total_episodes)
    .def_readonly("total_steps", &RMS::total_steps)
    .def_readonly("logfile", &RMS::logfile_path)
    .def(py::init([](py::buffer s, py::buffer a, py::buffer r, py::buffer p,
            py::buffer v, py::buffer q, py::buffer i,
            size_t capa, const char * pub_ep, int n_caches) {
          BV view[N_VIEW];
          view[0] = AS_BV(s);
          view[1] = AS_BV(a);
          view[2] = AS_BV(r);
          view[3] = AS_BV(p);
          view[4] = AS_BV(v);
          view[5] = AS_BV(q);
          view[6] = AS_BV(i);
          return std::unique_ptr<RMS>(new RMS(view,capa,&lcg64,pub_ep,n_caches));
        }),
        "state_buf"_a,
        "action_buf"_a,
        "reward_buf"_a,
        "prob_buf"_a,
        "value_buf"_a,
        "qvest_buf"_a,
        "info_buf"_a,
        "max_step"_a,
        "pub_endpoint"_a,
        "n_caches"_a)
    .def("set_logfile", &RMS::set_logfile)
    .def("print_info", [](RMS& rms) { rms.print_info(); })
    .def("rep_worker_main",  &RMS::rep_worker_main,  py::call_guard<py::gil_scoped_release>())
    .def("pull_worker_main", &RMS::pull_worker_main, py::call_guard<py::gil_scoped_release>())
    .def("rep_proxy_main",   &RMS::rep_proxy_main,   py::call_guard<py::gil_scoped_release>())
    .def("pull_proxy_main",  &RMS::pull_proxy_main,  py::call_guard<py::gil_scoped_release>())
    .def("pub_proxy_main",   &RMS::pub_proxy_main,   py::call_guard<py::gil_scoped_release>())
    .def("pub_bytes",        &RMS::pub_bytes,        py::call_guard<py::gil_scoped_release>())
    .def("get_batch", [](RMS& s,
          size_t batch_size) {
        std::vector<BV> ret = s.get_batch_view(batch_size);
        std::vector<py::array> prev;
        std::vector<py::array> next;
        for(int i=0; i<N_VIEW; i++)
          prev.emplace_back(py::dtype(ret[i].format_), ret[i].shape_, ret[i].stride_, nullptr);
        for(int i=N_VIEW; i<2*N_VIEW; i++)
          next.emplace_back(py::dtype(ret[i].format_), ret[i].shape_, ret[i].stride_, nullptr);
        pyarr_float entry_weight_arr({batch_size,});
        void * ptr[2*N_VIEW] = {nullptr};
        for(int i=0; i<N_VIEW; i++)
          ptr[i] = (void*)prev[i].mutable_data();
        for(int i=0; i<N_VIEW; i++)
          ptr[i+N_VIEW] = (void*)next[i].mutable_data();
        float * wptr = entry_weight_arr.mutable_data();
        bool succ = false;
        if(true) {
          py::gil_scoped_release release;
          succ = s.get_batch(batch_size,
            ptr[ 0], ptr[ 1], ptr[ 2], ptr[ 3], ptr[ 4], ptr[ 5], ptr[ 6],
            ptr[ 7], ptr[ 8], ptr[ 9], ptr[10], ptr[11], ptr[12], ptr[13],
            wptr);
        }
        if(not succ)
          throw std::runtime_error("get_batch() failed.");
        return std::make_tuple(prev,next,entry_weight_arr);
      },"batch_size"_a)
    ;
    */

}

