#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "buffer_view.hpp"

namespace py = pybind11;
using namespace py::literals;

typedef py::array_t<float, py::array::c_style> pyarr_float;
typedef py::array_t<uint8_t, py::array::c_style> pyarr_uint8;

BufView AS_BV(py::object b) {
  py::buffer_info info = py::cast<py::buffer>(b).request();
  return BufView(info.ptr, info.itemsize, info.format, info.shape, info.strides);
}

/**
 * Equivalent to `pickle.dumps(o)`
 */
py::bytes pickle_dumps(py::object o) {
  py::module pickle = py::module::import("pickle");
  py::object ret = pickle.attr("dumps")(o);
  return py::cast<py::bytes>(ret);
}

/**
 * Equivalent to `pickle.loads(s)`
 */
py::object pickle_loads(py::bytes s) {
  py::module pickle = py::module::import("pickle");
  py::object ret = pickle.attr("loads")(s);
  return ret;
}

/**
 * Get the structure description of object s
 */
py::object get_descr(py::object s) {
  if(py::isinstance<py::buffer>(s)) {
    py::buffer_info info = py::cast<py::buffer>(s).request();
    py::dict ret = py::dict("itemsize"_a=info.itemsize, "format"_a=info.format,
        "shape"_a=info.shape, "strides"_a=info.strides);
    return ret;
  }
  else if(py::isinstance<py::tuple>(s)) {
    py::tuple l = py::cast<py::tuple>(s);
    py::list ret;
    for(unsigned i=0; i<l.size(); i++)
      ret.append(get_descr(l[i]));
    return ret;
  }
  else if(py::isinstance<py::list>(s)) {
    py::list l = py::cast<py::list>(s);
    py::list ret;
    for(unsigned i=0; i<l.size(); i++)
      ret.append(get_descr(l[i]));
    return ret;
  }
  else {
    throw std::runtime_error("Invalid object.\n");
  }
}

/**
 * Get the structure's nbytes from description
 */
size_t get_descr_nbytes(py::object descr) {
  if(py::isinstance<py::dict>(descr)) {
    py::dict d = py::cast<py::dict>(descr);
    BufView v = BufView(nullptr,
        py::cast<ssize_t>(d["itemsize"]),
        py::cast<std::string>(d["format"]),
        py::cast<std::vector<ssize_t>>(d["shape"]),
        py::cast<std::vector<ssize_t>>(d["strides"]));
    return v.nbytes();
  }
  else if(py::isinstance<py::list>(descr)) {
    py::list l = py::cast<py::list>(descr);
    size_t ret = 0;
    for(unsigned i=0; i<l.size(); i++)
      ret += get_descr_nbytes(l[i]);
    return ret;
  }
  else {
    throw std::runtime_error("Invalid descr.\n");
  }
}

/**
 * Serialize s to the memory string at head, according to descr
 * return size of serialization
 */
size_t descr_serialize_to_mem(py::object s, py::object descr, void* head) {
  if(py::isinstance<py::dict>(descr)) {
    py::dict d = py::cast<py::dict>(descr);
    BufView v = BufView(nullptr,
        py::cast<ssize_t>(d["itemsize"]),
        py::cast<std::string>(d["format"]),
        py::cast<std::vector<ssize_t>>(d["shape"]),
        py::cast<std::vector<ssize_t>>(d["strides"]));
    py::buffer b = py::cast<py::buffer>(s);
    assert(v.is_consistent_with(AS_BV(b))); // descr mismatch
    memcpy(head, AS_BV(b).ptr_, v.nbytes());
    return v.nbytes();
  }
  else if(py::isinstance<py::list>(descr)) {
    py::list l = py::cast<py::list>(descr);
    size_t ret = 0;
    if(py::isinstance<py::tuple>(s)) {
      py::tuple ts = py::cast<py::tuple>(s);
      for(unsigned i=0; i<l.size(); i++) {
        ret += descr_serialize_to_mem(ts[i], l[i], (char*)head+ret);
      }
    }
    else if(py::isinstance<py::list>(s)) {
      py::list ls = py::cast<py::list>(s);
      for(unsigned i=0; i<l.size(); i++) {
        ret += descr_serialize_to_mem(ls[i], l[i], (char*)head+ret);
      }
    }
    else
      throw std::runtime_error("Python object does not match the descriptor.\n");
    return ret;
  }
  else {
    throw std::runtime_error("Invalid descr.\n");
  }
}

pyarr_uint8 descr_serialize(py::object s, py::object descr) {
  pyarr_uint8 ret({get_descr_nbytes(descr),});
  size_t size = descr_serialize_to_mem(s, descr, ret.mutable_data());
  assert(size == AS_BV(ret).nbytes());
  return ret;
}

py::object descr_unserialize_from_mem(py::object descr, void* head, size_t& offset) {
  if(py::isinstance<py::dict>(descr)) {
    py::dict d = py::cast<py::dict>(descr);
    BufView v = BufView(nullptr,
        py::cast<ssize_t>(d["itemsize"]),
        py::cast<std::string>(d["format"]),
        py::cast<std::vector<ssize_t>>(d["shape"]),
        py::cast<std::vector<ssize_t>>(d["strides"]));
    py::array ret(py::dtype(v.format_), v.shape_, v.stride_, nullptr);
    memcpy(ret.mutable_data(), (char*)head + offset, v.nbytes());
    offset += v.nbytes();
    return ret;
  }
  else if(py::isinstance<py::list>(descr)) {
    py::list l = py::cast<py::list>(descr);
    py::list ret;
    for(unsigned i=0; i<l.size(); i++) {
      ret.append(descr_unserialize_from_mem(l[i], head, offset));
    }
    return py::tuple(ret);
  }
  else {
    throw std::runtime_error("Invalid descr.\n");
  }
}

py::object descr_unserialize_from_mem(py::object descr, void* head) {
  size_t offset = 0;
  return descr_unserialize_from_mem(descr, head, offset);
}

py::object descr_unserialize(pyarr_uint8 b, py::object descr) {
  size_t offset = 0;
  return descr_unserialize_from_mem(descr, AS_BV(b).ptr_, offset);
}

