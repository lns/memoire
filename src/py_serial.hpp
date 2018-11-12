#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
using namespace py::literals;

#include "buffer_view.hpp"
#include "msg.pb.h"

typedef py::array_t<float, py::array::c_style> pyarr_float;
typedef py::array_t<uint8_t, py::array::c_style> pyarr_uint8;

/**
 * Generate a proto::Descriptor for a tuple of numpy array
 */
proto::Descriptor get_desc(py::list l) {
  proto::Descriptor desc;
  for(unsigned i=0; i<l.size(); i++) {
    if(not py::isinstance<py::buffer>(l[i]))
      throw std::runtime_error("Object in get_desc() is not a py::buffer\n");
    py::buffer_info info = py::cast<py::buffer>(l[i]).request();
    BufView v(info, false);
    *desc.add_view() = v.as_pb();
  }
  return desc;
}

/**
 * Calculate the total number of bytes from proto::Descriptor
 */
size_t get_desc_nbytes(const proto::Descriptor& desc) {
  size_t ret = 0;
  for(int i=0; i<desc.view_size(); i++)
    ret += BufView(desc.view(i)).nbytes();
  return ret;
}

/**
 * Serialize an bundle to memory, with check according to desc
 * @return  number of bytes written to memory
 */
template<bool do_check = true>
size_t serialize_to_mem(py::list bundle, void * dst, const proto::Descriptor& desc) {
  size_t ret = 0;
  char * head = static_cast<char*>(dst);
  if(do_check and static_cast<int>(bundle.size()) != desc.view_size())
    throw std::runtime_error("len(bundle) mismatch.\n");
  for(unsigned i=0; i<bundle.size(); i++) {
    if(not py::isinstance<py::buffer>(bundle[i]))
      throw std::runtime_error("bundle content is not a py::buffer\n");
    BufView v(py::cast<py::buffer>(bundle[i]));
    if(do_check and not v.is_consistent_with(desc.view(i)))
      throw std::runtime_error("bundle content mismatch\n");
    ret += v.to_memory(head + ret);
  }
  return ret;
}

