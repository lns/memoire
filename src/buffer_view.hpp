#pragma once

#define USE_PY

#include <cassert>
#include <cstring>
#include <string>
#include <vector>
#include "array_view.hpp"
#include "qlog.hpp"
#include "msg.pb.h"
#ifdef USE_PY
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
using namespace py::literals;
#endif

/**
 * A class for an array of runtime-sized, runtime-typed entries. Does not own the memory.
 * Also see py::buffer_info
 */
class BufView {
public:
  void * ptr_;              ///< head pointer
  ssize_t itemsize_;        ///< byte size of each item
  std::string format_;      ///< item format: 'B/b' u/int8, 'H/h' u/int16, 'I/i' u/int32, 'L/l' u/int64, 'e/f/d' float16/32/64.
  std::vector<ssize_t> shape_;   ///< shape (in items)
  std::vector<ssize_t> stride_;  ///< stride (in bytes)

  // For serialization
  class Data {
    ssize_t itemsize_;
    char format_[4];
    unsigned ndim_;
    ssize_t shape_[8];
    ssize_t stride_[8];
  public:
    void from(const BufView& buf) {
      itemsize_ = buf.itemsize_;
      assert(buf.format_.size() < sizeof(format_));
      strncpy(format_, buf.format_.c_str(), sizeof(format_));
      ndim_ = buf.shape_.size();
      assert(ndim_ < sizeof(shape_)/sizeof(shape_[0]));
      memcpy(shape_, buf.shape_.data(), ndim_*sizeof(shape_[0]));
      memcpy(stride_, buf.stride_.data(), ndim_*sizeof(stride_[0]));
    }

    void to(BufView& buf) {
      buf.itemsize_ = itemsize_;
      buf.format_ = std::string(format_);
      buf.shape_.resize(ndim_);
      memcpy(buf.shape_.data(), shape_, ndim_*sizeof(shape_[0]));
      buf.stride_.resize(ndim_);
      memcpy(buf.stride_.data(), stride_, ndim_*sizeof(stride_[0]));
    }
  };


  BufView() : ptr_{nullptr}, itemsize_{0}, format_{""}, shape_{}, stride_{} {}
  BufView(void * p, ssize_t is, const std::string& f, const std::vector<ssize_t>& sp, const std::vector<ssize_t>& st)
    : ptr_{p}, itemsize_{is}, format_{f}, shape_{sp}, stride_{st} {}
  BufView(void * p, ssize_t is, const std::string& f, const std::vector<ssize_t>& sp)
    : ptr_{p}, itemsize_{is}, format_{f}, shape_{sp}
  {
    make_c_stride();
  }
  BufView(const proto::BufView& v) { from_pb(&v); }
#ifdef USE_PY
  BufView(const py::buffer_info& info, bool keep_ptr)
    : BufView(keep_ptr ? info.ptr : nullptr, info.itemsize, info.format, info.shape, info.strides) {}
  BufView(const py::object& b) : BufView(py::cast<py::buffer>(b).request(), true) {}

  py::array new_array() {
    return py::array(py::dtype(format_), shape_, stride_, nullptr);
  }
#endif

  void make_c_stride() {
    // Fill stride to be c contiguous
    stride_.resize(ndim(), itemsize_);
    ssize_t s = itemsize_;
    for(int i=ndim()-1; i>=0; i--) {
      stride_[i] = s;
      s *= shape_[i];
    }
  }
  
  size_t ndim() const { return shape_.size(); }
  size_t size() const {
    ssize_t ret = 1;
    for(const auto& each: shape_)
      ret *= each;
    return ret;
  }
  size_t nbytes() const {
    if(ndim()==0)
      return itemsize_;
    return shape_[0] * stride_[0];
  }

  bool is_c_contiguous() const {
    if(shape_.size() != stride_.size())
      throw std::runtime_error("ndim does not match");
    ssize_t s = itemsize_;
    for(int i=ndim()-1; i>=0; i--) {
      if(stride_[i] != s)
        return false;
      s *= shape_[i];
    }
    return true;
  }

  bool is_consistent_with(const BufView& b) const {
    if(format_ != b.format_ or itemsize_ != b.itemsize_ or shape_ != b.shape_ or stride_ != b.stride_) {
      qlog_warning("%s() check failed.\n %s != %s\n.", __func__, str().c_str(), b.str().c_str());
      return false;
    }
    return true;
  }

  BufView operator[](size_t idx) const {
    assert(idx ==0 or (ndim()>0 and idx < (size_t)shape_[0]));
    auto new_shape = shape_;
    new_shape.erase(new_shape.begin());
    auto new_stride = stride_;
    new_stride.erase(new_stride.begin());
    return BufView((char*)ptr_ + idx * stride_[0], itemsize_, format_, new_shape, new_stride);
  }

  template<typename T>
  ArrayView<T> as_array() const {
    assert(is_c_contiguous());
    assert(sizeof(T) == itemsize_);
    return ArrayView<T>(ptr_, size());
  }

  /**
   * Copy from a piece of memory
   * @param src   can be nullptr to be omitted
   */
  size_t from_memory(const void * src) {
    if(src) {
      std::memcpy(ptr_, src, nbytes());
      return nbytes();
    }
    return 0;
  }

  /**
   * Copy to a piece of memory
   * @param dst   can be nullptr to be omitted
   */
  size_t to_memory(void * dst) const {
    if(dst) {
      std::memcpy(dst, ptr_, nbytes());
      return nbytes();
    }
    return 0;
  }

  /**
   * Copy from a Protocal Buffer
   */
  void from_pb(const proto::BufView * pb) {
    qassert(pb);
    ptr_ = nullptr;
    itemsize_ = pb->itemsize();
    format_ = pb->format();
    shape_.resize(pb->shape_size());
    for(int i=0; i<pb->shape_size(); i++)
      shape_[i] = pb->shape(i);
    stride_.resize(pb->stride_size());
    for(int i=0; i<pb->stride_size(); i++)
      stride_[i] = pb->stride(i);
    qassert(shape_.size() == stride_.size());
  }

  /**
   * Copy to a Protocal Buffer
   */
  void to_pb(proto::BufView * pb) const {
    qassert(pb);
    pb->set_itemsize(itemsize_);
    pb->set_format(format_);
    pb->clear_shape();
    for(unsigned i=0; i<shape_.size(); i++)
      pb->add_shape(shape_[i]);
    pb->clear_stride();
    for(unsigned i=0; i<stride_.size(); i++)
      pb->add_stride(stride_[i]);
  }

  proto::BufView as_pb() const {
    proto::BufView v;
    to_pb(&v);
    return v;
  }

  std::string shape_str() const {
    std::string ret;
    ret += "(";
    for(const auto& each : shape_)
      ret += std::to_string(each)+",";
    ret += ")";
    return ret;
  }

  std::string stride_str() const {
    std::string ret;
    ret += "(";
    for(const auto& each : stride_)
      ret += std::to_string(each/itemsize_)+",";
    ret += ")";
    return ret;
  }

  std::string dtype_str() const {
    char type = format_.size() > 0 ? format_[0] : '\0';
    switch(type) {
      case 'B': return std::string("uint8");
      case 'b': return std::string("int8");
      case 'H': return std::string("uint16");
      case 'h': return std::string("int16");
      case 'I': return std::string("uint32");
      case 'i': return std::string("int32");
      case 'L': return std::string("uint64");
      case 'l': return std::string("int64");
      case 'f': return std::string("float32");
      case 'd': return std::string("float64");
      default: return std::string("(bool?)");
    }
    return std::string("unknwon");
  }

  /**
   * Represent info in a string
   */
  std::string str() const {
    char buf[256];
    snprintf(buf, sizeof(buf),
        "<BufView ptr:%p, itemsize:%ld, format:'%s', dtype:'%s',",
        ptr_, itemsize_, format_.c_str(), dtype_str().c_str());
    std::string ret = std::string(buf);
    ret += " shape: " + shape_str() + ",";
    ret += " stride: " + stride_str() + ">";
    return ret;
  }

  /**
   * Print content
   * @param f   output file
   */
  void print(FILE * f = stdout) const {
    assert(ptr_);
    if(ndim() > 0) {
      fprintf(f, "[");
      for(int i=0; i<shape_[0]; i++)
        (*this)[i].print();
      fprintf(f, "],\n");
    } else {
      char type = format_.size() > 0 ? format_[0] : '\0';
      switch(type) {
        case 'B': fprintf(f,"%hhu,", *( uint8_t*)ptr_); break;
        case 'b': fprintf(f,"%hhd,", *(  int8_t*)ptr_); break;
        case 'H': fprintf(f, "%hu,", *(uint16_t*)ptr_); break;
        case 'h': fprintf(f, "%hd,", *( int16_t*)ptr_); break;
        case 'I': fprintf(f,  "%u,", *(uint32_t*)ptr_); break;
        case 'i': fprintf(f,  "%d,", *( int32_t*)ptr_); break;
        case 'L': fprintf(f, "%lu,", *(uint64_t*)ptr_); break;
        case 'l': fprintf(f, "%ld,", *( int64_t*)ptr_); break;
        case 'f': fprintf(f,  "%e,", *(   float*)ptr_); break;
        case 'd': fprintf(f, "%le,", *(  double*)ptr_); break;
        default: fprintf(f, "?%c,", type);
      }
    }
  }

};

