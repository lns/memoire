#pragma once

#include <cstdlib>
#include <cstring>    // memcpy
#include <algorithm>  // swap
#include <cassert>

/**
 * A memory class with ownership
 */
class Mem {
  void * ptr_;        ///< data pointer
  size_t len_;        ///< length in bytes
public:
  void * data() const { return ptr_; }
  size_t size() const { return len_; }

  Mem() : ptr_{nullptr}, len_{0} {}

  Mem(size_t len) : Mem() { resize(len); }

  ~Mem() {
    if(ptr_)
      free(ptr_);
  }

  void resize(size_t len) {
    ptr_ = realloc(ptr_, len);
    len_ = len;
    assert(len==0 or ptr_!=nullptr);
  }

  // copy
  Mem(const Mem& m) : Mem{m.len_} {
    std::memcpy(ptr_, m.ptr_, len_);
  }

  // move
  Mem(Mem&& m) : ptr_{m.ptr_}, len_{m.len_} {
    m.ptr_ = nullptr;
    m.len_ = 0;
  }

  // copy assign
  Mem& operator=(const Mem& m) {
    resize(m.len_);
    std::memcpy(ptr_, m.ptr_, len_);
    return *this;
  }

  // move assign
  Mem& operator=(Mem&& m) {
    if(ptr_)
      free(ptr_);
    ptr_ = m.ptr_;
    len_ = m.len_;
    m.ptr_ = nullptr;
    m.len_ = 0;
    return *this;
  }

  // swap
  friend void swap(Mem& l, Mem& r) {
    std::swap(l.ptr_, r.ptr_);
    std::swap(l.len_, r.len_);
  }

};

