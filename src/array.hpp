#pragma once

#include "mem.hpp"
#include <algorithm>  // std::max

/**
 * A class for a dynamic array
 */
template<typename T>
class Array {
  Mem mem_;                  ///< underlying memory
  const size_t entry_size_;  ///< size of each entry (in bytes)
  size_t entry_num_;         ///< number of entries
public:
  
  void * data() const { return mem_.data(); };
  size_t size() const { return entry_num_; };
  size_t capacity() const { return mem_.size() / entry_size_; }

  Array(size_t entry_size) : mem_(), entry_size_{entry_size}, entry_num_{0} {}

  T& operator[](size_t idx) const {
    return *(T*)((char*)data() + entry_size_ * idx);
  }

  void reserve(size_t num) {
    if(mem_.size() < entry_size_ * num)
      mem_.resize(entry_size_ * num);
  }

  void memcpy_back(const void * src) {
    if(mem_.size() < entry_size_ * (entry_num_+1))
      reserve(std::max(entry_num_*2, entry_num_+1)); // adjustable
    void * dst = (char*)(data()) + entry_size_ * entry_num_;
    std::memcpy(dst, src, entry_size_);
    entry_num_ ++;
  }

  void clear() {
    entry_num_ = 0;
  }

};

