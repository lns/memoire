#pragma once

#include "mem.hpp"
#include <algorithm>  // std::max

/**
 * A class for a vector of runtime-sized entries
 * @param T   every chunk of memory will be interpreted as an instance of T
 */
template<typename T>
class Vector {
  Mem mem_;                  ///< underlying memory
  size_t entry_num_;         ///< number of entries
public:
  const size_t entry_size;   ///< size of each entry (in bytes)
  
  void * data() const { return mem_.data(); };
  size_t size() const { return entry_num_; };
  size_t capacity() const { return mem_.size() / entry_size; }

  /**
   * Constructor
   * @param e_size  actual memory size of each entry
   */
  Vector(size_t e_size) : mem_(), entry_num_{0}, entry_size{e_size} {}

  /**
   * Access the ith entry
   * @param idx  index of entry (should be less than size())
   */
  T& operator[](size_t idx) const {
    char * head = reinterpret_cast<char*>(data());
    size_t offset = entry_size * idx;
    return *reinterpret_cast<T*>(head + offset);
  }

  /**
   * Reserve the memory size for num entries
   * @param num  number of entries
   */
  void reserve(size_t num) {
    if(mem_.size() < entry_size * num)
      mem_.resize(entry_size * num);
  }

  /**
   * Memcpy a piece of memory to the back of this vector
   * @param src  source pointer
   */
  void memcpy_back(const T * src) {
    if(mem_.size() < entry_size * (entry_num_+1))
      reserve(std::max(entry_num_*2, entry_num_+1)); // adjustable
    void * dst = reinterpret_cast<char*>(data()) + entry_size * entry_num_;
    std::memcpy(dst, src, entry_size);
    entry_num_ ++;
  }

  /**
   * Clear the data (but not free the memory)
   */
  void clear() {
    entry_num_ = 0;
  }

};

