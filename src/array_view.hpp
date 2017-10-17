#pragma once

#include <cassert>
#include <cstring>

/**
 * A class for an array of runtime-sized entries. Does not own the memory.
 * @param T   every chunk of memory will be interpreted as an instance of T
 */
template<typename T>
class ArrayView {
  void * data_;              ///< head pointer
  size_t entry_num_;         ///< number of entries
public:
  const size_t entry_size;
  
  T * data() const { return reinterpret_cast<T*>(data_); }
  size_t size() const { return entry_num_; };

  /**
   * Constructor
   * @param e_size  actual memory size of each entry
   */
  ArrayView(void * data, size_t size, size_t e_size) : data_{data}, entry_num_{size}, entry_size{e_size} {}
  ArrayView(void * data, size_t size) : data_{data}, entry_num_{size}, entry_size{sizeof(T)} {}

  /**
   * Access the ith entry
   * @param idx  index of entry (should be less than size())
   */
  T& operator[](size_t idx) const {
    assert(idx < entry_num_);
    char * head = reinterpret_cast<char*>(data());
    size_t offset = entry_size * idx;
    return *reinterpret_cast<T*>(head + offset);
  }

  /**
   * Copy from a piece of memory
   * @param src   can be nullptr to be omitted
   */
  void from_memory(const T* src) {
    if(src)
      std::memcpy(data_, src, entry_num_ * entry_size);
  }

  /**
   * Copy to a piece of memory
   * @param dst   can be nullptr to be omitted
   */
  void to_memory(T* dst) const {
    if(dst)
      std::memcpy(dst, data_, entry_num_ * entry_size);
  }

};

