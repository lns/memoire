/**
 * @file: prt_tree.hpp
 * @brief: Priority Tree (CPU)
 * @author: Qing
 * @date: 2018-02-27
 */
#pragma once

#include <vector>
#include <cassert>
#include "qrand.hpp"

static int base2ceil(int x) {
  int a = 1;
  while(x>a) {
    a <<= 1;
    assert(a>0);
  }
  return a;
}

class PrtTree {
public:
  /**
   * Construct a priority tree of capacity
   *
   * O(N)
   */
  PrtTree(qlib::RNG* rng, int capacity) {
    assert(rng);
    assert(capacity >= 0);
    rng_ = rng;
    size_ = base2ceil(std::max<int>(capacity, 1));
    w_.resize(2 * size_, 0.0);
  }
  PrtTree(PrtTree&& src) : rng_{src.rng_}, size_{src.size_}, w_{src.w_} {}
  PrtTree(const PrtTree& src) : rng_{src.rng_}, size_{src.size_}, w_{src.w_} {}

  /**
   * Destructor
   */
  virtual ~PrtTree() {}

  /**
   * Get the weight for index i
   *
   * O(1)
   */
  float get_weight(int i) const {
    assert(0 <= i and i < size_);
    return w_[size_ + i];
  }

  /**
   * Get the sum of weight
   *
   * O(1)
   */
  float get_weight_sum() const {
    return w_[1];
  }

  /**
   * Get index of minimum weight
   *
   * O(logN)
   */
  int get_argmin() const {
    int s=1;
    while(s < size_)
      s = w_[2*s] > w_[2*s+1] ? (2*s+1) : (2*s);
    assert(0 <= s-size_ and s-size_ < size_);
    return s-size_;
  }

  /**
   * Set the weight for index i, and update the tree structure.
   * Not thread-safe.
   *
   * O(logN)
   */
  void set_weight(int i, float weight) {
    assert(0 <= i and i < size_);
    w_[size_ + i] = weight;
    int s = size_;
    while(s > 1) {
      s >>= 1;
      i >>= 1;
      w_[s+i] = w_[2*s+2*i] + w_[2*s+2*i+1];
    }
  }

  /**
   * Set the weight for index i, without updating the tree structure.
   *
   * O(1)
   */
  void set_weight_without_update(int i, float weight) {
    assert(0 <= i and i < size_);
    w_[size_ + i] = weight;
  }

  /**
   * Update the tree structure
   *
   * O(N)
   */
  void update_all() {
    for(int i=size_-1; i>0; i--)
      w_[i] = w_[2*i] + w_[2*i+1];
  }

  /**
   * Sample according to the weight 
   *
   * O(logN)
   */
  int sample_index() {
    int s=1;
    while(s < size_) {
      //assert(0.0 <= w_[2*s]);
      //assert(w_[2*s] <= w_[s]);
      //assert(0.0 <= w_[2*s+1]);
      //assert(w_[2*s+1] <= w_[s]);
      double r;
      do {
        r = rng_->drand();
      } while(r <= 0.0 or r >= 1.0);
      r *= w_[s];
      //assert(0.0 <= r);
      //assert(r <= w_[s]);
#if 0
      if(w_[2*s] > 0.0 and w_[2*s+1] < 1.0 and r > w_[2*s]) { // debug
        qlog_warning("w[%d]:%f = w[%d]:%f + w[%d]:%f, r: %le\n", s, w_[s], 2*s, w_[2*s], 2*s+1, w_[2*s+1], r);
      }
#endif
      s = r > w_[2*s] ? (2*s+1) : (2*s);
      
    }
    assert(0 <= s-size_ and s-size_ < size_);
    return s-size_; 
  }

  /**
   * Sample reversely (smaller weight, larger probability)
   *
   * O(logN)
   */
  int sample_reversely() {
    int s=1;
    while(s < size_) {
      double r = w_[s] * rng_->drand();
      s = r > w_[2*s+1] ? (2*s+1) : (2*s);
    }
    assert(0 <= s-size_ and s-size_ < size_);
    return s-size_; 
  }

  /**
   * Clear all weight
   */
  void clear() {
    memset(w_.data(), 0, 2 * size_ * sizeof(float));
  }

  /**
   * Print weight along the selection path for index i.
   */
  void debug_print(int i) const {
    assert(0 <= i and i < size_);
    int s = i+size_;
    while(s > 1) {
      fprintf(stderr, "selected %d from w[%d]: %f, w[%d]: %f\n",
          s, s/2*2, w_[s/2*2], s/2*2+1, w_[s/2*2+1]);
      s = s/2;
    }
  }

public:
  // RNG
  qlib::RNG* rng_;
  // max index
  int size_;
  // the structure of w_ is [0.0, 1.0, 0.5, 0.5, 0.25, 0.25 ... ]
  std::vector<float> w_;
};

