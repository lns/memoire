#pragma once

#include <vector>
#include <mutex>
#include <thread>
#include <condition_variable>
#include "vector.hpp"

template<class K, typename T>
struct BoundedVector {
  std::vector<K> vec;
  Vector<T> buf;
  const int len;

  int front;
  int rear;
  int count;

  std::mutex lock;

  std::condition_variable not_full;
  std::condition_variable not_empty;

  BoundedVector(unsigned entry_size, int capacity) : buf{entry_size}, len{capacity}, front{0}, rear{0}, count{0} {
    vec.resize(len);
    buf.resize(len);
  }

  ~BoundedVector() {}

  void put(const K& k, const T * src) {
    std::unique_lock<std::mutex> l(lock);
    // wait till not full
    not_full.wait(l, [this](){return count != len; });
    // l is locked, put the data
    vec[rear] = k;
    memcpy(&buf[rear], src, buf.entry_size);
    rear = (rear + 1) % len;
    ++count;
    // unlock and notify
    l.unlock();
    not_empty.notify_one();
  }

  void get(K& k, T * dst) {
    std::unique_lock<std::mutex> l(lock);
    // wait till not empty
    not_empty.wait(l, [this](){return count != 0; });
    // l is locked, get the data
    k = vec[front];
    memcpy(dst, &buf[front], buf.entry_size);
    front = (front + 1) % len;
    --count;
    // unlock and notify
    l.unlock();
    not_full.notify_one();
  }

};

