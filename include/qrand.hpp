#pragma once

#include <cstdint>
#include <cstdlib>

namespace qlib {

class RNG {
public:
  virtual void seed(uint64_t s) =0;
  virtual uint64_t sample() =0;
  virtual double drand() =0;
};

class XOR64STAR : public RNG {
	uint64_t s_;
public:
	explicit XOR64STAR(uint64_t s = 0xDEADBEEF) { seed(s); }
  XOR64STAR(XOR64STAR&& src) : s_{src.s_} {}
  XOR64STAR(const XOR64STAR& src) : s_{src.s_} {}

	void seed(uint64_t s) override { s_ = s; }

	uint64_t sample() override {
		s_ ^= s_ >> 12; // a
		s_ ^= s_ << 25; // b
		s_ ^= s_ >> 27; // c
		return s_ * 2685821657736338717ull;
	}

	double drand() override {
		return static_cast<double>(sample())/(~0ull);
	}
};

class XOR128PLUS : public RNG {
	uint64_t s_[2];
public:
	explicit XOR128PLUS(uint64_t s0 = 0xDEADBEEF) { seed(s0); }
  XOR128PLUS(XOR128PLUS&& src) : s_{src.s_[0], src.s_[1]} {}
  XOR128PLUS(const XOR128PLUS& src) : s_{src.s_[0], src.s_[1]} {}

	void seed(uint64_t seed0) override {
		s_[0] = seed0;
		s_[1] = 0xCAFEBABE;
	}

	uint64_t sample() override {
		uint64_t x = s_[0];
		uint64_t const y = s_[1];
		s_[0] = y;
		x ^= x << 23; // a
		s_[1] = x ^ y ^ (x >> 17) ^ (y >> 26); // b, c
		return s_[1] + y;
	}

	double drand() override {
		return static_cast<double>(sample())/(~0ull);
	}

};

class LCG64 : public RNG {
	uint64_t s_;
public:
	explicit LCG64(uint64_t s = 0xDEADBEEF) { seed(s); }
  LCG64(LCG64&& src) : s_{src.s_} {}
  LCG64(const LCG64& src) : s_{src.s_} {}

	inline void seed(uint64_t s) override { s_ = s; }

	inline uint64_t sample() override {
		s_ = 18145460002477866997ull*s_ + 1;
		return s_;
	}

	inline double drand() override {
		return static_cast<double>(sample())/(~0ull);
	}
};

template<class T>
void shuffle(RNG* rng, T* bgn, const T* end) {
	for(auto it = bgn; it!=end; it++)
		std::swap(it[0], it[rng->sample() % (end-it)]);
}

}; // namespace qlib

