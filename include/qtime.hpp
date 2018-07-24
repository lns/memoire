#pragma once

#include <chrono>
#include <cstdint>
#include <ctime>
#include <string>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <atomic>
#include <algorithm>

namespace qlib {

// For new application, use std::this_thread::sleep_for(2us) instead

/**
 * Sleep for some time
 * @param sleep time in seconds
 */
inline void sleep(double sec) {
	struct timespec t;
	t.tv_sec = sec;
	t.tv_nsec = sec*1000000000;
	t.tv_nsec %= 1000000000;
	nanosleep(&t,nullptr);
}

// See http://www.intel.com/content/www/us/en/embedded/training/ia-32-ia-64-benchmark-code-execution-paper.html
// todo: do we need memory barrier?

#include <x86intrin.h>
inline uint64_t _start() {
	//unsigned aux;
	uint64_t ret;
	std::atomic_thread_fence(std::memory_order_seq_cst);
	//ret = __rdtscp(&aux);
	ret = __rdtsc();
	//std::atomic_thread_fence(std::memory_order_seq_cst);
	return ret;
}

inline uint64_t _stop() {
	unsigned aux;
	uint64_t ret;
	ret = __rdtscp(&aux);
	std::atomic_thread_fence(std::memory_order_seq_cst);
	return ret;
}

/**
 * CPU cycles measurer based on RDTSCP
 */
class CycleTimer {
	uint64_t bgn, end;
	size_t n;
	double min_cyc, sum_cyc, max_cyc;
public:
	CycleTimer(): n(0), min_cyc(std::numeric_limits<double>::max()), sum_cyc(0), max_cyc(0) {
		_start(); // cold start
		_stop();
	}

	inline void start() {
		bgn = _start();
	}

	// Return elapsed CPU cycles
	inline uint64_t stop() {
		end = _stop();
		uint64_t diff = end-bgn; // adjusted
		// Update
		n++;
		min_cyc = std::min<double>(min_cyc, diff);
		max_cyc = std::max<double>(max_cyc, diff);
		sum_cyc += diff;
		return diff;
	}

	inline void clear() {
    n = 0;
    min_cyc = std::numeric_limits<double>::max();
    sum_cyc = 0;
    max_cyc = 0;
  }
	inline double min() const { return min_cyc; }
	inline double avg() const { return sum_cyc/n; }
	inline double max() const { return max_cyc; }	
	inline size_t cnt() const { return n; }
};

/**
 * A CPU timer to measure performance
 * 
 * This is more accurate than CPUTimer in cpu_timer.hpp
 */
class Timer {
	std::chrono::steady_clock::time_point bgn, end;
	size_t n;
	double min_elapsed, sum_elapsed, max_elapsed;
public:
	Timer(): n(0), min_elapsed(std::numeric_limits<double>::max()), sum_elapsed(0), max_elapsed(0) {
		std::chrono::steady_clock::now(); // cold start
	}

	inline void start() {
		std::atomic_thread_fence(std::memory_order_seq_cst);
		bgn = std::chrono::steady_clock::now();
	}

	// Return elapsed time in millisecond (1/1000 second)
	inline double stop() {
		end = std::chrono::steady_clock::now();
		std::atomic_thread_fence(std::memory_order_seq_cst);
		double msec = std::chrono::duration<double, std::milli>(end-bgn).count();
		// Update
		n++;
		min_elapsed = std::min<double>(min_elapsed, msec);
		max_elapsed = std::max<double>(max_elapsed, msec);
		sum_elapsed += msec;
		return msec;
	}

	inline void clear() {
    n = 0;
    min_elapsed = std::numeric_limits<double>::max();
    sum_elapsed = 0;
    max_elapsed = 0;
  }
	inline double min() const { return min_elapsed; }
	inline double avg() const { return sum_elapsed/n; }
	inline double max() const { return max_elapsed; }	
	inline size_t cnt() const { return n; }
};

/**
 * Get fraction of time_point
 */
template<typename Clock, typename Unit>
inline unsigned long get_fraction(std::chrono::time_point<Clock> tp) {
	std::chrono::time_point<Clock> t = 
		std::chrono::time_point_cast<std::chrono::seconds>(tp);
	return std::chrono::duration_cast<Unit>(tp - t).count();
}

/**
 * Get milli-seconds
 */
template<typename Clock>
inline unsigned long get_msec(std::chrono::time_point<Clock> tp) {
	return get_fraction<Clock,std::chrono::milliseconds>(tp);
}

/**
 * Get micro-seconds
 */
template<typename Clock>
inline unsigned long get_usec(std::chrono::time_point<Clock> tp) {
	return get_fraction<Clock,std::chrono::microseconds>(tp);
}

/**
 * Get nano-seconds
 */
template<typename Clock>
inline unsigned long get_nsec(std::chrono::time_point<Clock> tp) {
	return get_fraction<Clock,std::chrono::nanoseconds>(tp);
}

/**
 * Get a nano-sec as time-dependent random seed
 */
inline unsigned long get_nsec() {
	return get_nsec(std::chrono::high_resolution_clock::now());
}

#if USE_DEPRECATED
/**
 * Get current time in a string format.
 * NOTE: Use a static char buffer, not safe for multiple calls at a time.
 */
inline const char* strtime(bool local=true) {
	std::time_t raw_time;
	std::tm * timeinfo;
	std::chrono::time_point<std::chrono::system_clock> time_point;
	time_point = std::chrono::system_clock::now();
	raw_time = std::chrono::system_clock::to_time_t(time_point);
	if(local)
		timeinfo = std::localtime(&raw_time);
	else
		timeinfo = std::gmtime(&raw_time); // UTC
	unsigned long msec = get_msec(time_point);
	static char buf[32];
	strftime(buf, 20, "%F %X", timeinfo);
	snprintf(buf+19, 12, ".%03lu", msec);
	return buf;
}
#endif

/**
 * Get current time in a std::string
 */
inline std::string timestr(bool local=true) {
	std::time_t raw_time;
	std::tm * timeinfo;
	std::chrono::time_point<std::chrono::system_clock> time_point;
	time_point = std::chrono::system_clock::now();
	raw_time = std::chrono::system_clock::to_time_t(time_point);
	if(local)
		timeinfo = std::localtime(&raw_time);
	else
		timeinfo = std::gmtime(&raw_time); // UTC
	unsigned long msec = get_msec(time_point);
	char buf[32];
	strftime(buf, 20, "%F %X", timeinfo);
	snprintf(buf+19, 12, ".%03lu", msec);
	return std::string(buf);
}

#if USE_DEPRECATED
/**
 * Convert std::time_t to std::string
 */
std::string ctime2str(const std::time_t& raw_time) {
	char buf[32];
	std::tm * timeinfo;
	timeinfo = std::localtime(&raw_time);
	strftime(buf, 20, "%F %X", timeinfo);
	return std::string(buf);
}

/**
 * Convert c_str() to std::time_t
 */
std::time_t str2ctime(const char * str) {
	std::time_t raw_time;
	std::tm tminfo;
	strptime(str, "%F %X", &tminfo);
	tminfo.tm_isdst = 0; // NOTE: Not specified by strptime, should be filled manually.
	raw_time = std::mktime(&tminfo);
	return raw_time;
}
#endif

/**
 * Convert std::string to time_point
 * (This depends on strptime() which is not in std c++)
 */
std::chrono::time_point<std::chrono::system_clock>
str2timepoint(const char * str, const char * format) {
	typedef std::chrono::duration<unsigned long, std::micro> Duration;
	std::time_t raw_time;
	std::tm tminfo;
	std::chrono::time_point<std::chrono::system_clock> time_point;
	strptime(str, format, &tminfo); //strptime(str, "%F %X", &tminfo);
	tminfo.tm_isdst = 0; // NOTE: Not specified by strptime, should be filled manually.
	raw_time = std::mktime(&tminfo); // local calendar time
	time_point = std::chrono::system_clock::from_time_t(raw_time);
	// fraction part: in micro sec
	const char * frac = strchr(str,'.');
	if(frac) {
		Duration us(std::llround(1e6*strtod(frac,NULL)));
		time_point += us;
	}
	return time_point;
}

/**
 * Convert time_point to std::string
 */
std::string timepoint2str(std::chrono::time_point<std::chrono::system_clock> tp,
		const char * format) {
	char b[256];
	std::time_t raw_time = std::chrono::system_clock::to_time_t(tp);
	std::tm * timeinfo = std::localtime(&raw_time); // local
	strftime(b,256,format,timeinfo);
	char * end = strchr(b,'\0');
	snprintf(end,b+256-end,".%06lu",qlib::get_usec(tp));
	return std::string(b);
}

/**
 * Convert usecs since epoch to time_point
 */
std::chrono::time_point<std::chrono::system_clock>
usecs2timepoint(unsigned long usecs) {
	typedef std::chrono::system_clock Clock;
	typedef std::chrono::time_point<Clock> TimePoint;
	return TimePoint(Clock::duration(std::chrono::microseconds(usecs)));
}

/**
 * Convert usecs to str
 */
std::string usecs2str(unsigned long usecs, const char * format) {
	return timepoint2str(usecs2timepoint(usecs),format);
}

/**
 * Convert time_point to usecs since epoch
 */
unsigned long timepoint2usecs(std::chrono::time_point<std::chrono::system_clock> tp) {
	return std::chrono::duration_cast<std::chrono::microseconds>
		(tp.time_since_epoch()).count();
}
/**
 * Convert str to usecs
 */
unsigned long str2usecs(const char * str, const char * format) {
	return timepoint2usecs(str2timepoint(str,format));
}

/**
 * Now in micro seconds since epoch
 */
inline unsigned long now() {
	return timepoint2usecs(std::chrono::system_clock::now());
}

} // namespace qlib

