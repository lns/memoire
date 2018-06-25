/**
 * @file qlog.hpp
 * @brief Thread-safe (and colorful!) functions to generate log.
 * @author Qing Wang
 * @date 2013-10-03, 2015-04-04, 2016-04-15
 */
#pragma once

#ifndef _QLOG_HPP_
#define _QLOG_HPP_

#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <mutex>
#include "qtime.hpp"

#define likely(x) __builtin_expect(!!(x), 1) 
#define unlikely(x) __builtin_expect(!!(x), 0)

static std::mutex _stderr_mutex;
static bool _colorful = true;
static bool _print_time = false;

inline bool qlog_get_colorful() { return _colorful; }
inline void qlog_set_colorful(bool b) { _colorful = b; }
inline bool qlog_get_print_time() { return _print_time; }
inline void qlog_set_print_time(bool b) { _print_time = b; }

#define LOCK() std::lock_guard<std::mutex> guard(_stderr_mutex)
#define UNLOCK()
//#define LOCK()
//#define UNLOCK()

// Foreground color
// [31m red
// [32m green
// [33m yellow
// [34m blue
// [35m magenta
// [36m cyan

#define CS_ERROR "\033[31m[ERROR]\033[m "
#define CS_WARNING "\033[33m[WARN]\033[m "
#define CS_DEBUG "\033[36m[DEBUG]\033[m "
#define CS_INFO "\033[32m[INFO]\033[m "
#define CS_ASSERT "\033[35m[ASSERT]\033[m "
#define CS_THROW "\033[34m[THROW]\033[m "

#define S_ERROR "[ERROR] "
#define S_WARNING "[WARN] "
#define S_DEBUG "[DEBUG] "
#define S_INFO "[INFO] "
#define S_ASSERT "[ASSERT] "
#define S_THROW "[THROW] "

/**
 * Print Pretty Backtrace
 */
#ifdef __linux__
#include "backward.hpp"
#define print_bt() do { \
	backward::StackTrace st; st.load_here(32); \
	backward::Printer p; p.address = true;\
	p.print(st,stderr); \
} while (0)
#else
#define print_bt()
#endif

/**
 * Print where it goes wrong and throw;
 */
#define qlog_error(...) \
	do { LOCK(); \
		fprintf(stderr,"In %s(), %s:%d\n",__func__, __FILE__, __LINE__); \
		if(errno) fprintf(stderr,"[ERRNO:%02d] '%s'\n",errno,strerror(errno)); \
		errno = 0; \
		fprintf(stderr, qlog_get_colorful()? CS_ERROR : S_ERROR); \
		if(qlog_get_print_time()) fprintf(stderr, "[%s] ", qlib::timestr().c_str()); \
		fprintf(stderr, __VA_ARGS__); \
		UNLOCK(); \
		throw std::runtime_error("Critical error."); \
	}while(0)

/**
 * Print where it goes wrong and continue.
 */
#define qlog_warning(...) \
	do { LOCK(); \
		fprintf(stderr,"In %s(), %s:%d\n",__func__, __FILE__, __LINE__); \
		fprintf(stderr, qlog_get_colorful()? CS_WARNING : S_WARNING); \
		if(qlog_get_print_time()) fprintf(stderr, "[%s] ", qlib::timestr().c_str()); \
		fprintf(stderr, __VA_ARGS__); \
		UNLOCK(); \
	}while(0)

/**
 * Print where it goes wrong and continue.
 */
#ifdef PRINT_DEBUG
#define qlog_debug(...) \
	do { LOCK(); \
		fprintf(stderr,"In %s(), %s:%d\n",__func__, __FILE__, __LINE__); \
		fprintf(stderr, qlog_get_colorful()? CS_DEBUG : S_DEBUG); \
		if(qlog_get_print_time()) fprintf(stderr, "[%s] ", qlib::timestr().c_str()); \
		fprintf(stderr, __VA_ARGS__); \
		UNLOCK(); \
	}while(0)
#else
#define qlog_debug(...)
#endif

/**
 * Print something.
 */
#define qlog_info(...) \
	do { LOCK(); \
		fprintf(stderr, qlog_get_colorful()? CS_INFO : S_INFO); \
		if(qlog_get_print_time()) fprintf(stderr, "[%s] ", qlib::timestr().c_str()); \
		fprintf(stderr, __VA_ARGS__); \
		UNLOCK(); \
	}while(0)

/**
 * Assert, print where and exit on failure.
 */
#define qassert(x) \
	do { if(unlikely(!(x))) { LOCK(); \
		fprintf(stderr,"In %s(), %s:%d\n",__func__, __FILE__, __LINE__); \
		fprintf(stderr, qlog_get_colorful()? CS_ASSERT : S_ASSERT); \
		if(qlog_get_print_time()) fprintf(stderr, "[%s] ", qlib::timestr().c_str()); \
		fprintf(stderr, "'%s' failed.\n",#x); \
    throw std::runtime_error("Assertion failed."); \
		UNLOCK(); \
	}}while(0)

/**
 * Throw runtime_error
 */
#define qthrow(x) \
	do { LOCK(); \
		fprintf(stderr,"In %s(), %s:%d\n",__func__, __FILE__, __LINE__); \
		fprintf(stderr, qlog_get_colorful()? CS_THROW : S_THROW); \
		if(qlog_get_print_time()) fprintf(stderr, "[%s] ", qlib::timestr().c_str()); \
		fprintf(stderr, "RuntimeError: '%s'\n", x); \
    throw std::runtime_error(x); \
    UNLOCK(); \
	}while(0)

#endif // _QLOG_HPP_

