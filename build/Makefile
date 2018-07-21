# makefile

ROOT=..
ENABLE_DEBUG=1
PYBIND11_ROOT=$(HOME)/ext/pybind11

CXX=g++
CXXFLAGS=-std=c++11 -O2 -Wall -D_GLIBCXX_USE_CXX11_ABI=0 -g -DTIME_CONNECTION

ifeq ($(ENABLE_DEBUG),1)
	BACKWARD_MACRO=-DBACKWARD_HAS_BFD=1
	BACKWARD_LD=-lbfd
else
	BACKWARD_MACRO=
	BACKWARD_LD=
endif

PY27INC=-I$(PYBIND11_ROOT)/include `python2.7 -m pybind11 --includes`
PY34INC=`python3.4 -m pybind11 --includes`

PYINC=$(PY27INC)

INCLUDES=$(BACKWARD_MACRO) -I$(ROOT)/src -I$(ROOT)/include
LDFLAGS=$(BACKWARD_LD) -lzmq -lpthread

ALL=memoire.so

all: $(ALL)

memoire.so: $(ROOT)/memoire.cpp $(ROOT)/src/*
	$(CXX) $(CXXFLAGS) -fPIC -shared $(PYINC) $(INCLUDES) -o $@ $< $(LDFLAGS)

clean:
	-rm -f $(ALL)

.PHONY: clean

