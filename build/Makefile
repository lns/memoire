# makefile

ROOT=..
ENABLE_DEBUG=0
PYBIND11_ROOT=$(HOME)/ext/pybind11

CXX=g++
CXXFLAGS=-std=c++11 -O3 -Wall

ifeq ($(ENABLE_DEBUG),1)
	BACKWARD_MACRO=-DBACKWARD_HAS_BFD=1 -g -DTIME_CONNECTION
	BACKWARD_LD=-lbfd
else
	BACKWARD_MACRO=#-DNDEBUG
	BACKWARD_LD=
endif

# Default python interpreter
PYINC=`python -m pybind11 --includes`

# Build for specific version
#PY27INC=`python2.7 -m pybind11 --includes`
#PY34INC=`python3.4 -m pybind11 --includes`
#PYINC=$(PY27INC)

INCLUDES=$(BACKWARD_MACRO) -I$(ROOT)/src -I$(ROOT)/include -I$(PYBIND11_ROOT)/include 
LDFLAGS=$(BACKWARD_LD) -lzmq -lpthread

ALL=memoire.so

all: $(ALL)

memoire.so: $(ROOT)/memoire.cpp $(ROOT)/src/*
	$(CXX) $(CXXFLAGS) -fPIC -shared $(PYINC) $(INCLUDES) -o $@ $< $(LDFLAGS)

clean:
	-rm -f $(ALL)

.PHONY: clean

