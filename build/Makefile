# makefile

ROOT=..
ENABLE_DEBUG=1
PYBIND11_ROOT=$(HOME)/ext/pybind11
# libprotobuf.a should be compiled with CXXFLAGS=-fPIC, or the compilation of memoire.so
# may fail with a relocation error. ("..can not be used when making a shared object;")
PROTOBUF_STATIC=/usr/local/lib/libprotobuf.a

CXX=g++
CXXFLAGS=-std=c++11 -O3 -Wall -fvisibility=hidden

ifeq ($(ENABLE_DEBUG),1)
	BACKWARD_MACRO=-DBACKWARD_HAS_BFD=1 -g -DPRINT_DEBUG
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

INCLUDES=$(BACKWARD_MACRO) -I$(ROOT)/src -I$(ROOT)/include -I$(ROOT)/proto -I$(PYBIND11_ROOT)/include 
LDFLAGS=$(BACKWARD_LD) -lzmq -lpthread

ALL=memoire.so memoire.o msg.pb.o

all: $(ALL)

msg.pb.o: $(ROOT)/proto/msg.pb.cc $(ROOT)/proto/*.proto
	$(CXX) $(CXXFLAGS) -fPIC -c $(INCLUDES) $(PYINC) -o $@ $<

memoire.o: $(ROOT)/memoire.cpp $(ROOT)/src/* $(ROOT)/include/*
	$(CXX) $(CXXFLAGS) -fPIC -c $(INCLUDES) $(PYINC) -o $@ $<

memoire.so: memoire.o msg.pb.o $(PROTOBUF_STATIC)
	$(CXX) $(CXXFLAGS) -fPIC -shared $(INCLUDES) $(PYINC) -o $@ $^ $(LDFLAGS)

clean:
	-rm -f $(ALL)

.PHONY: clean

