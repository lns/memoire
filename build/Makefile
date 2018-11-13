# makefile

ROOT=..
ENABLE_DEBUG=0
PYBIND11_ROOT=$(HOME)/ext/pybind11
ZMQ_ROOT=$(HOME)/ext/libzmq/prefix
PROTOBUF_ROOT=$(HOME)/ext/protobuf/build
# libprotobuf.a should be compiled with CXXFLAGS=-fPIC, or the compilation of memoire.so
# may fail with a relocation error. ("..can not be used when making a shared object;")
PROTOBUF_STATIC=$(PROTOBUF_ROOT)/lib/libprotobuf.a
ZMQ_STATIC=$(ZMQ_ROOT)/lib/libzmq.a

CXX=g++
CXXFLAGS=-std=c++11 -O3 -Wall -fvisibility=hidden
PROTOC=$(PROTOBUF_ROOT)/bin/protoc

ifeq ($(ENABLE_DEBUG),1)
	BACKWARD_MACRO=-DBACKWARD_HAS_BFD=1 -g -DPRINT_DEBUG
	BACKWARD_LD=-lbfd
else
	BACKWARD_MACRO=#-DNDEBUG
	BACKWARD_LD=
endif

# Default python interpreter
PYINC=`python3 -m pybind11 --includes`

# Build for specific version
#PY27INC=`python2.7 -m pybind11 --includes`
#PY34INC=`python3.4 -m pybind11 --includes`
#PYINC=$(PY27INC)

INCLUDES=$(BACKWARD_MACRO) -I. -I$(ROOT)/src -I$(ROOT)/include -I$(PYBIND11_ROOT)/include -I$(ZMQ_ROOT)/include -I$(PROTOBUF_ROOT)/include
LDFLAGS=$(BACKWARD_LD) -lpthread

ALL=memoire.so memoire.o msg.pb.o

all: $(ALL)

msg.pb.cc: $(ROOT)/proto/*.proto
	$(PROTOC) --cpp_out=. --python_out=. --proto_path=$(ROOT)/proto/ $(ROOT)/proto/*.proto

msg.pb.o: msg.pb.cc $(ROOT)/proto/*.proto
	$(CXX) $(CXXFLAGS) -fPIC -c $(INCLUDES) $(PYINC) -o $@ $<

memoire.o: $(ROOT)/memoire.cpp msg.pb.cc $(ROOT)/src/* $(ROOT)/include/*
	$(CXX) $(CXXFLAGS) -fPIC -c $(INCLUDES) $(PYINC) -o $@ $<

memoire.so: memoire.o msg.pb.o $(PROTOBUF_STATIC) $(ZMQ_STATIC)
	$(CXX) $(CXXFLAGS) -fPIC -shared $(INCLUDES) $(PYINC) -o $@ $^ $(LDFLAGS)

clean:
	-rm -f $(ALL) msg.pb.* msg_pb2.py

.PHONY: clean

