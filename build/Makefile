# makefile

ROOT=..
ENABLE_DEBUG=1
PYTHON_VERSION=2 # python2.* or python3.*

CXX=g++
CXXFLAGS=-std=c++11 -O2 -Wall -D_GLIBCXX_USE_CXX11_ABI=0 -g

ifeq ($(ENABLE_DEBUG),1)
	BACKWARD_MACRO=-DBACKWARD_HAS_BFD=1
	BACKWARD_LD=-lbfd
else
	BACKWARD_MACRO=
	BACKWARD_LD=
endif

ifeq ($(PYTHON_VERSION),3)
	PYINC=`python3.4 -m pybind11 --includes`
else
	PYINC=`python -m pybind11 --includes`
endif

INCLUDES=$(BACKWARD_MACRO) $(PYINC) -I$(ROOT)/src -I$(HOME)/include
LDFLAGS=$(BACKWARD_LD) -lzmq -lpthread

all: memoire.so

memoire.so: $(ROOT)/src/libmemoire.cpp $(ROOT)/src/*
	$(CXX) $(CXXFLAGS) -fPIC -shared $(INCLUDES) -o $@ $< $(LDFLAGS)
	scp $@ root@10.240.126.69:/data1/qing/memoire/

clean:
	-rm -f memoire.so

.PHONY: clean

