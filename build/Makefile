# makefile

ROOT=..
ENABLE_DEBUG=1

CXX=g++
CXXFLAGS=-std=c++11 -O2 -Wall -D_GLIBCXX_USE_CXX11_ABI=0 -g

ifeq ($(ENABLE_DEBUG),1)
	BACKWARD_MACRO=-DBACKWARD_HAS_BFD=1
	BACKWARD_LD=-lbfd
else
	BACKWARD_MACRO=
	BACKWARD_LD=
endif

PY27INC=`python -m pybind11 --includes`
PY34INC=`python3.4 -m pybind11 --includes`
PY36INC=-I$(ROOT)/include/python3.6m

INCLUDES=$(BACKWARD_MACRO) -I$(ROOT)/src -I$(HOME)/include
LDFLAGS=$(BACKWARD_LD) -lzmq -lpthread

all: memoire.py27.so memoire.py36.so

memoire.py27.so: $(ROOT)/memoire.cpp $(ROOT)/src/*
	$(CXX) $(CXXFLAGS) -fPIC -shared $(PY27INC) $(INCLUDES) -o $@ $< $(LDFLAGS)
	#scp $@ root@10.240.126.69:/data1/qing/memoire/

memoire.py36.so: $(ROOT)/memoire.cpp $(ROOT)/src/*
	$(CXX) $(CXXFLAGS) -fPIC -shared $(PY36INC) $(INCLUDES) -o $@ $< $(LDFLAGS)

clean:
	-rm -f memoire.so

.PHONY: clean

