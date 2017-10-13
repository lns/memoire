PROJ_ROOT=..
ENABLE_DEBUG=1

CXX=g++
CXXFLAGS=-std=c++11 -Wall -O2 -g

ifeq ($(ENABLE_DEBUG),1)
	BACKWARD_MACRO=-DBACKWARD_HAS_BFD=1
	BACKWARD_LD=-lbfd
else
	BACKWARD_MACRO=
	BACKWARD_LD=
endif

INCLUDES=$(BACKWARD_MACRO) -I$(PROJ_ROOT)/src
LDFLAGS=$(BACKWARD_LD) -lpthread


main: main.cpp $(PROJ_ROOT)/src/*.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(INCLUDES) $(LDFLAGS)

clean:
	-rm main

.PHONY: clean

