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

INCLUDES=$(BACKWARD_MACRO) -I$(PROJ_ROOT)/src -I$(HOME)/include
LDFLAGS=$(BACKWARD_LD) -lzmq -lpthread

ALL=main server

all: $(ALL)

main: main.cpp $(PROJ_ROOT)/src/*.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(INCLUDES) $(LDFLAGS)

server: server.cpp  $(PROJ_ROOT)/src/*.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(INCLUDES) $(LDFLAGS)

clean:
	-rm -f $(ALL)

.PHONY: clean

