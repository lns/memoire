PROJ_ROOT=..

CXX=g++
CXXFLAGS=-std=c++11 -Wall -O2 -g
INCLUDES=-I$(PROJ_ROOT)/src
LDFLAGS=

main: main.cpp $(PROJ_ROOT)/src/*.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(INCLUDES) $(LDFLAGS)

clean:
	-rm main

.PHONY: clean

