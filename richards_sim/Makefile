# Makefile per il progetto `richards_sim`
# Compila `richards.cpp` in eseguibile `richards` con supporto Eigen.

CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -I/usr/include/eigen3
LDFLAGS ?=

TARGET := richards
SRCS := richards.cpp
OBJS := $(SRCS:.cpp=.o)

.PHONY: all run debug clean

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

run: all
	./$(TARGET)

debug: CXXFLAGS += -g -O0
debug: clean all

clean:
	rm -f $(TARGET) $(OBJS) *.csv

