# Makefile for amoc.cpp
# Builds the AMOC simulation program and generates the plot

CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O2
TARGET = amoc
SRC = amoc.cpp

.PHONY: all run clean help

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)
	@echo "✓ Compilation completed: $(TARGET)"

# Run the program
run: $(TARGET)
	./$(TARGET)

# Clean generated files
clean:
	rm -f $(TARGET) *.dat *.gnuplot
	@echo "✓ Cleaning completed"

# Help
help:
	@echo "Use : make [target]"
	@echo ""
	@echo "Target available:"
	@echo "  all    - Compiles the program (default)"
	@echo "  run    - Compiles and executes the program"
	@echo "  clean  - Remove generated files"
	@echo "  help   - Show this message"
