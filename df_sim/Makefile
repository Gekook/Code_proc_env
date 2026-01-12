# Makefile for df.cpp
# Builds the shallow-flow solver and provides run/clean/help targets

CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra
INCLUDES = -I /usr/include/eigen3

TARGET = df2
SRC = df2.cpp

.PHONY: all run clean help

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TARGET) $(SRC)
	@echo "✓ Built $(TARGET)"

# Run the program
run: $(TARGET)
	./$(TARGET)

# Clean executable and outputs
clean:
	rm -f $(TARGET) eta_*.csv
	@echo "✓ Cleaned"

# Help
help:
	@echo "Usage: make [target]"
	@echo "Targets:"
	@echo "  all    - Build the program (default)"
	@echo "  run    - Build and run the program"
	@echo "  clean  - Remove the executable and CSV outputs"
	@echo "  help   - Show this message"
