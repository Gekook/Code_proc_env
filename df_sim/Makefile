# Makefile for df.cpp
# Builds the shallow-flow solver and provides run/clean/help targets

CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wall -Wextra
TARGET = df
SRC = df.cpp

.PHONY: all run clean help

# Default target: build the executable
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)
	@echo "✓ Built $(TARGET)"

# Run the program (after building)
run: $(TARGET)
	./$(TARGET)

# Remove the executable and generated CSV files
clean:
	rm -f $(TARGET) eta_*.csv
	@echo "✓ Cleaned"

# Short help
help:
	@echo "Usage: make [target]"
	@echo "Targets:"
	@echo "  all    - Build the program (default)"
	@echo "  run    - Build and run the program" 
	@echo "  clean  - Remove the executable and CSV outputs"
	@echo "  help   - Show this message"
