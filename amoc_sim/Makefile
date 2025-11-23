# Makefile per amoc.cpp
# Compila il programma di simulazione AMOC e genera il plot

CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O2
TARGET = amoc
SRC = amoc.cpp

.PHONY: all run clean help

# Target di default
all: $(TARGET)

# Compila l'eseguibile
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)
	@echo "✓ Compilation completed: $(TARGET)"

# Esegui il programma
run: $(TARGET)
	./$(TARGET)

# Pulisci i file generati
clean:
	rm -f $(TARGET) *.dat *.gnuplot
	@echo "✓ Cleaning completed"

# Aiuto
help:
	@echo "Use : make [target]"
	@echo ""
	@echo "Target available:"
	@echo "  all    - Compiles the program (default)"
	@echo "  run    - Compiles and executes the program"
	@echo "  clean  - Remove files generated"
	@echo "  help   - Show this message"
