CXX      := g++
CXXFLAGS := -std=c++17 -O2 -pipe -Wall

SRC  := main.cpp
TARGET := tsp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@

run: $(TARGET)
	@echo "Usage: make run file=<graph.txt>"
	@./$(TARGET) $(file)

clean:
	rm -f $(TARGET)
