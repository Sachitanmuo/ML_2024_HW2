HW2: main.cpp Model.cpp
	g++ -std=c++11 main.cpp Model.cpp -o HW2
PYTHON_INCLUDE = -I/usr/include/python3.11.5

your_target: your_source.cpp
	g++ -o your_target your_source.cpp $(PYTHON_INCLUDE) -lpythonX.Y
.PHONY: clean
clean:
	rm -f HW2