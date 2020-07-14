CXX=clang++
CXX_FLAGS=-std=c++17 -O3 -ffast-math -mavx2 -mfma -fomit-frame-pointer

benchmark: benchmark.cpp
	$(CXX) $(CXX_FLAGS) $^ -o benchmark
	./benchmark
	rm -rf benchmark
