default:test

asm:tridiagonal_s_avx512.S tridiagonal_d_avx512.S

CC = icc
CXX = icpc

MIC_ARCH = -xmic-avx512
DIVISION = #-DDIV

OMP_OPT = -qopenmp -DTHREADING=1
OMP_LIB = -qopenmp -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread
TBB_OPT = -qopenmp-simd -DTHREADING=2
TBB_LIB = -ltbb -lmkl_core -lmkl_intel_lp64 -lmkl_tbb_thread 

OPT = -O3 $(OMP_OPT) $(MIC_ARCH) -DNDEBUG $(DIVISION)

CFLAGS = -std=c11 $(OPT)
CXXFLAGS = -std=c++11 $(OPT) #-qopt-report-phase=vec #-no-prec-div -no-prec-sqrt
CLIB = $(OMP_LIB)

OBJECTS = tridiagonal_d_avx512.o tridiagonal_s_avx512.o tridiagonal_basic.o test.o check.o

tridiagonal_d_avx512.o: tridiagonal_d_avx512.cpp tridiagonal.h vec.h threading.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
tridiagonal_s_avx512.o: tridiagonal_s_avx512.cpp tridiagonal.h vec.h threading.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
tridiagonal_d_avx512.S: tridiagonal_d_avx512.cpp tridiagonal.h vec.h threading.h
	$(CXX) $(CXXFLAGS) -S $< -o $@
tridiagonal_s_avx512.S: tridiagonal_s_avx512.cpp tridiagonal.h vec.h threading.h
	$(CXX) $(CXXFLAGS) -S $< -o $@

tridiagonal_basic.o: tridiagonal_basic.cpp tridiagonal.h threading.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
test.o: test.cpp tridiagonal.h helpers.h threading.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
check.o: check.c tridiagonal.h
	$(CC) $(CFLAGS) -c $< -o $@

test:$(OBJECTS)
	$(CXX) $(CLIB) $^ -o $@

clean:
	rm test *.o
