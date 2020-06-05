# ----------------------------
#    Compiler configuration
# ----------------------------

# C configuration
CC       = gcc
CFLAGS   = -O2 -Wall -march=native -fopenmp -D CUDA_ARCH=700

# C++ configuration
CXX      = g++
CXXFLAGS = -O2 -Wall -march=native --std=c++0x -fopenmp $(IDIRS) -D CUDA_ARCH=700

# CPP configuration
CPPFLAGS = -DNDEBUG

# NVCC configuration
DIVISION = -use_fast_math
NVFLAGS  = $(DIVISION) --gpu-architecture=compute_70 --gpu-code=sm_70 -D CUDA_ARCH=700

# ----------------------------
#    Default directories
# ----------------------------

# Cuda directories
CUDA_HOME = /usr/local/cuda-10.0
CUDALIB  = $(CUDA_HOME)/lib64
CUDAINC  = $(CUDA_HOME)/include

# Additional include directories
IDIRS    = -I. -I.. -I$(CUDAINC) -I../..
WIN95LIB = ./lib

# ----------------------------
#    Linker options
# ----------------------------

# Linker flags
LDFLAGS  = -Wl,--no-as-needed -L$(CUDALIB) -L/usr/lib64 -lcudart -lcuda -lcufft -fopenmp -openmp -lcusparse -lm -lstdc++


#In case of compiling with Windows, add
#-L$(WIN95LIB) after -L$(CUDALIB)
#in the Liker flags line.
