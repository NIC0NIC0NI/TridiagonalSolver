default:test tridiagonal_wm_pcr.ptx resource_usage.txt

REGLIMIT = #--maxrregcount 70
GPU_FLAGS = -arch=sm_70
CUDA_PATH = /usr/local/cuda-10.0
INTEL_PATH = /opt/intel

CXX = icpc
CUCC = nvcc

BPLG = ./third_party/BPLib
BPLIBFLG = -I$(BPLG) -DBPLG
BPLIBOBJ = tridiagonal_third_party.o $(BPLG)/alg/lib-BPLG/KTridiagCR.o $(BPLG)/alg/lib-BPLG/KTridiagLF.o $(BPLG)/alg/lib-BPLG/KTridiagWM.o $(BPLG)/alg/lib-BPLG/KTridiagPCR.o

CUFLAGS = --std c++11 -O3 $(GPU_FLAGS) $(BPLIBFLG)
CXXFLAGS = -std=c++11 -O3
CULIBS = -L$(CUDA_PATH)/lib64  -L$(CUDA_PATH)/lib64/stubs -lcuda -lcudart -lcusparse
CXXLIBS = -mkl=parallel $(CULIBS)

OBJECTS = tridiagonal_wm_pcr.o tridiagonal_basic.o tridiagonal_onchip.o test.o utils.o $(BPLIBOBJ)

.PHONY:kernels
kernels: tridiagonal_wm_pcr.cu tridiagonal.h intrin.h
	$(CUCC) $(CUFLAGS) --resource-usage -c $< -o tridiagonal_wm_pcr.o 2> resource_usage_raw.txt

tridiagonal_wm_pcr.o: tridiagonal_wm_pcr.cu tridiagonal.h intrin.h
	$(CUCC) $(CUFLAGS) $(REGLIMIT) --resource-usage -c $< -o $@ 2> resource_usage_raw.txt
tridiagonal_wm_pcr.ptx: tridiagonal_wm_pcr.cu tridiagonal.h intrin.h
	$(CUCC) $(CUFLAGS) $(REGLIMIT) -ptx $< -o $@

tridiagonal_basic.o: tridiagonal_basic.cu tridiagonal.h
	$(CUCC) $(CUFLAGS) -c $< -o $@
tridiagonal_third_party.o: tridiagonal_third_party.cu tridiagonal.h
	$(CUCC) $(CUFLAGS) -c $< -o $@
tridiagonal_onchip.o: tridiagonal_onchip.cu tridiagonal.h
	$(CUCC) $(CUFLAGS) -c $< -o $@
test.o: test.cpp tridiagonal.h helpers.h utils.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
utils.o: utils.cu tridiagonal.h helpers.h utils.h
	$(CUCC) $(CUFLAGS) -c $< -o $@

test:$(OBJECTS)
	$(CXX) $^ -o $@ $(CXXLIBS)
resource_usage_format:resource_usage_format.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
resource_usage_raw.txt: tridiagonal_wm_pcr.o
resource_usage.txt:resource_usage_format resource_usage_raw.txt
	./resource_usage_format < resource_usage_raw.txt

third_party_bplg:
	$(MAKE) -C $(BPLG)

$(BPLG)/alg/lib-BPLG/KTridiagCR.o:third_party_bplg
$(BPLG)/alg/lib-BPLG/KTridiagLF.o:third_party_bplg
$(BPLG)/alg/lib-BPLG/KTridiagWM.o:third_party_bplg
$(BPLG)/alg/lib-BPLG/KTridiagPCR.o:third_party_bplg

clean:
	rm test resource_usage_format *.o
