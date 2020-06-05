include ../../Makefile.inc

FFTFILE1 = ../lib-BPLG/KCosine
FFTFILE2 = ../lib-BPLG/KFourier
FFTFILE3 = ../lib-BPLG/KHartley
FFTFILE4 = ../lib-BPLG/KReal
FFTFILE6 = ../lib-BPLG/KScanLF
FFTFILE7 = ../lib-BPLG/KScanKS
FFTFILE8 = ../lib-BPLG/KSort
FFTFILE9 = ../lib-BPLG/KTridiagLF
FFTFILE10= ../lib-BPLG/KTridiagCR
FFTFILE11= ../lib-BPLG/KTridiagPCR
FFTFILE12= ../lib-BPLG/KTridiagWM

BPLG_O = BPLGFourier.o BPLGCosine.o BPLGRealFT.o BPLGHartley.o BPLGScanLF.o BPLGScanKS.o BPLGSort.o BPLGTridiagLF.o BPLGTridiagCR.o BPLGTridiagPCR.o BPLGTridiagWM.o

all: $(FFTFILE1).o $(FFTFILE2).o $(FFTFILE3).o $(FFTFILE4).o $(FFTFILE6).o $(FFTFILE7).o $(FFTFILE8).o $(FFTFILE9).o $(FFTFILE10).o $(FFTFILE11).o $(FFTFILE12).o $(BPLG_O)  

../lib-BPLG/BPLGFourier.o : BPLGFourier.hxx BPLGFourier.cpp ../lib-BPLG/KFourier.hxx ../lib-BPLG/KFourier.o

../lib-BPLG/BPLGCosine.o : BPLGCosine.hxx BPLGCosine.cpp ../lib-BPLG/KCosine.hxx ../lib-BPLG/KCosine.o

../lib-BPLG/BPLGRealFT.o : BPLGRealFT.hxx BPLGRealFT.cpp ../lib-BPLG/KReal.hxx ../lib-BPLG/KReal.o

../lib-BPLG/BPLGHartley.o : BPLGHartley.hxx BPLGHartley.cpp ../lib-BPLG/KHartley.hxx ../lib-BPLG/KHartley.o

../lib-BPLG/BPLGScanLF.o : BPLGScanLF.hxx BPLGScanLF.cpp ../lib-BPLG/KScanLF.hxx ../lib-BPLG/KScanLF.o 

../lib-BPLG/BPLGScanKS.o : BPLGScanKS.hxx BPLGScanKS.cpp ../lib-BPLG/KScanKS.hxx ../lib-BPLG/KScanKS.o

../lib-BPLG/BPLGSort.o : BPLGSort.hxx BPLGSort.cpp ../lib-BPLG/KSort.hxx ../lib-BPLG/KSort.o

../lib-BPLG/BPLGTridiagLF.o : BPLGTridiagLF.hxx BPLGTridiagLF.cpp ../lib-BPLG/KTridiagLF.hxx ../lib-BPLG/KTridiagLF.o

../lib-BPLG/BPLGTridiagCR.o : BPLGTridiagCR.hxx BPLGTridiagCR.cpp ../lib-BPLG/KTridiagCR.hxx ../lib-BPLG/KTridiagCR.o

../lib-BPLG/BPLGTridiagPCR.o : BPLGTridiagPCR.hxx BPLGTridiagPCR.cpp ../lib-BPLG/KTridiagPCR.hxx ../lib-BPLG/KTridiagPCR.o

../lib-BPLG/BPLGTridiagWM.o : BPLGTridiagWM.hxx BPLGTridiagWM.cpp ../lib-BPLG/KTridiagWM.hxx ../lib-BPLG/KTridiagWM.o

$(FFTFILE1).o: $(FFTFILE1).hxx $(FFTFILE1).cu $(ALGFILE) KLauncher.hxx 
	nvcc $(IDIRS) $(CPPFLAGS) $(NVFLAGS) -c $(FFTFILE1).cu -o $(FFTFILE1).o

$(FFTFILE2).o: $(FFTFILE2).hxx $(FFTFILE2).cu $(ALGFILE) KLauncher.hxx 
	nvcc $(IDIRS) $(CPPFLAGS) $(NVFLAGS) -c $(FFTFILE2).cu -o $(FFTFILE2).o

$(FFTFILE3).o: $(FFTFILE3).hxx $(FFTFILE3).cu $(ALGFILE) KLauncher.hxx
	nvcc $(IDIRS) $(CPPFLAGS) $(NVFLAGS) -c $(FFTFILE3).cu -o $(FFTFILE3).o

$(FFTFILE4).o: $(FFTFILE4).hxx $(FFTFILE4).cu $(ALGFILE) KLauncher.hxx
	nvcc $(IDIRS) $(CPPFLAGS) $(NVFLAGS) -c $(FFTFILE4).cu -o $(FFTFILE4).o

$(FFTFILE6).o: $(FFTFILE6).hxx $(FFTFILE6).cu $(ALGFILE) KLauncher2.hxx
	nvcc $(IDIRS) $(CPPFLAGS) $(NVFLAGS) -c $(FFTFILE6).cu -o $(FFTFILE6).o

$(FFTFILE7).o: $(FFTFILE7).hxx $(FFTFILE7).cu $(ALGFILE) KLauncher2.hxx
	nvcc $(IDIRS) $(CPPFLAGS) $(NVFLAGS) -c $(FFTFILE7).cu -o $(FFTFILE7).o

$(FFTFILE8).o: $(FFTFILE8).hxx $(FFTFILE8).cu $(ALGFILE) KLauncher1.hxx
	nvcc $(IDIRS) $(CPPFLAGS) $(NVFLAGS) -c $(FFTFILE8).cu -o $(FFTFILE8).o

$(FFTFILE9).o: $(FFTFILE9).hxx $(FFTFILE9).cu $(ALGFILE) KLauncher3.hxx
	nvcc $(IDIRS) $(CPPFLAGS) $(NVFLAGS) -c $(FFTFILE9).cu -o $(FFTFILE9).o

$(FFTFILE10).o: $(FFTFILE10).hxx $(FFTFILE10).cu $(ALGFILE) KLauncher3.hxx
	nvcc $(IDIRS) $(CPPFLAGS) $(NVFLAGS) -c $(FFTFILE10).cu -o $(FFTFILE10).o

$(FFTFILE11).o: $(FFTFILE11).hxx $(FFTFILE11).cu $(ALGFILE) KLauncher3.hxx
	nvcc $(IDIRS) $(CPPFLAGS) $(NVFLAGS) -c $(FFTFILE11).cu -o $(FFTFILE11).o

$(FFTFILE12).o: $(FFTFILE12).hxx $(FFTFILE12).cu $(ALGFILE) KLauncher3.hxx
	nvcc $(IDIRS) $(CPPFLAGS) $(NVFLAGS) -c $(FFTFILE12).cu -o $(FFTFILE12).o

clean:
	rm -rf $(FFTFILE1).o $(FFTFILE2).o $(FFTFILE3).o $(FFTFILE4).o $(FFTFILE6).o $(FFTFILE7).o $(FFTFILE8).o $(FFTFILE9).o $(FFTFILE10).o $(FFTFILE11).o $(FFTFILE12).o
	rm -rf $(BPLG_O)
	
	

