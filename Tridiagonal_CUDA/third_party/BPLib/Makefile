include Makefile.inc

default: alg/lib-BPLG/KTridiagLF.o alg/lib-BPLG/KTridiagCR.o alg/lib-BPLG/KTridiagPCR.o alg/lib-BPLG/KTridiagWM.o

run: BPLib
	./BPLib

clean:
	rm -Rf x64
	rm -f a.out BPLib.o BPLib
	cd alg; make clean
	cd inc; make clean
	cd tools; make clean

# BPLG GPU implementation

BPLG_H = alg/BPLG/BPLGFourier.hxx alg/BPLG/BPLGHartley.hxx alg/BPLG/BPLGCosine.hxx alg/BPLG/BPLGRealFT.hxx alg/BPLG/BPLGScanLF.hxx alg/BPLG/BPLGScanKS.hxx alg/BPLG/BPLGSort.hxx alg/BPLG/BPLGTridiagLF.hxx alg/BPLG/BPLGTridiagCR.hxx alg/BPLG/BPLGTridiagPCR.hxx alg/BPLG/BPLGTridiagWM.hxx
BPLG_O = alg/BPLG/BPLGFourier.o alg/BPLG/BPLGHartley.o alg/BPLG/BPLGCosine.o alg/BPLG/BPLGRealFT.o alg/BPLG/BPLGScanLF.o alg/BPLG/BPLGScanKS.o alg/BPLG/BPLGSort.o alg/BPLG/BPLGTridiagLF.o alg/BPLG/BPLGTridiagCR.o alg/BPLG/BPLGTridiagPCR.o alg/BPLG/BPLGTridiagWM.o
BPLG_K = alg/lib-BPLG/KFourier.o alg/lib-BPLG/KHartley.o alg/lib-BPLG/KCosine.o alg/lib-BPLG/KReal.o alg/lib-BPLG/KScanLF.o alg/lib-BPLG/KScanKS.o alg/lib-BPLG/KSort.o alg/lib-BPLG/KTridiagLF.o alg/lib-BPLG/KTridiagCR.o alg/lib-BPLG/KTridiagPCR.o alg/lib-BPLG/KTridiagWM.o

alg/BPLG/BPLGFourier.o:
	cd alg/BPLG; make
	
alg/BPLG/BPLGHartley.o:
	cd alg/BPLG; make

alg/BPLG/BPLGCosine.o:
	cd alg/BPLG; make

alg/BPLG/BPLGRealFT.o:
	cd alg/BPLG; make

alg/BPLG/BPLGScanLF.o:
	cd alg/BPLG; make

alg/BPLG/BPLGScanKS.o:
	cd alg/BPLG; make

alg/BPLG/BPLGSort.o:
	cd alg/BPLG; make

alg/BPLG/BPLGTridiagLF.o:
	cd alg/BPLG; make

alg/BPLG/BPLGTridiagCR.o:
	cd alg/BPLG; make

alg/BPLG/BPLGTridiagPCR.o:
	cd alg/BPLG; make

alg/BPLG/BPLGTridiagWM.o:
	cd alg/BPLG; make

alg/BPLG/KFourier.o:
	cd alg/lib-BPLG; make

alg/BPLG/KHartley.o:
	cd alg/lib-BPLG; make
	
alg/BPLG/KReal.o:
	cd alg/lib-BPLG; make

alg/BPLG/KCosine.o:
	cd alg/lib-BPLG; make

alg/BPLG/KScanLF.o:
	cd alg/lib-BPLG; make

alg/BPLG/KScanKS.o:
	cd alg/lib-BPLG; make

alg/BPLG/KSort.o:
	cd alg/lib-BPLG; make

alg/BPLG/KTridiagLF.o:
	cd alg/lib-BPLG; make

alg/BPLG/KTridiagCR.o:
	cd alg/lib-BPLG; make

alg/BPLG/KTridiagPCR.o:
	cd alg/lib-BPLG; make

alg/BPLG/KTridiagWM.o:
	cd alg/BPLG; make



# Algorithm related object files

ALGORITHMS_H = $(BPLG_H)
ALGORITHMS_O = $(BPLG_O)
ALGORITHMS_K = $(BPLG_K)

TRANSFORM_H = alg/Transform.hxx
TRANSFORM_O = alg/Transform.o

alg/Transform.o:
	cd alg; make

# Helper tools object files

TOOLS_H = tools/cfgmgr.hxx tools/cudacl.hxx tools/cuvector.hxx tools/timer.hxx tools/tausgen.hxx
TOOLS_O = tools/cfgmgr.o tools/cuvector.o tools/timer.o tools/tausgen.o

tools/cfgmgr.o:
	cd tools; make

tools/cuvector.o:
	cd tools; make

tools/timer.o:
	cd tools; make


# Main program build

BPLib: BPLib.o $(ALGORITHMS_O) $(ALGORITHMS_K) $(TRANSFORM_O) $(TOOLS_O)

BPLib.o: BPLib.cpp $(TOOLS_H) $(TRANSFORM_H) $(ALGORITHMS_H) 

# Comprobar si es necesario
.PHONY: $(TOOLS_O) $(ALGORITHMS_O) $(TRANSFORM_O)
