//- =======================================================================
//+ BPLG Scan LF
//- =======================================================================

//---- Header Dependencies -----------------------------------------------
#include "BPLGScanLF.hxx"
#include "../lib-BPLG/KScanLF.hxx"

//---- Library Header Definitions ----------------------------------------

// direct (dir=1) or inverse (dir=-1)
Transform* BPLGScanLF::calc(int dir) {
	// Load vectors and check pointers
	CUVector<float>* gpuVector = gpuVector_r();
	CUVector<float>::error("Invalid CUDA pointer", !gpuVector);
	float* data = (float*)gpuVector->gpuData();

	

	CUVector<float>* gpuVectorOut = gpuVector_r();
	CUVector<float>::error("Invalid CUDA pointer", !gpuVectorOut);
	float* dataOutput = (float*) gpuVectorOut->gpuData();

	int errCode = KScanLF(data,dataOutput, dir, getX(), getY(), getZ());
	CUVector<float>::error("Error launching BPLGScanLF");
	CUVector<float>::sync();
	switch(dir) { 
		case  3:  
			updateCpu_r(); break;
		case -3: 
			updateCpu_r(); break;
	}

	return errCode ? NULL : this;
}
