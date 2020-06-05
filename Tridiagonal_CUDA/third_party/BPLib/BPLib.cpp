//- =======================================================================
//+ BPLib v1.0
//+ Butterfly Processing Library
//- -----------------------------------------------------------------------
//+ Designed to improve general signal type handling
//- =======================================================================


//----- System Include Section -------------------------------------------
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <complex>
using namespace std;

//---- Custom Include Section --------------------------------------------
#include "tools/timer.hxx"
#include "tools/cuvector.hxx"
#include "tools/cfgmgr.hxx"
#include "implicitCfg.hxx"
#include "alg/Transform.hxx"

//---- Helper functions --------------------------------------------------
#define tError(...) { fprintf(stderr, "%s(%i) : %s > ", __FILE__, __LINE__, __FUNCTION__); \
		fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n\n"); exit(-1); }

typedef std::complex<float> Complex;

enum FFT_DIR {
	ALG_FWD_E =  3, // Mode forward, for comparing results
	ALG_FWD   =  1, // Mode forward by default, direct transform
	ALG_DEBUG =  0, // Mode debug, only exchange
	ALG_INV   = -1, // Mode backward by default, inverse transform
	ALG_INV_E = -3  // Mode backward, for comparing results
};

#ifdef _DEBUG
	const int debug = 1;
#else
	const int debug = 0;
#endif

//---- Signal Processing Algorithms --------------------------------------

// GPU Algorithms, based on BPLGFFT
#include "alg/BPLG/BPLGFourier.hxx"
#include "alg/BPLG/BPLGHartley.hxx"
#include "alg/BPLG/BPLGCosine.hxx"
#include "alg/BPLG/BPLGRealFT.hxx"
#include "alg/BPLG/BPLGScanLF.hxx"
#include "alg/BPLG/BPLGScanKS.hxx"
#include "alg/BPLG/BPLGSort.hxx"
#include "alg/BPLG/BPLGTridiagLF.hxx"
#include "alg/BPLG/BPLGTridiagCR.hxx"
#include "alg/BPLG/BPLGTridiagPCR.hxx"
#include "alg/BPLG/BPLGTridiagWM.hxx"


Transform* algInstance(int algNum, int dimx, int dimy = 1, int dimz = 1) {
	switch(algNum) {
		
		
		case  1: return new BPLGFourier (dimx, dimy, dimz);
		case  2: return new BPLGHartley (dimx, dimy, dimz);
		case  3: return new BPLGCosine  (dimx, dimy, dimz);
		case  4: return new BPLGRealFT  (dimx, dimy, dimz);
		case  5: return new BPLGScanLF  (dimx, dimy, dimz);
		case  6: return new BPLGScanKS  (dimx, dimy, dimz);
		case  7: return new BPLGSort	  (dimx, dimy, dimz);
		case  8: return new BPLGTridiagWM(dimx, dimy, dimz);
		case  9: return new BPLGTridiagCR(dimx,dimy,dimz);
		case 10: return new BPLGTridiagPCR(dimx,dimy,dimz);
		case 11: return new BPLGTridiagLF(dimx, dimy,dimz);
		
		default: tError("Invalid algorithm parameter"); break;
	}
	return 0;
}


//---- Main Code ---------------------------------------------------------

int main(int argc, char *argv[]) {
	// Load configuration
	char defaultCfg[16] = "config.ini";
	if(argc == 1) return printf("%s", implicitCfg);
	ConfigManager config(implicitCfg); // First, configuration by default
	config.load(argc == 2 ? argv[1] : defaultCfg); // Second, using the file
	config.setDomain(debug ? "debug" : "release");
	config.load(argc, argv); // Finally, argv configuration

	// Obtaining parameters
	int fft_xmin = config.getInt("xmin"); // Minimum Size 1D
	int fft_xmax = config.getInt("xmax"); // Maximum Size 1D
	int fft_ymin = config.getInt("ymin"); // Minimum Size 2D
	int fft_ymax = config.getInt("ymax"); // Maximum Size 2D
	int fft_alg  = config.getInt("alg");  // Choosen Algorithm
	int fft_mem  = config.getInt("mem");  // Allocated memory
	int fft_sec  = config.getInt("sec");  // Benchmark time
	int fft_nxn  = config.getInt("nxn");  // Only square problems
	int verbose  = config.getInt("verbose"); // Mode verbose

	// Parameters checking
	if(fft_xmax == 1) fft_xmin = 1;
	if(fft_xmax < fft_xmin) fft_xmax = fft_xmin;
	if(fft_ymax == 1) fft_ymin = 1;
	if(fft_ymax < fft_ymin) fft_ymax = fft_ymin;
	if(fft_xmin < 1) tError("X Dimension must be >= 1");
	if(fft_ymin < 1) tError("Y Dimension must be >= 1");
	

	// Blocking rows for Tridiagonal Systems
	if((fft_alg) > 7) {
		fft_ymin = fft_ymax = 4;
		fft_nxn = 0;
	}


	// GPU Initialization (by default id = 0)
	int gpuId = config.getInt("gpuid");
	printf("BPLib> Trying to use gpuid=%i\n", gpuId);
	CUVector<float>::gpuInit(gpuId);

	// Can the algorithm be instanced?
	Transform* testAlg = algInstance(fft_alg, 2);
	printf("BPLib> Algorithm '%s', %s mode\n",
		testAlg->toString(), debug ? "debug" : "release");
	delete testAlg;

	
	long long iters = 0; // Number of performed iters.
	Timer clk;           // Timer for measuring
	
	// Iterating over vertical sizes, power of two
	for(int dimY = fft_ymin; dimY <= fft_ymax; dimY *= 2) {
		if(dimY * fft_xmin > fft_mem) break;
		if(!fft_nxn) printf("BPLib> Launching N = {%i..%i, %i}:\n",
			fft_xmin, fft_xmax, dimY);

		for(int dimX = fft_xmin; dimX <= fft_xmax; dimX *= 2) {
			const int dimXY = dimX * dimY;
		
			if(dimXY == 1) continue;  
			if(dimXY > fft_mem) break; // Out of range
			const int batchXY = verbose & 0x02 ? 1 : fft_mem / dimXY;
			Transform *alg = 0, *ref = 0;

			// Ignoring no-square problems (optionally)
			if(fft_nxn && fft_ymax > 1 && dimX != dimY) continue;
		
			// Create an algorithm with the desired configuration
			alg = algInstance(fft_alg, dimX, dimY, batchXY)->init(!debug);

	
			
			// Checking if size can be executed by desired algorithm
			if(!alg->calc(ALG_FWD_E)) goto freeResources;
			

			
			
			
			if(fft_alg<=4)
			{
				// Executing taking time and iters.
				for(clk.start(), iters = 0; clk.time() <= fft_sec; iters++) {
					alg->calc(ALG_FWD); // Forward
								
					alg->calc(ALG_INV); // Backward
				}
			}
			else {
				for(clk.start(), iters = 0; clk.time() <= fft_sec; iters++) {
					alg->calc(ALG_FWD); // Forward
				}
			}
			
			alg->compare(NULL);
	
			// Showing results
			printf("BPLib> NxM = (%7i,%4i), b =%8i, it =%5lli",
				dimX, dimY, batchXY, iters);

			// Printing performance, FWD+REV => 2*batch
			if(iters > 0) {
				double time = clk.time();
				double sigTime = time / (2 * iters * batchXY);
				double gflops = (fft_alg<=4)?alg->gflops(time, 2 * iters) : alg->gflops(time, iters);
				printf(", t = %.1e, GF =%7.2f", sigTime, gflops);
				
			}
			printf("\n");

			freeResources:  
			if(alg) delete alg;
			if(ref) delete ref;
		}
	}

	cudaDeviceReset();
	printf("BPLib> End.\n");
}

