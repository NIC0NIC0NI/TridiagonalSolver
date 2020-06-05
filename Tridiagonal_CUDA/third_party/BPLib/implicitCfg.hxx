//- =======================================================================
//+ Implicit configuration file
//- =======================================================================

char implicitCfg[] =
  	"// Firstly, choose an algorithm. For example,        \n"
	"// $ ./BPLib alg=7                                         \n"
	"//                                                         \n"
	"// The options for the first digit (algorithm) are:        \n"
	
	"//  1 = BPLG Fourier                                       \n"
	"//  2 = BPLG Hartley                                       \n"
	"//  3 = BPLG Cosine                                        \n"
	"//  4 = BPLG RealFT                                         \n"
	"//  5 = BPLG Scan LF                                        \n"
	"//  6 = BPLG Scan KS                                        \n"
	"//  7 = BPLG BMCS Sorting                                   \n"
	"//  8 = BPLG Tridiag WM                                    \n"
	"//  9 = BPLG Tridiag CR                                    \n"
	"// 10 = BPLG Tridiag PCR                                   \n"
	"// 11 = BPLG Tridiag LF                                    \n"
	"//                                                         \n"
	"/ *** Configuration by default and parameters list:   *** /\n"
	//"cfgFile = config.ini // config file by default"
	"gpuid=  0;       // GPU ID on MultiGPU Platforms           \n"
	"xmin =  4;       // Minimum size of 1D-Transforms          \n"
	"ymin =  1;       // Minimum size of 2D-Transforms          \n"
	"ymax =  1;       // Maximum size of 2D-Transforms	    \n"
	"alg  =  0;       // Chosen algorithm ID		    \n"
	"\n"
	"[release]        /* ----- Release Section ----- */         \n"
	"nxn  =        1; // If executing 2D-problems, they are square\n"
	"sec  =       10; // Time for each benchmark	            \n"
	"xmax =     8192; // Maximum size for 1D-Transforms         \n"
	"mem  = 16777216; // Allocated memory for executions        \n"
	"\n"
	"[debug]          /* -----   Debug Section ----- */         \n"
	"nxn  =      0;   // Execute random 2D-problems             \n"
	"sec     =   1;   // Time for each benchmark                \n"
	"xmax    = 512;   // Maximum size for 1D-Transforms         \n"
	"mem     = 512;   // Allocated memory for executions        \n"
	"verbose =   0;   // Verbose amount during debug	    \n"
;




