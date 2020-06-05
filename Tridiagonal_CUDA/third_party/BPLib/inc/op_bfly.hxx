//- =======================================================================
//+ Butterfly operator 
//- =======================================================================

#pragma once
#ifndef _OP_BFLY
#define _OP_BFLY

//---- Header Dependencies -----------------------------------------------

#include "inc/complex.hxx"

//---- Butterfly transform for single vector -----------------------------
template<int RAD, int DIR> inline __device__ void
butterfly(COMPLEX* a, int stride = 1) { }

//---- Batch butterfly transform -----------------------------------------
//- Stride is not supported for these functions

template<int SIZE, int RAD, int DIR> inline __device__ void
butterfly(COMPLEX* data) {
	#pragma unroll
	for(int i = 0; i < SIZE; i += RAD)
		butterfly<RAD, DIR>(data + i);
}

//---- Buttefly definitions for scalar variables -------------------------

// Precomputed angles for "cos(x * (PI/2) / 8)", "x = [1,7]"
#define ANG_1_8 0.98078528040323044912618223613424
#define ANG_2_8 0.92387953251128675612818318939679
#define ANG_3_8 0.83146961230254523707878837761791
#define ANG_4_8 0.70710678118654752440084436210485
#define ANG_5_8 0.55557023301960222474283081394853
#define ANG_6_8 0.38268343236508977172845998403040
#define ANG_7_8 0.19509032201612826784828486847702


template<> inline __device__ void
butterfly< 2, 1>(COMPLEX* data, int stride)
{ 
    COMPLEX t0 = data[0];
    data[ 0 * stride] = t0 + data[1 * stride]; 
    data[ 1 * stride] = t0 - data[1 * stride];
}

template<> inline __device__ void
butterfly< 2,-1>(COMPLEX* data, int stride)
{ 
    COMPLEX t0 = data[0];
    data[ 0 * stride] = t0 + data[1 * stride]; 
    data[ 1 * stride] = t0 - data[1 * stride];
}

template<> inline __device__ void
butterfly< 4, 1>(COMPLEX* data, int stride)
{ 
	butterfly< 2, 1>(data + 0 * stride, 2 * stride);
	butterfly< 2, 1>(data + 1 * stride, 2 * stride);

    COMPLEX t3 = data[3 * stride];
	data[ 3 * stride] = make_COMPLEX(t3.y, -t3.x);

	butterfly< 2, 1>(data + 0 * stride, 1 * stride);
	butterfly< 2, 1>(data + 2 * stride, 1 * stride);
}

template<> inline __device__ void
butterfly< 4,-1>(COMPLEX* data, int stride)
{ 
	butterfly< 2,-1>(data + 0 * stride, 2 * stride);
	butterfly< 2,-1>(data + 1 * stride, 2 * stride);

    COMPLEX t3 = data[3 * stride];
	data[ 3 * stride] = make_COMPLEX(-t3.y, t3.x);

	butterfly< 2,-1>(data + 0 * stride, 1 * stride);
	butterfly< 2,-1>(data + 2 * stride, 1 * stride);
}

template<> inline __device__ void
butterfly< 8, 1>(COMPLEX* data, int stride)
{ 
	butterfly< 4, 1>(data + 0 * stride, 2 * stride);
	butterfly< 4, 1>(data + 1 * stride, 2 * stride);

    COMPLEX t3 = data[3 * stride];
	data[ 3 * stride]  = make_COMPLEX(    t3.y,-   t3.x);
	data[ 5 * stride] *= make_COMPLEX( ANG_4_8,-ANG_4_8);
	data[ 7 * stride] *= make_COMPLEX(-ANG_4_8,-ANG_4_8);

	butterfly< 2, 1>(data + 0 * stride, 1 * stride);
	butterfly< 2, 1>(data + 2 * stride, 1 * stride);
	butterfly< 2, 1>(data + 4 * stride, 1 * stride);
	butterfly< 2, 1>(data + 6 * stride, 1 * stride);
}

template<> inline __device__ void
butterfly< 8,-1>(COMPLEX* data, int stride)
{ 
	butterfly< 4,-1>(data + 0 * stride, 2 * stride);
	butterfly< 4,-1>(data + 1 * stride, 2 * stride);

    COMPLEX t3 = data[3 * stride];
	data[ 3 * stride]  = make_COMPLEX(   -t3.y,    t3.x);
	data[ 5 * stride] *= make_COMPLEX( ANG_4_8, ANG_4_8);
	data[ 7 * stride] *= make_COMPLEX(-ANG_4_8, ANG_4_8);

	butterfly< 2,-1>(data + 0 * stride, 1 * stride);
	butterfly< 2,-1>(data + 2 * stride, 1 * stride);
	butterfly< 2,-1>(data + 4 * stride, 1 * stride);
	butterfly< 2,-1>(data + 6 * stride, 1 * stride);
}

template<> inline __device__ void
butterfly<16, 1>(COMPLEX* data, int stride)
{ 
	butterfly< 4, 1>(data + 0 * stride, 4 * stride);
	butterfly< 4, 1>(data + 1 * stride, 4 * stride);
	butterfly< 4, 1>(data + 2 * stride, 4 * stride);
	butterfly< 4, 1>(data + 3 * stride, 4 * stride);

    COMPLEX t6 = data[6 * stride];
	data[ 5 * stride] *= make_COMPLEX( ANG_4_8,-ANG_4_8);
	data[ 6 * stride]  = make_COMPLEX(    t6.y,-   t6.x);
	data[ 7 * stride] *= make_COMPLEX(-ANG_4_8,-ANG_4_8);

	data[ 9 * stride] *= make_COMPLEX( ANG_2_8,-ANG_6_8);
	data[10 * stride] *= make_COMPLEX( ANG_4_8,-ANG_4_8);
	data[11 * stride] *= make_COMPLEX( ANG_6_8,-ANG_2_8);

	data[13 * stride] *= make_COMPLEX( ANG_6_8,-ANG_2_8);
	data[14 * stride] *= make_COMPLEX(-ANG_4_8,-ANG_4_8);
	data[15 * stride] *= make_COMPLEX(-ANG_2_8, ANG_6_8);

	butterfly< 4, 1>(data + 0 * stride, 1 * stride);
	butterfly< 4, 1>(data + 4 * stride, 1 * stride);
	butterfly< 4, 1>(data + 8 * stride, 1 * stride);
	butterfly< 4, 1>(data +12 * stride, 1 * stride);
}

template<> inline __device__ void
butterfly<16,-1>(COMPLEX* data, int stride)
{ 
	butterfly< 4,-1>(data + 0 * stride, 4 * stride);
	butterfly< 4,-1>(data + 1 * stride, 4 * stride);
	butterfly< 4,-1>(data + 2 * stride, 4 * stride);
	butterfly< 4,-1>(data + 3 * stride, 4 * stride);

    COMPLEX t6 = data[6 * stride];
	data[ 5 * stride] *= make_COMPLEX( ANG_4_8, ANG_4_8);
	data[ 6 * stride]  = make_COMPLEX(-   t6.y,    t6.x);
	data[ 7 * stride] *= make_COMPLEX(-ANG_4_8, ANG_4_8);

	data[ 9 * stride] *= make_COMPLEX( ANG_2_8, ANG_6_8);
	data[10 * stride] *= make_COMPLEX( ANG_4_8, ANG_4_8);
	data[11 * stride] *= make_COMPLEX( ANG_6_8, ANG_2_8);

	data[13 * stride] *= make_COMPLEX( ANG_6_8, ANG_2_8);
	data[14 * stride] *= make_COMPLEX(-ANG_4_8, ANG_4_8);
	data[15 * stride] *= make_COMPLEX(-ANG_2_8,-ANG_6_8);

	butterfly< 4,-1>(data + 0 * stride, 1 * stride);
	butterfly< 4,-1>(data + 4 * stride, 1 * stride);
	butterfly< 4,-1>(data + 8 * stride, 1 * stride);
	butterfly< 4,-1>(data +12 * stride, 1 * stride);
}

// Only for CUDA Cap 3.5, up to 256 registers without spilling 
template<> inline __device__ void
butterfly<32, 1>(COMPLEX* data, int stride)
{ 
	butterfly<16, 1>(data + 0 * stride, 2 * stride);
	butterfly<16, 1>(data + 1 * stride, 2 * stride);

	COMPLEX t3 = data[3 * stride];
	data[ 3 * stride]  = make_COMPLEX(    t3.y,-   t3.x);
	data[ 5 * stride] *= make_COMPLEX( ANG_4_8,-ANG_4_8);
	data[ 7 * stride] *= make_COMPLEX(-ANG_4_8,-ANG_4_8);
	data[ 9 * stride] *= make_COMPLEX( ANG_2_8,-ANG_6_8);
	data[11 * stride] *= make_COMPLEX(-ANG_6_8,-ANG_2_8);
	data[13 * stride] *= make_COMPLEX( ANG_6_8,-ANG_2_8);
	data[15 * stride] *= make_COMPLEX(-ANG_2_8,-ANG_6_8);
	data[17 * stride] *= make_COMPLEX( ANG_1_8,-ANG_7_8);
	data[19 * stride] *= make_COMPLEX(-ANG_7_8,-ANG_1_8);
	data[21 * stride] *= make_COMPLEX( ANG_5_8,-ANG_3_8);
	data[23 * stride] *= make_COMPLEX(-ANG_3_8,-ANG_5_8);
	data[25 * stride] *= make_COMPLEX( ANG_3_8,-ANG_5_8);
	data[27 * stride] *= make_COMPLEX(-ANG_5_8,-ANG_3_8);
	data[29 * stride] *= make_COMPLEX( ANG_7_8,-ANG_1_8);
	data[31 * stride] *= make_COMPLEX(-ANG_1_8,-ANG_7_8);

	butterfly< 2, 1>(data + 0 * stride, 1 * stride);
	butterfly< 2, 1>(data + 2 * stride, 1 * stride);
	butterfly< 2, 1>(data + 4 * stride, 1 * stride);
	butterfly< 2, 1>(data + 6 * stride, 1 * stride);
	butterfly< 2, 1>(data + 8 * stride, 1 * stride);
	butterfly< 2, 1>(data +10 * stride, 1 * stride);
	butterfly< 2, 1>(data +12 * stride, 1 * stride);
	butterfly< 2, 1>(data +14 * stride, 1 * stride);
	butterfly< 2, 1>(data +16 * stride, 1 * stride);
	butterfly< 2, 1>(data +18 * stride, 1 * stride);
	butterfly< 2, 1>(data +20 * stride, 1 * stride);
	butterfly< 2, 1>(data +22 * stride, 1 * stride);
	butterfly< 2, 1>(data +24 * stride, 1 * stride);
	butterfly< 2, 1>(data +26 * stride, 1 * stride);
	butterfly< 2, 1>(data +28 * stride, 1 * stride);
	butterfly< 2, 1>(data +30 * stride, 1 * stride);
}

// Only for CUDA Cap 3.5, up to 256 registers without spilling 
template<> inline __device__ void
butterfly<32,-1>(COMPLEX* data, int stride)
{ 
	butterfly<16,-1>(data + 0 * stride, 2 * stride);
	butterfly<16,-1>(data + 1 * stride, 2 * stride);

	COMPLEX t3 = data[3 * stride];
	data[ 3 * stride]  = make_COMPLEX(-   t3.y,    t3.x);
	data[ 5 * stride] *= make_COMPLEX( ANG_4_8, ANG_4_8);
	data[ 7 * stride] *= make_COMPLEX(-ANG_4_8, ANG_4_8);
	data[ 9 * stride] *= make_COMPLEX( ANG_2_8, ANG_6_8);
	data[11 * stride] *= make_COMPLEX(-ANG_6_8, ANG_2_8);
	data[13 * stride] *= make_COMPLEX( ANG_6_8, ANG_2_8);
	data[15 * stride] *= make_COMPLEX(-ANG_2_8, ANG_6_8);
	data[17 * stride] *= make_COMPLEX( ANG_1_8, ANG_7_8);
	data[19 * stride] *= make_COMPLEX(-ANG_7_8, ANG_1_8);
	data[21 * stride] *= make_COMPLEX( ANG_5_8, ANG_3_8);
	data[23 * stride] *= make_COMPLEX(-ANG_3_8, ANG_5_8);
	data[25 * stride] *= make_COMPLEX( ANG_3_8, ANG_5_8);
	data[27 * stride] *= make_COMPLEX(-ANG_5_8, ANG_3_8);
	data[29 * stride] *= make_COMPLEX( ANG_7_8, ANG_1_8);
	data[31 * stride] *= make_COMPLEX(-ANG_1_8, ANG_7_8);

	butterfly< 2,-1>(data + 0 * stride, 1 * stride);
	butterfly< 2,-1>(data + 2 * stride, 1 * stride);
	butterfly< 2,-1>(data + 4 * stride, 1 * stride);
	butterfly< 2,-1>(data + 6 * stride, 1 * stride);
	butterfly< 2,-1>(data + 8 * stride, 1 * stride);
	butterfly< 2,-1>(data +10 * stride, 1 * stride);
	butterfly< 2,-1>(data +12 * stride, 1 * stride);
	butterfly< 2,-1>(data +14 * stride, 1 * stride);
	butterfly< 2,-1>(data +16 * stride, 1 * stride);
	butterfly< 2,-1>(data +18 * stride, 1 * stride);
	butterfly< 2,-1>(data +20 * stride, 1 * stride);
	butterfly< 2,-1>(data +22 * stride, 1 * stride);
	butterfly< 2,-1>(data +24 * stride, 1 * stride);
	butterfly< 2,-1>(data +26 * stride, 1 * stride);
	butterfly< 2,-1>(data +28 * stride, 1 * stride);
	butterfly< 2,-1>(data +30 * stride, 1 * stride);
}


//---- Real Butterfly Operators ------------------------------------------

// Version sin angulo (multiplicar 0.5f*v1 si DIR<0)
template<int DIR> subkernel void
rbutterfly(COMPLEX &v1, COMPLEX &v2) {
	v1 = make_COMPLEX(v1.x + v1.y, v1.x - v1.y);
	// if(DIR < 0) v1 = v1 * 0.5f;
	v2 = conj(v2);
}

// Version con angulo (usar ang negativo si DIR<0)
template<int DIR> subkernel void
rbutterfly(COMPLEX &v1, COMPLEX &v2, const COMPLEX &ang) {
	v2 = conj(v2);
	COMPLEX r1 = v1 + v2;
	COMPLEX r2 = v1 - v2;
	COMPLEX ang_r2 = ang * r2;
	// if(DIR < 0) ang_r2 = -ang_r2;
	v1 = 0.5f *     (r1 - ang_r2);
	v2 = 0.5f * conj(r1 + ang_r2);
}

// Version con factor
template<int DIR> subkernel void
rbutterfly(COMPLEX &v1, COMPLEX &v2, const float &frac) {
	if(frac > 1e-5f) {
		COMPLEX tmp;
		sincosf(frac, &tmp.x, &tmp.y);
		if(DIR < 0) tmp.y = -tmp.y;
		rbutterfly<DIR>(v1, v2, tmp);
	} else {
		rbutterfly<DIR>(v1, v2);
		if(DIR < 0) v1 *= 0.5f;
	}
}



// Version para un vector de dimension conocida
template<int N, int DIR> inline __device__ void
	rbutterfly(COMPLEX* sOut) {   //! Llamar unicamente las especializaciones
		while(1);                 //! bucle infinito para detectar su uso
}


template<> inline __device__ void
rbutterfly< 2, 1>(COMPLEX* sOut) {
	rbutterfly< 1>(sOut[0], sOut[1]);
}

template<> inline __device__ void
rbutterfly< 4, 1>(COMPLEX* sOut) {
	rbutterfly< 1>(sOut[0], sOut[2]);
	rbutterfly< 1>(sOut[1], sOut[3], make_COMPLEX(0.70710678f, 0.70710678f));
}

template<> inline __device__ void
rbutterfly< 8, 1>(COMPLEX* sOut) {
	rbutterfly< 1>(sOut[0], sOut[4]);
	rbutterfly< 1>(sOut[1], sOut[7], make_COMPLEX(0.38268343f, 0.92387953f));
	rbutterfly< 1>(sOut[2], sOut[6], make_COMPLEX(0.70710678f, 0.70710678f));
	rbutterfly< 1>(sOut[3], sOut[5], make_COMPLEX(0.92387953f, 0.38268343f));
}

template<> inline __device__ void
rbutterfly<16, 1>(COMPLEX* sOut) {
	rbutterfly< 1>(sOut[0], sOut[ 8]);
	rbutterfly< 1>(sOut[1], sOut[15], make_COMPLEX(0.19509032f, 0.98078528f));
	rbutterfly< 1>(sOut[2], sOut[14], make_COMPLEX(0.38268343f, 0.92387953f));
	rbutterfly< 1>(sOut[3], sOut[13], make_COMPLEX(0.55557023f, 0.83146961f));
	rbutterfly< 1>(sOut[4], sOut[12], make_COMPLEX(0.70710678f, 0.70710678f));
	rbutterfly< 1>(sOut[5], sOut[11], make_COMPLEX(0.83146961f, 0.55557023f));
	rbutterfly< 1>(sOut[6], sOut[10], make_COMPLEX(0.92387953f, 0.38268343f));
	rbutterfly< 1>(sOut[7], sOut[ 9], make_COMPLEX(0.98078528f, 0.19509032f));
}


template<> inline __device__ void
rbutterfly< 2,-1>(COMPLEX* sOut) {
	rbutterfly<-1>(sOut[0], sOut[1]); sOut[0] *= 0.5;
}

template<> inline __device__ void
rbutterfly< 4,-1>(COMPLEX* sOut) {
	rbutterfly<-1>(sOut[0], sOut[2]); sOut[0] *= 0.5;
	rbutterfly<-1>(sOut[1], sOut[3], make_COMPLEX(0.70710678f,-0.70710678f));
}

template<> inline __device__ void
rbutterfly< 8,-1>(COMPLEX* sOut) {
	rbutterfly<-1>(sOut[0], sOut[4]); sOut[0] *= 0.5;
	rbutterfly<-1>(sOut[1], sOut[7], make_COMPLEX( 0.38268343f,-0.92387953f));
	rbutterfly<-1>(sOut[2], sOut[6], make_COMPLEX( 0.70710678f,-0.70710678f));
	rbutterfly<-1>(sOut[3], sOut[5], make_COMPLEX( 0.92387953f,-0.38268343f));
}

template<> inline __device__ void
rbutterfly<16,-1>(COMPLEX* sOut) {
	rbutterfly<-1>(sOut[0], sOut[ 8]); sOut[0] *= 0.5;
	rbutterfly<-1>(sOut[1], sOut[15], make_COMPLEX( 0.19509032f,-0.98078528f));
	rbutterfly<-1>(sOut[2], sOut[14], make_COMPLEX( 0.38268343f,-0.92387953f));
	rbutterfly<-1>(sOut[3], sOut[13], make_COMPLEX( 0.55557023f,-0.83146961f));
	rbutterfly<-1>(sOut[4], sOut[12], make_COMPLEX( 0.70710678f,-0.70710678f));
	rbutterfly<-1>(sOut[5], sOut[11], make_COMPLEX( 0.83146961f,-0.55557023f));
	rbutterfly<-1>(sOut[6], sOut[10], make_COMPLEX( 0.92387953f,-0.38268343f));
	rbutterfly<-1>(sOut[7], sOut[ 9], make_COMPLEX( 0.98078528f,-0.19509032f));
}


#endif // _OP_BFLY

