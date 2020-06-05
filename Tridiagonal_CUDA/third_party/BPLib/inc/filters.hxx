//- =======================================================================
//+ Data filters for signal transforms
//- =======================================================================

#pragma once
#ifndef FILTERS_HXX
#define FILTERS_HXX

//---- Header Dependencies -----------------------------------------------
#include <algorithm>

//---- Library Header Definitions ----------------------------------------


// Performing a BitRev of a data vector[0..N-1]
template<typename T> void
vBitRev(T* data, int N, int stride = 1) {
	for(int posA = 1; posA < N; posA++) { 
		int posB = 0;
		for(int i = 1; i < N; i<<=1) // 0001, 0010, 0100, 1000
			posB = (posB << 1) | ((i & posA) != 0);
		if(posA > posB)
			std::swap(data[posA * stride], data[posB * stride]);
	}
}

//Inversing a data vector[1..N-1]
template<typename T> void
vDataRev(T* data, int N, int stride = 1) {
	for(int posA = 1; posA < N; posA++) { // Keeps first
		int posB = N - posA;
		if(posA >= posB) break;
		std::swap(data[posA * stride], data[posB * stride]);
	}
}

// Implementacion básica que escala usando un factor el vector data[1..N-1]
template<typename T1, typename T2> void
vDataScale(T1* data, int N, T2 factor, int stride = 1) {
	for(int posA = 0; posA < N; posA++)
		data[posA * stride] = data[posA * stride] * factor;
}

// Implementacion básica que escala usando un factor el vector data[1..N-1]
template<typename T1, typename T2> inline void
vDataScale(T1* data, int N, int batch, T2 factor) {
	vDataScale(data, N * batch, factor);
}

// Empaqueta un vector de datos N en otro N/2+1
template<typename T> void
vDataCpack(T* sOut, T* sIn, int N, int batch = 1) {
	const int halfN1 = N / 2 + 1;
	for(int j = 0; j < batch; j++)
		for(int i = 0; i < halfN1; i++)
			sOut[j * halfN1 + i] = sIn[j * N + i];
}

// Empaqueta un vector de datos N/2+1 en otro N/2
template<typename T> void
vDataCunpack(T* sOut, T* sIn, int N, int batch = 1) {
	const int halfN1 = N / 2 + 1;
	for(int j = 0; j < batch; j++) {
		for(int i = 0; i < halfN1; i++)
			sOut[j * N + i] = sIn[j * halfN1 + i];
		for(int i = halfN1; i < N; i++)
			sOut[j * N + i] = std::conj(sIn[j * halfN1 + (N - i)]);
	}
}

#endif // FILTERS_HXX


