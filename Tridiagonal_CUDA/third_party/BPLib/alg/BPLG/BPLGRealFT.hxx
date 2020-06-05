//- =======================================================================
//+ BPLG Real Fourier Transform v1.0
//- =======================================================================

#pragma once
#ifndef BPLG_REALFT_HXX
#define BPLG_REALFT_HXX

//---- Header Dependencies -----------------------------------------------

#include "../Transform.hxx"
#include "tools/cuvector.hxx"

//---- Class Declaration -------------------------------------------------

class BPLGRealFT : public Transform {

public:

	// Crea una transformada, el batch es la ultima dimension
	BPLGRealFT(int dimX, int dimY = 1, int dimZ = 1) :
	  Transform(dimX, dimY, dimZ) { swap = 0; init(); }

	// Libera la memoria reservada para el vector temporal
	~BPLGRealFT() { if(swap) delete swap; }

	// Calcula la transformada, directa (dir=1) o inversa (dir=-1)
	Transform* calc(int dir = 1);

	// Inicializa los datos, secuencial (mode=0) o aleatorio (mode=1)
	Transform* init(int mode = -1);

	// Devuelve el nombre del algoritmo
	inline const char* toString() { return "BPLGRealFT"; }

	// Imprime los datos por pantalla, etiqueta con nombre opcional
	Transform* print(const char* name = 0);

	// Compara el error relativo con respecto a otro algoritmo
	double compare(Transform* ref, const char* label = 0);

	// Devuelve el numero de gflops del metodo
	inline double gflops(double time, long long iters = 1) const {
		return gflops_r(time, iters); }

protected:

	// Vector que contiene los datos temporales
	CUVector<Complex>* swap;

	// Actualizar vector complejo
	void updateC(int dir);

	//Actualizar vector real
	void updateR(int dir);

	// Ultima direccion: +1 -> datos complejos, -1 -> datos reales
	int last_dir;

};

#endif // BPLG_REALFT_HXX
