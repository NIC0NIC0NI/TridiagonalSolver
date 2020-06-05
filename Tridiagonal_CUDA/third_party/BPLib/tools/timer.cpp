//- =======================================================================
//+ Timer class v3.1
//- =======================================================================

//---- Header File -------------------------------------------------------
#include "timer.hxx"

//---- Include Section ---------------------------------------------------
#include <cstdio>
#include <omp.h>

//---- Library Code ------------------------------------------------------

// Convierte de clock_t a segundos en doble precision
inline double 
Timer::getSecs() const {
	return omp_get_wtime();
	// clock();
}

// Constructor de clase, inicia el timer
Timer::Timer() {
	ini = end = getSecs();
}

// Inicia el timer, devuelve tiempo entre llamadas sucesivas
double
Timer::start() {
	end = ini; 
	ini = getSecs();
	return ini - end; // end <= ini
}

// Devuelve el tiempo desde el inicio
double
Timer::time() const {
	return getSecs() - ini;
}

// Detiene el temporizador y devuelve el tiempo en segundos
double
Timer::stop() {
	if(end <= ini) end = getSecs(); // end >= ini
	return end - ini;
}

// Imprime el paso de tiempo con la etiqueta especificada
double
Timer::print(const char* text) {
	if(!text) text = "timer";
	double retVal = start(); // Modifica el estado
	printf("%s : %.2f sec\n", text, retVal);
	return retVal;
}
