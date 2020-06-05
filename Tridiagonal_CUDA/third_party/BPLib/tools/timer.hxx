//- =======================================================================
//+ Timer class v3.1
//- =======================================================================

#pragma once
#ifndef TIMER_HXX
#define TIMER_HXX

//---- Class Declaration -------------------------------------------------

// Clase timer para tomar tiempos de ejecucion
class Timer {

	// Estado del objeto, contadores inicial y final
	double ini, end;

	// Funcion clock compatible con multiples sistemas
	double getSecs() const;

public:

	// Constructor de clase, inicia el timer
	Timer();

	// Inicia el timer, devuelve tiempo entre llamadas sucesivas
	double start();

	// Devuelve el tiempo desde el inicio
	double time() const;

	// Detiene el temporizador y devuelve el tiempo en segundos
	double stop();

	// Imprime el paso de tiempo con la etiqueta especificada
	double print(const char* text = 0);

};

#endif // TIMER_HXX
