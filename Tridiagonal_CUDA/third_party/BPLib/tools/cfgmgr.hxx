//- =======================================================================
//+ Gestor de configuraciones v2.0
//- =======================================================================

#pragma once
#ifndef CFGMGR_HXX
#define CFGMGR_HXX

//---- Header Dependencies -----------------------------------------------
#include <map>
#include <cstring>
#include <cstdio>

//---- Class Declaration -------------------------------------------------

class ConfigManager {

public:

	// Constructor
	ConfigManager(char* cfgdata = NULL);
	
	// Intenta cargar la configuracion desde el string o fichero
	int load(char* str);

	// Intenta cargar la configuracion desde la linea de comandos
	int load(int argc, char* argv[]);

	// Intenta cargar la configuracion desde fichero o stdin
	int load(FILE* fIn = 0);

	// Guarda la configuracion actual al fichero especificado
	int save(const char* cfgdata = NULL) const;

	// Imprime la configuracion actual por pantalla
	void print(const char* name = NULL) const;

	// Establece el dominio actual
	void setDomain(const char* domain = NULL);

	// Establece el dato con la etiqueta especificada
	inline const char* set(const char* key, const char* val = ""  ) {
		return set("%s", key,  val); }
	inline const char* set(const char* key, const int   val = 0   ) {
		return set("%i", key, &val); }
	inline const char* set(const char* key, const float val = 0.0f) {
		return set("%f", key, &val); }

	// Devuelve el dato con la etiqueta especificada (notFound: NULL)
	inline const char*  get(const char* key, const char*  val) {
		return (const  char*) get("%s", key, val); }
	inline const int*   get(const char* key, int&         val) {
		return (const   int*) get("%i", key, &val); }
	inline const float* get(const char* key, float&       val) {
		return (const float*) get("%f", key, &val); }

	// Devuelve el dato con la etiqueta especificada (notFound: default)
	inline const char* getChar (const char* key) { char* val = NULL;
		get("%s", key,  val); return val; }
	inline int         getInt  (const char* key) { int   val = 0;
		get("%i", key, &val); return val; }
	inline float       getFloat(const char* key) { float val = 0.0f;
		get("%f", key, &val); return val; }

private:

	// Comparador de texto para el mapa (case sensitive)
	struct classcomp { bool operator() (const char *s1, const char* s2) const
		{ return strcmp(s1, s2) < 0; }
	};

	// Estructura mapa con los datos de la configuracion
	std::map<const char*, const char*, classcomp> _config;

	// Dominio actual en el que se insertan los datos
	char _domain[16];

protected:

	// Metodo auxiliar para establecer un valor con la clave especificada
	const char* set(const char* format, const char* key, const void* val);

	// Metodo auxiliar para obtener el valor de la clave especificada
	const void* get(const char* format, const char* key, const void* val);

	// Busca la clave especificada y devuelve NULL si no la encuentra
	const char* search(const char* key, const char* domain = NULL);

	// Metodo auxiliar que establece una clave-valor desde un string
	int readline(char* line);

};

#endif // CFGMGR_HXX

