//- =======================================================================
//+ Configuration Manager
//- =======================================================================

//---- Header Dependencies -----------------------------------------------
#include "cfgmgr.hxx"
#include <cstdio>
#include <cstdlib>

//---- MSVC/GCC compatibility --------------------------------------------
#if defined _WIN32 || _WIN64
	#define strdup(_str) _strdup(_str)
	#define strncpy(_dst, _src, _len) strcpy_s(_dst, _len, _src)
	#define fopen(_fname, _fmode) _fopen((char*)_fname, _fmode)
	#define strtok_r(_str, _tok, _ctx) strtok_s(_str, _tok, _ctx)
	#define snprintf(_buf, _dim, ...) sprintf_s(_buf, _dim, __VA_ARGS__)
	#define sscanf(_buf, _dim, ...) sscanf_s(_buf, _dim, __VA_ARGS__)

	FILE* _fopen(char* fname, char *fmode) {
		FILE *f1; 
		int errCode = fopen_s(&f1, fname, fmode);
		return errCode ? NULL : f1;
	}
#endif

//---- Library Header Definitions ----------------------------------------

ConfigManager::ConfigManager(char* fname) {
	_domain[0] = '\0';
	if(load(fname) < 0 && fname) 
		printf("Aviso, no se pudo leer '%s'", fname);
}

int
ConfigManager::readline(char* line) {
	if(!line) return -1;
	if(line[0] == '\n' || line[0] == '\r') return 2;

	// Read the first token
	char *context = NULL;
	char* token1 = strtok_r(line, " \n\r\t=;]", &context);
	if(!token1) return 0;
	switch(token1[0]) {
		case '\0': case '/': case '!': case '%': return 2;
		case '[' : strncpy(_domain, token1 + 1, sizeof(_domain)); return 4;
	}

	// Read the second token
	char* token2 = strtok_r(NULL, " \n\r\t=%", &context);
	if(!token2) return 0;

	
	return set(token1, token2) ? 1 : 0;
}

void
ConfigManager::setDomain(const char* domain) {
	if(domain == NULL) _domain[0] = '\0'; // By default domain = ""
	else strncpy(_domain, domain, sizeof(_domain));
}

int
ConfigManager::load(int argc, char* argv[]) {
	if(argc == 1 || !argv) return 0;
	int preserveDomain = _domain[0];
	int cont = 0;
	for(int i = 1; i < argc; i++)
		if(readline(argv[i])) cont++;
	if(!preserveDomain) setDomain();
	return cont;
}

int
ConfigManager::load(FILE* fIn) {
	if(!fIn) fIn = stdin;
	char line[96];
	int cont;
	for(cont = 0; !feof(fIn); cont++) {
		fgets(line, 90, fIn);
		if(readline(line) == -1) break;
	}
	setDomain(); // Dominio por defecto
	return cont;
}


int
ConfigManager::load(char* str) {
	if(!str) return 0;
	int cont = 0, next = 0;

	// Intentamos cargar de string
	for(char* line = str; line[0]; line+=next) {
		for(next = 0; line[next]!='\0'; next++)
			if(line[next]=='\n') { next++; break; }
		if(readline(line)) cont++; else break;
	}
	
	// Intentamos cargar de fichero
	if(cont == 0) {
		FILE* fIn = fopen(str, "r");
		if(!fIn) return -1;
		cont = load(fIn);
		fclose(fIn);
	}

	setDomain(); // Reseteamos el dominio
	return cont; // Devolvemos el numero de parametros
}

int
ConfigManager::save(const char* fname) const {
	FILE *fOut = fname ? fopen(fname, "w") : stdout;
	if(!fOut) return -1;
	std::map<const char*, const char*, ConfigManager::classcomp>::const_iterator it;
	for(it = _config.begin(); it != _config.end(); it++)
		fprintf(fOut, "%s = %s\n", it->first, it->second);
	if(fOut != stdout) fclose(fOut);
	return (int)_config.size();
}

void
ConfigManager::print(const char* name) const {
	if(!name) name = "<Config>";
	printf("%s[%i], domain: '%s'\n",
		name, (int)_config.size(), _domain);
	save();
	printf("\n");
}

const char*
ConfigManager::set(const char* format, const char* key, const void* val) {
	if(!key || !val) return NULL;

	// Busca si la clave ya existe para liberar la memoria
	const char* value = search(key, _domain);
	if(value) free((char*)value);

	// Creamos la copia de la clave comprobando si ya tiene el dominio
	char* keyCopy = (char*)malloc(32 * sizeof(char));
	if(strchr(key, '.') != NULL) strncpy(keyCopy, key, 32);
	else snprintf(keyCopy, 31, "%s.%s", _domain, key);

	// Creamos la copia del valor con el formato especificado
	char* valCopy = (char*)malloc(32 * sizeof(char));
	snprintf(valCopy, 31, format, val);

	// Asignamos el par clave-valor y devolvemos el puntero
	return _config[keyCopy] = valCopy;
}

const void*
ConfigManager::get(const char* format, const char* key, const void* val) {
	// Buscamos la clave, si no se encuentra retornamos nulo
	const char* data = search(key);
	if(!data) return NULL;

	// Devolvemos el valor encontrado una vez convertido al formato
	if(sscanf(data, format, val)) return val;

	// Si falla la conversion retornamos nulo
	return NULL;
}

const char*
ConfigManager::search(const char* key, const char* domain) {
	if(!key) return NULL;

	// Si no se especifica domain busca en el actual y despues en raiz
	if(!domain) {
		const char* tmp = search(key, _domain);
		if(tmp || _domain[0]=='\0') return tmp;
		domain = "";
	}

	// Si se especifica domain busca solamente dentro del parametro
	char buffer[32];
	snprintf(buffer, sizeof(buffer), "%s.%s", domain, key);
	std::map<const char*, const char*, ConfigManager::classcomp>::const_iterator it;
	it = _config.find(buffer);
	if(it != _config.end()) return it->second;

	// Si no se encontro la clave retornamos nulo
	return NULL;
}
