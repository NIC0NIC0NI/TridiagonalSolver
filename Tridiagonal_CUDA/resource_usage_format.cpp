#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <unordered_map>
#include <cxxabi.h>

const char before_name[] = "Function properties for";
const char before_reg[] = "Used";
const char seperator[] = ", ";
const std::string terms[] = {"registers", "cmem[0]", "cmem[2]", "smem", "stack frame", "spill stores", "spill loads"};

template<typename T, std::size_t N> std::size_t arrlen(T(&)[N]) { return N; }

#define OUTFILE "resource_usage.txt"
#define NAMELEN 26

char *funcname(char *demangled) {
    char *p;
    char *q;
    for(p = demangled; *p != ' '; ++p);
    for(q = ++p; *q != '<'; ++q);
    for(++q; *q != ','; ++q);
    q[0] = '>';
    q[1] = '\0';
    return p;
}

int main() {
    std::string s;
    std::ofstream fout(OUTFILE);
    std::unordered_map<std::string, std::size_t> usage;
    fout << std::setw(NAMELEN) << "name" << seperator;
    for(const auto &t : terms) {
        usage[t] = 0;
        fout << t << seperator;
    }
    fout << std::endl;

    std::ios_base::sync_with_stdio(false);

    while(std::getline(std::cin, s)) {
        if(s.find("ptxas info") != std::string::npos) {
            int dummy, pos1 = s.find(before_name), pos2 = s.find(before_reg);
            if(pos1 != std::string::npos) {
                char *name = abi::__cxa_demangle(s.substr(pos1 + arrlen(before_name)).c_str(), 0, 0, &dummy);
                char *fname = funcname(name);
                fout << std::setw(NAMELEN) << fname << seperator;
                std::free(name);
                for(const auto &t : terms) {
                    usage[t] = 0;
                }

            } else if(pos2 != std::string::npos) {
                int value;
                std::string name, unit;
                std::istringstream is(s.substr(pos2 + arrlen(before_reg)));
                is >> value >> name;
                while(name.back() == ',') {
                    name.pop_back();
                    usage[name] = value;
                    is >> value >> unit >> name;
                }
                usage[name] = value;

                /* OUTPUT */
                fout << std::right;
                for(const auto &t : terms) {
                    fout << std::setw(t.length()) << usage[t] << seperator;
                }
                fout << std::endl;
            }

        } else if(s.find(terms[5]) != std::string::npos) {
            std::istringstream is(s);
            std::string name1, name2, unit;
            int value;
            is >> value >> unit >> name1 >> name2;
            while(name2.back() == ',') {
                name2.pop_back();
                usage[name1.append(" ").append(name2)] = value;
                is >> value >> unit >> name1 >> name2;
            }
            usage[name1.append(" ").append(name2)] = value;
        } else {
            /* other outputs that is not related to resource usage */
            std::cout << s << std::endl;
        }
    }
    return 0;
}


