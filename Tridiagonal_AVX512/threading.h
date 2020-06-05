#ifndef THREADING_H_INCLUDED
#define THREADING_H_INCLUDED

#ifndef THREADING
#define THREADING 1
#endif

#if THREADING == 0 /* single-threaded */
#define PARALLEL_FOR_BEGIN(j, begin, end) for(int j = begin; j < end; ++j) {
#define PARALLEL_FOR_END                  }
#elif THREADING == 1
#define PARALLEL_FOR_BEGIN(j, begin, end) _Pragma("omp parallel for schedule(static)") \
                                          for(int j = begin; j < end; ++j) {
#define PARALLEL_FOR_END                  }
#elif THREADING == 2
#include <tbb/parallel_for.h>
#define PARALLEL_FOR_BEGIN(j, begin, end) tbb::affinity_partitioner partitioner; \
                                          tbb::parallel_for(begin, end, [=](int j) {
#define PARALLEL_FOR_END                  }, partitioner);
#elif THREADING == 3
#include <cilk/cilk.h>
#define PARALLEL_FOR_BEGIN(j, begin, end) cilk_for (int j = begin; j < end; ++j) {
#define PARALLEL_FOR_END                  }
#endif


#endif
