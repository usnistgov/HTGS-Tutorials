//
// Created by tjb3 on 11/16/15.
//

#ifndef HTGS_FFTWMEMORY_H
#define HTGS_FFTWMEMORY_H

#include <fftw3.h>
#include <stitching-types.h>

#include <htgs/api/IMemoryAllocator.hpp>

class FFTWMemory: public htgs::IMemoryAllocator<fftw_t> {
 public:
  FFTWMemory(size_t size) : IMemoryAllocator(size) { }

  ~FFTWMemory() {

  }

  fftw_t *memAlloc(size_t size) {
#ifdef USE_DOUBLE
    return fftw_alloc_complex(size);
#else
    return fftwf_alloc_complex(size);
#endif
  }

  fftw_t *memAlloc() {
#ifdef USE_DOUBLE
    return fftw_alloc_complex(size());
#else
    return fftwf_alloc_complex(size());
#endif
  }
  void memFree(fftw_t *&memory) {
#ifdef USE_DOUBLE
    fftw_free(memory);
#else
    fftwf_free(memory);
#endif
  }
};

#endif //HTGS_FFTWMEMORY_H
