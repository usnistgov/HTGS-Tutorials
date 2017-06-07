//
// Created by tjb3 on 11/16/15.
//


#ifndef HTGS_READMEMORY_H
#define HTGS_READMEMORY_H

#include <htgs/api/IMemoryAllocator.hpp>

class ReadMemory: public htgs::IMemoryAllocator<img_t> {
 public:
  ReadMemory(size_t size) : IMemoryAllocator(size) { }

  ~ReadMemory() { }

  img_t *memAlloc(size_t size) {
    return (img_t *) malloc(sizeof(img_t) * size);
  }

  img_t *memAlloc() {
    return (img_t *) malloc(sizeof(img_t) * size());
  }
  void memFree(img_t *&memory) { free(memory); }
};

#endif //HTGS_READMEMORY_H
