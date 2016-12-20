//
// Created by tjb3 on 11/17/15.
//

#ifndef HTGS_FFTDATA_H
#define HTGS_FFTDATA_H


#include <fftw-image-tile.h>
#include <htgs/api/IData.hpp>
#include <htgs/api/MemoryData.hpp>

class FFTData: public htgs::IData {

 public:
  FFTData(ImageStitching::FFTWImageTile *tile, int order) : IData(order) {
    this->tile = tile;
    this->readMemory = nullptr;
    this->fftMemory = nullptr;
  }

  ~FFTData() {
  }

  ImageStitching::FFTWImageTile *getTile() const {
    return this->tile;
  }

  std::shared_ptr<htgs::MemoryData<img_t *>> getReadMemory() {
    return this->readMemory;
  }

  std::shared_ptr<htgs::MemoryData<fftw_t *>> getFFTMemory() {
    return this->fftMemory;
  }


  void setReadMemory(std::shared_ptr<htgs::MemoryData<img_t *>> readMemory) {
    this->readMemory = readMemory;
  }

  void setFftMemory(std::shared_ptr<htgs::MemoryData<fftw_t *>> fftMemory) {
    this->fftMemory = fftMemory;
  }

 private:
  ImageStitching::FFTWImageTile *tile;
  std::shared_ptr<htgs::MemoryData<img_t *>> readMemory;
  std::shared_ptr<htgs::MemoryData<fftw_t *>> fftMemory;
};


#endif //HTGS_FFTDATA_H


