//
// Created by tjb3 on 12/2/15.
//

#ifndef HTGS_FFTDATA_H
#define HTGS_FFTDATA_H

#include <cuda-image-tile.h>
#include <htgs/api/MemoryData.hpp>

class FFTData: public htgs::IData {

 public:

  FFTData(ImageStitching::CUDAImageTile *tile, int order) : IData(order) {
    this->tile = tile;
  }

  ~FFTData() { }

  ImageStitching::CUDAImageTile *getTile() const {
    return tile;
  }

  std::shared_ptr<htgs::MemoryData<img_t *>> getReadData() const {
    return readData;
  }

  void setReadData(std::shared_ptr<htgs::MemoryData<img_t *>> readData) {
    FFTData::readData = readData;
  }

  std::shared_ptr<htgs::MemoryData<cuda_t *>> getFftData() const {
    return fftData;
  }

  void setFftData(std::shared_ptr<htgs::MemoryData<cuda_t *>> fftData) {
    FFTData::fftData = fftData;
  }

 private:
  ImageStitching::CUDAImageTile *tile;
  std::shared_ptr<htgs::MemoryData<img_t *>> readData;
  std::shared_ptr<htgs::MemoryData<cuda_t *>> fftData;

};

#endif //HTGS_FFTDATA_H
