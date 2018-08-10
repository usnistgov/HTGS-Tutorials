//
// Created by tjb3 on 11/18/15.
//

#ifndef HTGS_FFTTASK_H
#define HTGS_FFTTASK_H


#include <htgs/api/ICudaTask.hpp>
#include "../data/FFTData.h"

class FFTTask: public htgs::ICudaTask<FFTData, FFTData> {
 public:
  FFTTask(int *cudaIds, int numGpus, ImageStitching::CUDAImageTile *initTile, int startCol,
          int startRow, int extentWidth, int extentHeight) : ICudaTask(cudaIds, numGpus) {
    this->initTile = initTile;
    this->startCol = startCol;
    this->startRow = startRow;
    this->extentWidth = extentWidth;
    this->extentHeight = extentHeight;
  }

  ~FFTTask() { }

  virtual void initializeCudaGPU() override;

  virtual void executeTask(std::shared_ptr<FFTData> data) override;

  virtual void shutdownCuda() override;

  virtual std::string getName() override;

  virtual htgs::ITask<FFTData, FFTData> *copy() override;

 private:
  ImageStitching::CUDATileWorkerMemory *memory;
  ImageStitching::CUDAImageTile *initTile;
  int startCol;
  int startRow;
  int extentWidth;
  int extentHeight;
};


#endif //HTGS_FFTTASK_H
