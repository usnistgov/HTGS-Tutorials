//
// Created by tjb3 on 11/18/15.
//

#ifndef HTGS_FFTTASK_H
#define HTGS_FFTTASK_H


#include <fftw-image-tile.h>
#include <fftw-worker-memory.h>
#include <htgs/api/ITask.hpp>

#include "../data/FFTData.h"


class FFTTask: public htgs::ITask<FFTData, FFTData> {
 public:
  FFTTask(int numThreads,
          ImageStitching::FFTWImageTile *initTile,
          int startCol,
          int startRow,
          int extentWidth,
          int extentHeight)
      : ITask(numThreads) {
    this->initTile = initTile;
    this->startCol = startCol;
    this->startRow = startRow;
    this->extentWidth = extentWidth;
    this->extentHeight = extentHeight;
  }

  ~FFTTask() {
  }

  virtual void initialize(int pipelineId,
                          int numPipeline) override;

  virtual void shutdown() override;

  virtual void executeTask(std::shared_ptr<FFTData> data) override;

  virtual std::string getName() override;

  virtual htgs::ITask<FFTData, FFTData> *copy() override;

  virtual bool isTerminated(std::shared_ptr<htgs::BaseConnector> inputConnector) override;

 private:
  ImageStitching::FFTWTileWorkerMemory *memory;
  ImageStitching::FFTWImageTile *initTile;
  int startCol;
  int startRow;
  int extentWidth;
  int extentHeight;
  int pipelineId;
};


#endif //HTGS_FFTTASK_H
