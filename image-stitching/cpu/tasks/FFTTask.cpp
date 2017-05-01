//
// Created by tjb3 on 11/18/15.
//

#include <fftw-image-tile.h>

#include "FFTTask.h"
#include "../memory/FFTMemoryRule.h"

void FFTTask::initialize(int pipelineId,
                         int numPipeline) {
  this->memory = new ImageStitching::FFTWTileWorkerMemory(this->initTile);
  this->pipelineId = pipelineId;
}

void FFTTask::shutdown() {
  delete memory;
}

void FFTTask::executeTask(std::shared_ptr<FFTData> data) {
  ImageStitching::FFTWImageTile *tile = data->getTile();

  DEBUG("FFTTask: " << this->pipelineId << " Computing FFT for " << tile->getFilename());
  if (data->getFFTMemory() != nullptr) {
        tile->computeFFT(this->memory->getFFTInP(), data->getFFTMemory()->get());
//    tile->computeFFT();
  }
  else {
    tile->computeFFT();
  }

  this->addResult(data);
}

std::string FFTTask::getName() {
  return "FFT Task";
}

htgs::ITask<FFTData, FFTData> *FFTTask::copy() {
  return new FFTTask(this->getNumThreads(),
                     this->initTile,
                     this->startCol,
                     this->startRow,
                     this->extentWidth,
                     this->extentHeight);
}

bool FFTTask::isTerminated(std::shared_ptr<htgs::BaseConnector> inputConnector) {
  return inputConnector->isInputTerminated();
}
