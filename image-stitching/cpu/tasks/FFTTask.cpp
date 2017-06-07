//
// Created by tjb3 on 11/18/15.
//

#include <fftw-image-tile.h>
#include <htgs/api/ITask.hpp>

#include "FFTTask.h"
#include "../memory/FFTMemoryRule.h"

void FFTTask::initialize() {
  this->memory = new ImageStitching::FFTWTileWorkerMemory(this->initTile);
}

void FFTTask::shutdown() {
  delete memory;
}

void FFTTask::executeTask(std::shared_ptr<FFTData> data) {
  ImageStitching::FFTWImageTile *tile = data->getTile();

  if (data->getFFTMemory() != nullptr) {
        tile->computeFFT(this->memory->getFFTInP(), data->getFFTMemory()->get());
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

bool FFTTask::canTerminate(std::shared_ptr<htgs::AnyConnector> inputConnector) {
  return inputConnector->isInputTerminated();
}
