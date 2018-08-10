//
// Created by tjb3 on 11/18/15.
//


#include "FFTTask.h"


void FFTTask::initializeCudaGPU() {
  this->memory = new ImageStitching::CUDATileWorkerMemory(this->initTile);
  ImageStitching::CUDAImageTile::bindFwdPlanToStream(this->getStream(), this->getPipelineId());
}

void FFTTask::shutdownCuda() {
  delete memory;
}

std::string FFTTask::getName() {
  return "FFT Task";
}

htgs::ITask<FFTData, FFTData> *FFTTask::copy() {
  return new FFTTask(this->getCudaIds(),
                     this->getNumGPUs(),
                     this->initTile,
                     this->startCol,
                     this->startRow,
                     this->extentWidth,
                     this->extentHeight);
}

void FFTTask::executeTask(std::shared_ptr<FFTData> data) {
  ImageStitching::CUDAImageTile *tile = data->getTile();

  tile->setDev(this->getCudaId());

  if (data->getFftData() != nullptr) {
    tile->computeFFT(data->getFftData()->get(), this->memory, this->getStream(), true);
  }
  else {
    tile->computeFFT();
  }

  this->addResult(data);
}

