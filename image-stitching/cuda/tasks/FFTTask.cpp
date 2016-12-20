//
// Created by tjb3 on 11/18/15.
//


#include "FFTTask.h"


void FFTTask::initializeCudaGPU(CUcontext context, CUstream stream, int cudaId, int numGPUs, int pipelineId,
                                int numPipelines) {
  this->memory = new ImageStitching::CUDATileWorkerMemory(this->initTile);
  this->pipelineId = pipelineId;
  ImageStitching::CUDAImageTile::bindFwdPlanToStream(stream, pipelineId);
}

void FFTTask::shutdownCuda() {
  delete memory;
}

std::string FFTTask::getName() {
  return "FFT Task";
}

htgs::ITask<FFTData, FFTData> *FFTTask::copy() {
  return new FFTTask(this->getContexts(),
                     this->getCudaIds(),
                     this->getNumGPUs(),
                     this->initTile,
                     this->startCol,
                     this->startRow,
                     this->extentWidth,
                     this->extentHeight);
}

bool FFTTask::isTerminated(std::shared_ptr<htgs::BaseConnector> inputConnector) {
  return inputConnector->isInputTerminated();
}


void FFTTask::executeGPUTask(std::shared_ptr<FFTData> data, CUstream stream) {
  ImageStitching::CUDAImageTile *tile = data->getTile();

  tile->setDev(this->getCudaId());

  DEBUG("FFTTask: " << this->pipelineId << " Computing FFT for " << tile->getFilename());
  if (data->getFftData() != nullptr) {
    tile->computeFFT(data->getFftData()->get(), this->memory, stream, true);
  }
  else {
    tile->computeFFT();
  }

  this->addResult(data);
}

