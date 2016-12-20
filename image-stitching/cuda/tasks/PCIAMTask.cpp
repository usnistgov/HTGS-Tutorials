//
// Created by tjb3 on 11/23/15.
//

#include "PCIAMTask.h"
#include <cuda-stitching.h>
#include <util-stitching.h>

std::string PCIAMTask::getName() {
  return "PCIAM Task";
}

htgs::ITask<PCIAMData, CCFData> *PCIAMTask::copy() {
  return new PCIAMTask(this->getContexts(), this->getCudaIds(), this->getNumGPUs(), this->tile);
}

bool PCIAMTask::isTerminated(std::shared_ptr<htgs::BaseConnector> inputConnector) {
  return inputConnector->isInputTerminated();
}

void PCIAMTask::initializeCudaGPU(CUcontext context, CUstream stream, int cudaId, int numGPUs, int pipelineId,
                                  int numPipelines) {
  this->memory = new ImageStitching::CUDATileWorkerMemory(this->tile);
//    cudaMalloc((void **)&pcmMemory, sizeof(double) * this->tile->getSize());
  cudaMalloc((void **) &originDevMem, sizeof(cuda_t) * ImageStitching::CUDAImageTile::fftSize);
  cudaMalloc((void **) &neighborDevMem, sizeof(cuda_t) * ImageStitching::CUDAImageTile::fftSize);

  ImageStitching::CUDAImageTile::bindBwdPlanToStream(stream, pipelineId);

  this->pipelineId = pipelineId;
  this->gpuId = cudaId;

}

void PCIAMTask::executeGPUTask(std::shared_ptr<PCIAMData> data, CUstream stream) {

  std::shared_ptr<FFTData> originData = data->getOrigin();
  std::shared_ptr<FFTData> neighborData = data->getNeighbor();
  ImageStitching::CUDAImageTile *origin = originData->getTile();
  ImageStitching::CUDAImageTile *neighbor = neighborData->getTile();

  PCIAMDirection direction = data->getDirection();

  DEBUG("PCIAMTask: Computing PCIAM between " << origin->getFilename() << " and " << neighbor->getFilename()
            << " direction: " << direction);

//    ImageStitching::CorrelationTriple triple = phaseCorrelationImageAlignmentFFTW(neighbor, origin, memory);

  cuda_t *originMemory;
  cuda_t *neighborMemory;
  real_t *pcmMemory = this->memory->getPcm();

  if (this->autoCopy(this->originDevMem, originData->getFftData(), ImageStitching::CUDAImageTile::fftSize)) {
    originMemory = this->originDevMem;
  }
  else {
    originMemory = originData->getFftData()->get();
  }

  if (this->autoCopy(this->neighborDevMem, neighborData->getFftData(), ImageStitching::CUDAImageTile::fftSize)) {
    neighborMemory = this->neighborDevMem;
  }
  else {
    neighborMemory = neighborData->getFftData()->get();
  }

  peakCorrelationMatrix(originMemory, neighborMemory, pcmMemory, this->memory, stream, this->pipelineId);
  int *indices = multiPeakCorrelationMatrixIndices(pcmMemory,
                                                   NUM_PEAKS,
                                                   origin->getWidth(),
                                                   origin->getHeight(),
                                                   this->memory,
                                                   stream);

  CCFData *ccfData = new CCFData(indices, data);
  addResult(ccfData);

  if (this->hasMemReleaser("fft")) {
    this->memRelease("fft", data->getOrigin()->getFftData());
    this->memRelease("fft", data->getNeighbor()->getFftData());
  }

}

void PCIAMTask::shutdownCuda() {
}

void PCIAMTask::debug() {
}
