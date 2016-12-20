//
// Created by tjb3 on 11/23/15.
//

#include "PCIAMTask.h"
#include <fftw-stitching.h>

void PCIAMTask::initialize(int pipelineId,
                           int numPipeline) {
}

void PCIAMTask::shutdown() {
  delete this->memory;
}

void PCIAMTask::executeTask(std::shared_ptr<PCIAMData> data) {
  ImageStitching::FFTWImageTile *origin = data->getOrigin()->getTile();
  ImageStitching::FFTWImageTile *neighbor = data->getNeighbor()->getTile();

  PCIAMDirection direction = data->getDirection();

  ImageStitching::CorrelationTriple triple = phaseCorrelationImageAlignmentFFTW(neighbor, origin, memory);


  switch (direction) {
    case PCIAM_DIRECTION_NORTH:
      origin->setNorthTranslation(triple);
//            std::cout << "North: " << origin->getFilename() << "-> " << neighbor->getFilename() << " " << triple << std::endl;
      break;
    case PCIAM_DIRECTION_WEST:
      origin->setWestTranslation(triple);
//            std::cout << "West: " << std::string(origin->getFilename()) << "-> " << std::string(neighbor->getFilename()) << " " << triple << std::endl;
      break;
  }

  // release memory
  if (this->hasMemReleaser("read")) {
    this->memRelease("read", data->getOrigin()->getReadMemory());
    this->memRelease("read", data->getNeighbor()->getReadMemory());
  }

  if (this->hasMemReleaser("fft")) {
    this->memRelease("fft", data->getOrigin()->getFFTMemory());
    this->memRelease("fft", data->getNeighbor()->getFFTMemory());
  }

}

std::string PCIAMTask::getName() {
  return "PCIAM Task";
}

htgs::ITask<PCIAMData, PCIAMData> *PCIAMTask::copy() {
  return new PCIAMTask(this->getNumThreads(), this->tile);
}

bool PCIAMTask::isTerminated(std::shared_ptr<htgs::BaseConnector> inputConnector) {
  return inputConnector->isInputTerminated();
}
