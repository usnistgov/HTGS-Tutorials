//
// Created by tjb3 on 12/3/15.
//

#include <util-stitching.h>
#include "CCFTask.h"

void CCFTask::initialize(int pipelineId, int numPipeline) {
}

void CCFTask::shutdown() {
}

void CCFTask::executeTask(std::shared_ptr<CCFData> data) {
  this->multiCcfs.clear();

  int *indices = data->getIndices();
  std::shared_ptr<FFTData> origin = data->getOrigin();
  std::shared_ptr<FFTData> neighbor = data->getNeighbor();

  ImageStitching::CUDAImageTile *originTile = origin->getTile();
  ImageStitching::CUDAImageTile *neighborTile = neighbor->getTile();

  for (int i = 0; i < NUM_PEAKS; i++) {
    int x = indices[i] % originTile->getWidth();
    int y = indices[i] / originTile->getWidth();

    if (data->getDirection() == PCIAM_DIRECTION_NORTH) {
      multiCcfs.push_back(peakCrossCorrelationUD(neighborTile, originTile, x, y));
    } else if (data->getDirection() == PCIAM_DIRECTION_WEST) {
      multiCcfs.push_back(peakCrossCorrelationLR(neighborTile, originTile, x, y));
    }
  }

  ImageStitching::CorrelationTriple maxCorr = getMaxCorrelationTriple(multiCcfs);


  if (data->getDirection() == PCIAM_DIRECTION_NORTH) {
    originTile->setNorthTranslation(maxCorr);
  } else if (data->getDirection() == PCIAM_DIRECTION_WEST) {
    originTile->setWestTranslation(maxCorr);
  }

  // release memory
  if (this->hasMemReleaser("read")) {
    this->memRelease("read", data->getOrigin()->getReadData());
    this->memRelease("read", data->getNeighbor()->getReadData());
  }

}

std::string CCFTask::getName() {
  return "CCF Task";
}

htgs::ITask<CCFData, htgs::VoidData> *CCFTask::copy() {
  return new CCFTask(this->getNumThreads());
}

bool CCFTask::isTerminated(std::shared_ptr<htgs::BaseConnector> inputConnector) {
  return inputConnector->isInputTerminated();
}

void CCFTask::debug() {
}
