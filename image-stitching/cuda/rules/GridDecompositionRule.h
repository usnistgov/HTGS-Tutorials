//
// Created by tjb3 on 12/1/15.
//

#include <htgs/api/IRule.hpp>
#include <htgs/debug/debug_message.h>

#include "../data/FFTData.h"


#ifndef HTGS_DECOMPOSITIONRULE_H
#define HTGS_DECOMPOSITIONRULE_H

class GridDecompositionRule: public htgs::IRule<FFTData, FFTData> {

 public:
  GridDecompositionRule(int startRow, int startCol, int extentWidth, int extentHeight, int numPipelines) {
    this->startCol = startCol;
    this->startRow = startRow;
    this->extentWidth = extentWidth;
    this->extentHeight = extentHeight;
    this->numPipelines = numPipelines;
    allocPartitionGrid();
    initPartitionGrid();
  }

  ~GridDecompositionRule() {
    deallocPartitionGrid();
  }

  bool isRuleTerminated(int pipelineId) {
    return false;
  }

  void applyRule(std::shared_ptr<FFTData> data, int pipelineId) {
    ImageStitching::CUDAImageTile *tile = data->getTile();

    int r = tile->getRowIdx();
    int c = tile->getColIdx();

    if (gridPartition[r][c] == pipelineId) {
      DEBUG_VERBOSE("Sending data to pipeline: " << pipelineId);
      addResult(data);
    }
  }

  virtual std::string getName() {
    return "Grid Decomposition Rule";
  }

  void shutdownRule(int pipelineId) {
  }

 private:

  void initPartitionGrid() {
    int sliceWidth = extentWidth;
    int sliceHeight = (int) ceil((double) extentHeight / (double) this->numPipelines);

    for (int r = 0; r < extentHeight; r++) {
      for (int c = 0; c < extentWidth; c++) {
        for (int sliceNum = 0; sliceNum < this->numPipelines; sliceNum++) {
          if (r <= sliceHeight * (sliceNum + 1) && r >= sliceHeight * sliceNum)
            gridPartition[r][c] = sliceNum;
        }

      }
    }

//        printPartitionGrid();
  }

  void printPartitionGrid() {
    std::cout << std::endl << "---------------------------------------------------" << std::endl;

    for (int r = 0; r < extentHeight; r++) {
      for (int c = 0; c < extentWidth; c++) {
        std::cout << std::to_string(gridPartition[r][c]) + " ";
      }
      std::cout << std::endl;
    }


    std::cout << "---------------------------------------------------" << std::endl;
  }
  void allocPartitionGrid() {
    gridPartition = new int *[extentHeight];
    for (int r = 0; r < extentHeight; r++) {
      gridPartition[r] = new int[extentWidth];
    }
  }

  void deallocPartitionGrid() {
    for (int r = 0; r < extentHeight; r++)
      delete[] gridPartition[r];

    delete[]gridPartition;
  }

  int startRow;
  int startCol;
  int extentWidth;
  int extentHeight;
  int numPipelines;
  int **gridPartition;

};
#endif //HTGS_DECOMPOSITIONRULE_H

