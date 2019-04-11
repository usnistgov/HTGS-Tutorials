
// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
// NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
// You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

//
// Created by tjb3 on 6/15/16.
//

#ifndef HTGS_TUTORIALS_COPYINWINDOWPANEL_H
#define HTGS_TUTORIALS_COPYINWINDOWPANEL_H

#include <htgs/api/ICudaTask.hpp>
#include "../memory/MatrixMemoryRule.h"
#include <cuda.h>
#include <cublas_v2.h>

class CopyInPanelWindowTask : public htgs::ICudaTask<MatrixPanelData, MatrixPanelData> {
 public:
  CopyInPanelWindowTask(int blockSize, int *cudaIds, int numGpus, long leadingDimensionFullMatrix, int numBlocksWidth, std::string memoryEdge, PanelState panelState) :
      ICudaTask(cudaIds, numGpus),
      blockSize(blockSize),
      leadingDimensionFullMatrix(leadingDimensionFullMatrix),
      numBlocksWidth(numBlocksWidth),
      memoryEdge(memoryEdge),
      panelState(panelState)
  {
    panelCache = new htgs::StateContainer<std::shared_ptr<MatrixPanelData>>(numBlocksWidth, 1, nullptr);
  }

  virtual void executeTask(std::shared_ptr<MatrixPanelData> data) override {

    // Received a panel that is to be used for cache.
    int panelColumn = data->getPanelCol();

    // If the panel already exists, then no need to do the copy
    if (this->panelCache->has(panelColumn))
    {
      auto prevPanel = this->panelCache->get(panelColumn);

      // Copy over original metadata from the previous panel
      data->setOrigMemory(prevPanel->getOrigMemory());
      data->setOrigHeight(prevPanel->getOrigHeight());
      data->setMemoryData(prevPanel->getMemoryData());
      data->setOriginalOperatingDiagonal(prevPanel->getOriginalOperatingDiagonal());

      int originalDiagonal = prevPanel->getOriginalOperatingDiagonal();
      int curDiagonal = data->getPanelOperatingDiagonal();

      // Update the cuda memory start address to point to the operating diagonal address
      double *startAddress = &prevPanel->getMemoryData()->get()[IDX2C((curDiagonal-originalDiagonal)*blockSize, 0, leadingDimensionFullMatrix)];

      // Update address to operate with the data's diagonal
      data->setCudaMemoryStartAddr(startAddress);
      data->setPanelState(panelState);
      data->setWindowed(true);

      addResult(data);
    }
    else {

      // CPU Memory
      double *memoryIn = data->getMemory();

      // Cuda Memory
      int releaseCount = panelColumn - data->getPanelOperatingDiagonal();

      auto memoryOut = this->getMemory<double>(memoryEdge, new MatrixMemoryRule(releaseCount));

      cublasSetMatrixAsync((int) data->getHeight(), (int) blockSize, sizeof(double),
                           memoryIn, (int) leadingDimensionFullMatrix,
                           memoryOut->get(), (int) leadingDimensionFullMatrix, this->getStream());

      data->setOrigMemory(memoryIn);
      data->setOrigHeight(data->getHeight());
      data->setMemoryData(memoryOut);
      data->setCudaMemoryStartAddr(memoryOut->get());
      data->setPanelState(panelState);
      data->setWindowed(true);

      // Sets what diagonal this panel originally operated with
      data->setOriginalOperatingDiagonal(data->getPanelOperatingDiagonal());

      // Cache the panel
      panelCache->set(panelColumn, data);

      this->syncStream();

      this->addResult(data);
    }
  }

  virtual void shutdownCuda() {
    delete panelCache;
  }

  virtual std::string getName() {
    return "CudaCopyInPanelWindowTask";
  }

  virtual CopyInPanelWindowTask *copy() {
    return new CopyInPanelWindowTask(
                                this->blockSize,
                                this->getCudaIds(),
                                this->getNumGPUs(),
                                this->leadingDimensionFullMatrix,
                                this->numBlocksWidth,
                                this->memoryEdge,
                                panelState);
  }

 private:
  int blockSize;
  long leadingDimensionFullMatrix;
  int numBlocksWidth;
  std::string memoryEdge;
  PanelState panelState;
  htgs::StateContainer<std::shared_ptr<MatrixPanelData>> *panelCache;
};

#endif //HTGS_TUTORIALS_COPYINWINDOWPANEL_H
