
// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
// NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
// You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

//
// Created by tjb3 on 2/23/16.
//

#ifndef HTGS_MATRIXPANELDATA_H
#define HTGS_MATRIXPANELDATA_H

#include <htgs/api/IData.hpp>

#include "MatrixBlockData.h"

enum class PanelState {
  NONE,
  ALL_FACTORED,
  TOP_FACTORED,
  UPDATED
};

class MatrixPanelData : public htgs::IData {
 public:

  MatrixPanelData(long height, long blockSize, int panelCol, int panelOperatingDiagonal, PanelState panelState) :
      height(height), blockSize(blockSize), panelCol(panelCol), panelOperatingDiagonal(panelOperatingDiagonal), panelState(panelState)
  {
    memory = nullptr;
    memoryData = nullptr;
    windowed = false;
  }

  double *getStartMemoryAddr() const {
    return &memory[blockSize*panelOperatingDiagonal];
  }

  double *getMemory() const {
    return memory;
  }
  void setMemory(double *memory) {
    this->memory = memory;
  }
  const MatrixMemoryData_t &getMemoryData() const {
    return memoryData;
  }
  void setMemoryData(const MatrixMemoryData_t &memoryData) {
    this->memoryData = memoryData;
  }

  double *getCudaMemoryStartAddr() const {
    return cudaMemoryStartAddr;
  }
  void setCudaMemoryStartAddr(double *cudaMemoryStartAddr) {
    this->cudaMemoryStartAddr = cudaMemoryStartAddr;
  }

  long getHeight() const {
    return height;
  }
  int getPanelCol() const {
    return panelCol;
  }
  int getPanelOperatingDiagonal() const {
    return panelOperatingDiagonal;
  }

  PanelState getPanelState() const {
    return panelState;
  }

  void setPanelState(PanelState panelState) {
    this->panelState = panelState;
  }

  int getOriginalOperatingDiagonal() const {
    return originalOperatingDiagonal;
  }
  void setOriginalOperatingDiagonal(int originalOperatingDiagonal) {
    MatrixPanelData::originalOperatingDiagonal = originalOperatingDiagonal;
  }

  bool isWindowed() const {
    return windowed;
  }
  void setWindowed(bool windowed) {
    MatrixPanelData::windowed = windowed;
  }

  double *getOrigMemory() const {
    return origMemory;
  }
  void setOrigMemory(double *origMemory) {
    MatrixPanelData::origMemory = origMemory;
  }

  long getOrigHeight() const {
    return origHeight;
  }
  void setOrigHeight(long origHeight) {
    MatrixPanelData::origHeight = origHeight;
  }

 private:
  double *memory;
  double *origMemory;
  long origHeight;
  MatrixMemoryData_t memoryData;
  double *cudaMemoryStartAddr;
  long height;
  long blockSize;
  int panelCol;
  int panelOperatingDiagonal;
  int originalOperatingDiagonal;
  PanelState panelState;
  bool windowed;
};
#endif //HTGS_MATRIXPANELDATA_H
