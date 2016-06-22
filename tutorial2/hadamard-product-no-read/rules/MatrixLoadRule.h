//
// Created by tjb3 on 2/23/16.
//

#ifndef HTGS_MATRIXLOADRULE_H
#define HTGS_MATRIXLOADRULE_H
#include <htgs/api/IRule.hpp>
#include "../data/MatrixBlockData.h"
#include "../data/MatrixBlockMulData.h"

class MatrixLoadRule : public htgs::IRule<MatrixBlockData<double *>, MatrixBlockMulData> {

 public:
  MatrixLoadRule(int blockWidth, int blockHeight) {
    this->blockWidth = blockWidth;
    this->blockHeight = blockHeight;

    this->blockDataArrayA = allocateDataGrid();
    this->blockDataArrayB = allocateDataGrid();

    initDataGrid(blockDataArrayA);
    initDataGrid(blockDataArrayB);
  }

  ~MatrixLoadRule() {
    deAllocDataGrid(blockDataArrayA);
    deAllocDataGrid(blockDataArrayB);
  }

  bool isRuleTerminated(int pipelineId) {
    return false;
  }

  void shutdownRule(int pipelineId) {}

  void applyRule(std::shared_ptr<MatrixBlockData<double *>> data, int pipelineId) {
    std::shared_ptr<MatrixRequestData> request = data->getRequest();

    switch (request->getType()) {
      case MatrixType::MatrixA:
        blockDataArrayA[request->getRow()][request->getCol()] = data;

        if (blockDataArrayB[request->getRow()][request->getCol()] != nullptr) {
          addResult(new MatrixBlockMulData(data, blockDataArrayB[request->getRow()][request->getCol()]));
        }
        break;
      case MatrixType::MatrixB:
        blockDataArrayB[request->getRow()][request->getCol()] = data;

        if (blockDataArrayA[request->getRow()][request->getCol()] != nullptr) {
          addResult(new MatrixBlockMulData(blockDataArrayA[request->getRow()][request->getCol()], data));
        }
        break;
      case MatrixType::MatrixC:
        break;
    }
  }

  std::string getName() {
    return "MatrixLoadRule";
  }

 private:
  std::shared_ptr<MatrixBlockData<double *>> **allocateDataGrid() {
    std::shared_ptr<MatrixBlockData<double *>>
        **dataGrid = new std::shared_ptr<MatrixBlockData<double *>> *[blockHeight];
    for (int r = 0; r < blockHeight; r++) {
      dataGrid[r] = new std::shared_ptr<MatrixBlockData<double *>>[blockWidth];
    }
    return dataGrid;
  }

  void deAllocDataGrid(std::shared_ptr<MatrixBlockData<double *>> **dataGrid) {
    for (int r = 0; r < blockHeight; r++) {
      delete[] dataGrid[r];
    }
    delete[] dataGrid;

  }

  void initDataGrid(std::shared_ptr<MatrixBlockData<double *>> **grid) {
    for (int r = 0; r < blockHeight; r++)
      for (int c = 0; c < blockWidth; c++)
        grid[r][c] = nullptr;
  }

  std::shared_ptr<MatrixBlockData<double *>> **blockDataArrayA;
  std::shared_ptr<MatrixBlockData<double *>> **blockDataArrayB;
  int blockWidth;
  int blockHeight;
};
#endif //HTGS_MATRIXLOADRULE_H
