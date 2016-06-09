//
// Created by tjb3 on 2/23/16.
//

#ifndef HTGS_MATRIXACCUMULATERULE_H
#define HTGS_MATRIXACCUMULATERULE_H

#include <vector>
#include <htgs/api/IRule.hpp>
#include "../data/MatrixRequestData.h"
#include "../data/MatrixBlockMulData.h"
#include "../data/MatrixBlockData.h"

class MatrixAccumulateRule : public htgs::IRule<MatrixBlockData<double *>, MatrixBlockMulData<double *> > {
public:
    MatrixAccumulateRule(int blockWidth, int blockHeight, int blockWidthMatrixA) {
      matrixContainer = this->allocStateContainer(blockWidth, blockHeight);
      totalCount = blockWidth * blockHeight * blockWidthMatrixA + blockWidth * blockHeight * (blockWidthMatrixA-1);
      count = 0;
    }

    ~MatrixAccumulateRule() {
      free(matrixContainer);
    }

    bool isRuleTerminated(int pipelineId) {
        return count == totalCount;
    }

    void shutdownRule(int pipelineId) { }

    void applyRule(std::shared_ptr<MatrixBlockData<double *>> data, int pipelineId) {
      auto request = data->getRequest();

      int row = request->getRow();
      int col = request->getCol();

      if (matrixContainer->has(row, col))
      {
        auto blkData = matrixContainer->get(row, col);
        matrixContainer->remove(row, col);
        addResult(new MatrixBlockMulData<double *>(blkData, data));
      }
      else{
        matrixContainer->set(row, col, data);
      }
      count++;
    }

    std::string getName() {
        return "MatrixAccumulateRule";
    }

private:
  htgs::StateContainer<std::shared_ptr<MatrixBlockData<double *>>> *matrixContainer;
  int count;
  int totalCount;
};

#endif //HTGS_MATRIXACCUMULATERULE_H
