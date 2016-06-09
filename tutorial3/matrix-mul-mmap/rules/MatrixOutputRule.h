//
// Created by tjb3 on 2/23/16.
//

#ifndef HTGS_MATRIXOUTPUTRULE_H
#define HTGS_MATRIXOUTPUTRULE_H

#include <vector>
#include <htgs/api/IRule.hpp>
#include "../data/MatrixRequestData.h"
#include "../data/MatrixBlockMulData.h"
#include "../data/MatrixBlockData.h"

class MatrixOutputRule : public htgs::IRule<MatrixBlockData<double *>, MatrixBlockData<double *> > {
public:
    MatrixOutputRule(int blockWidth, int blockHeight, int blockWidthMatrixA) {
      matrixCountContainer = this->allocStateContainer<int>(blockWidth, blockHeight, 0);
      numBlocks = 2 * blockWidthMatrixA - 1;
    }

    ~MatrixOutputRule() {
      free(matrixCountContainer);
    }

    bool isRuleTerminated(int pipelineId) {
        return false;
    }

    void shutdownRule(int pipelineId) { }

    void applyRule(std::shared_ptr<MatrixBlockData<double *>> data, int pipelineId) {
      auto request = data->getRequest();

      int row = request->getRow();
      int col = request->getCol();

      int count = matrixCountContainer->get(row, col);
      count++;
      matrixCountContainer->set(row, col, count);
      if (count == numBlocks)
      {
        addResult(data);
      }
    }

    std::string getName() {
        return "MatrixOutputRule";
    }

private:
  htgs::StateContainer<int> *matrixCountContainer;
  int numBlocks;
};

#endif //HTGS_MATRIXACCUMULATERULE_H
