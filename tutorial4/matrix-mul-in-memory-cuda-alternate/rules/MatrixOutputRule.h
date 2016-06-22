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
      matrixCountContainer = this->allocStateContainer<int>(blockHeight, blockWidth, 0);
      numBlocks = blockWidthMatrixA + blockWidthMatrixA - 1;
    }

    ~MatrixOutputRule() {
      delete matrixCountContainer;
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
//      std::cout << this->getName() << "Received " << row << ", " << col  << " -- " << count << " of " << numBlocks << std::endl;
      matrixCountContainer->set(row, col, count);
      if (count == numBlocks)
      {
//        std::cout << this->getName() << " sending " << row << ", " << col << " -- " << count << " of " << numBlocks << std::endl;
        addResult(data);
      }
      else if (count > numBlocks)
      {
        std::cout << "Additional vals received" << std::endl;
      }
    }

    std::string getName() {
        return "MatrixOutputRule";
    }

private:
  htgs::StateContainer<int> *matrixCountContainer;
  int numBlocks;
};

#endif //MatrixOutputRule
