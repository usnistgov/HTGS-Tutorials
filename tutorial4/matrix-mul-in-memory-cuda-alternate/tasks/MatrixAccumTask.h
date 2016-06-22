//
// Created by tjb3 on 2/23/16.
//


#ifndef HTGS_MATRIXACCUMTASK_H
#define HTGS_MATRIXACCUMTASK_H

#include "../data/MatrixBlockMulData.h"
#include "../data/MatrixBlockData.h"
#include "../../../tutorial-utils/util-matrix.h"
#include <htgs/api/ITask.hpp>

class MatrixAccumTask : public htgs::ITask<MatrixBlockMulData<double *>, MatrixBlockData<double *>> {

 public:
  MatrixAccumTask(int numThreads);

  virtual ~MatrixAccumTask();

  virtual void executeTask(std::shared_ptr<MatrixBlockMulData<double *>> data);

  virtual std::string getName() {
    return "MatrixAccumTask";
  }
  virtual MatrixAccumTask *copy() {
    return new MatrixAccumTask(this->getNumThreads());
  }

};

#endif //HTGS_MATRIXACCUMTASK_H
