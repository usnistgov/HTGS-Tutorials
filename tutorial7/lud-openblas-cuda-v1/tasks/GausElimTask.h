//
// Created by tjb3 on 7/28/16.
//

#ifndef HTGS_TUTORIALS_GAUSELIMTASK_H
#define HTGS_TUTORIALS_GAUSELIMTASK_H

#include <htgs/api/ITask.hpp>
#include <cblas.h>
#include <lapacke.h>
#include "../data/MatrixBlockData.h"
#include "../../../tutorial-utils/util-matrix.h"
class GausElimTask : public htgs::ITask<MatrixBlockData<double *>, MatrixBlockData<double *>>
{
 public:
  GausElimTask(int numThreads, long fullMatrixHeight, long fullMatrixWidth, int blockSize) :
  ITask(numThreads), fullMatrixHeight(fullMatrixHeight), fullMatrixWidth(fullMatrixWidth), blockSize(blockSize)
  {
    this->ipiv = new int[blockSize*blockSize];
  }

  ~GausElimTask()
  {
    delete []ipiv;
  }

  virtual void executeTask(std::shared_ptr<MatrixBlockData<double *>> data) {

    double *matrix = data->getMatrixData();

    LAPACKE_dgetf2(LAPACK_COL_MAJOR, data->getMatrixHeight(), data->getMatrixWidth(), matrix, fullMatrixHeight, ipiv);

    /*
    for (int diag = 0; diag < data->getMatrixWidth()-1; diag++)
    {
      const double diagVal = 1.0 / matrix[IDX2C(diag, diag, fullMatrixHeight)];

      // elimination step
      for (int r = diag+1; r < data->getMatrixHeight(); r++)
      {
        matrix[IDX2C(r, diag, fullMatrixHeight)] = matrix[IDX2C(r, diag, fullMatrixHeight)] * diagVal;
      }

      // update step
      for (int col = diag+1; col < data->getMatrixWidth(); col++)
      {
        const double colVal = matrix[IDX2C(diag, col, fullMatrixHeight)];

        for (int row = diag+1; row < data->getMatrixHeight(); row++)
        {
          double rowVal = matrix[IDX2C(row, diag, fullMatrixHeight)];

          matrix[IDX2C(row,col, fullMatrixHeight)] = matrix[IDX2C(row,col, fullMatrixHeight)] - colVal * rowVal;
        }
      }

    }
     */

    addResult(data);
  }
  virtual GausElimTask *copy() {
    return new GausElimTask(this->getNumThreads(), fullMatrixHeight, fullMatrixWidth, blockSize);
  }

  virtual std::string getName() {
    return "GausElimTask";
  }

 private:
  long fullMatrixWidth;
  long fullMatrixHeight;
  int *ipiv;
  int blockSize;
};

#endif //HTGS_TUTORIALS_GAUSELIMTASK_H
