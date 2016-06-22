//
// Created by tjb3 on 2/23/16.
//


#include "MatrixAccumTask.h"


MatrixAccumTask::MatrixAccumTask(int numThreads) : ITask(numThreads) {}

MatrixAccumTask::~MatrixAccumTask() {}

void MatrixAccumTask::executeTask(std::shared_ptr<MatrixBlockMulData<double *>> data) {

  auto matAData = data->getMatrixA();
  auto matBData = data->getMatrixB();

  auto matrixA = matAData->getMatrixData();
  auto matrixB = matBData->getMatrixData();

  long width = matAData->getMatrixWidth();
  long height = matAData->getMatrixHeight();

//    std::shared_ptr<double> result(new double[width*height], std::default_delete<double[]>());

//    double *result = new double[width*height];


  for (long j = 0; j < width; j++)
  {
    for (long i = 0; i < height; i++)
    {
//        matrixA.get()[i*width+j] += matrixB.get()[i*width+j];
//        result.get()[i*width+j] = matrixA.get()[i*width+j] + matrixB.get()[i*width+j];
      matrixA[IDX2C(i, j, height)] = matrixA[IDX2C(i, j, height)] + matrixB[IDX2C(i, j, height)];
//        result[i*width+j] = matrixA[i*width+j] + matrixB[i*width+j];
    }
  }

//    delete [] matrixA;
//    matrixA = nullptr;
  delete [] matrixB;
  matrixB = nullptr;

//    releasePointers.push_back(matrixB);
//    auto matRequest = matAData->getRequest();
//    std::shared_ptr<MatrixRequestData> matReq(new MatrixRequestData(matRequest->getRow(), matRequest->getCol(), MatrixType::MatrixC));

//    addResult(new MatrixBlockData<double *>(matReq, result, width, height));
  addResult(data->getMatrixA());

}