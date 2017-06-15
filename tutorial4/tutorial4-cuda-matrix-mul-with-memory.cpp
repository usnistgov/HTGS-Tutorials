//
// Created by tjb3 on 6/13/17.
//

#include <fstream>
#include <htgs/api/TaskGraphConf.hpp>
#include <htgs/api/TaskGraphRuntime.hpp>
#include <cublasXt.h>
#include "../tutorial-utils/matrix-library/tasks/CudaCopyInTask.h"
#include "../tutorial-utils/matrix-library/operations/matmul.h"
#include "../tutorial-utils/matrix-library/args/MatMulArgs.h"
#include "../tutorial-utils/matrix-library/tasks/CudaCopyOutTask.h"
#include "../tutorial-utils/matrix-library/allocator/CudaAllocator.h"
#include "../tutorial-utils/matrix-library/tasks/LoadMatrixTask.h"

#include "../tutorial-utils/util-matrix.h"
#include "../tutorial-utils/SimpleClock.h"
#include "../tutorial-utils/util-cuda.h"

#include "../tutorial3/tasks/MatMulAccumTask.h"
#include "../tutorial3/tasks/MatMulOutputTask.h"
#include "../tutorial3/rules/MatMulDistributeRule.h"
#include "../tutorial3/rules/MatMulLoadRule.h"
#include "../tutorial3/rules/MatMulAccumulateRule.h"



#include "tasks/MatrixMulBlkCudaTask.h"

int validateResults(double *matrixC, double *matrixC_HTGS, size_t fullMatrixAHeight, size_t fullMatrixBWidth) {

  if (!validateMatMulResults(20, matrixC, matrixC_HTGS, fullMatrixAHeight*fullMatrixBWidth))
  {
    return -1;
  }

  return 0;
}

void computeSequentialMatMul(double *matrixA,
                               double *matrixB,
                               double *matrixC,
                               size_t fullMatrixAHeight,
                               size_t fullMatrixAWidth,
                               size_t fullMatrixBWidth,
                               cublasXtHandle_t handle) {

    double alpha = 1.0;
    double beta = 0.0;

    cublasXtDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, fullMatrixAHeight, fullMatrixBWidth, fullMatrixAWidth, &alpha,
                  matrixA, fullMatrixAHeight,
                  matrixB, fullMatrixAHeight,
                  &beta, matrixC, fullMatrixAHeight);

  cudaDeviceSynchronize();
}

int main(int argc, char *argv[])
{
  MatMulArgs matMulArgs;
  matMulArgs.processArgs(argc, argv);

  size_t matrixAHeight = matMulArgs.getMatrixAHeight();
  size_t matrixBWidth = matMulArgs.getMatrixBWidth();
  size_t sharedDim = matMulArgs.getSharedDim();

  size_t blockSize = matMulArgs.getBlockSize();
  size_t numReadThreads = matMulArgs.getNumReadThreads();
  size_t numProdThreads = matMulArgs.getNumMatMulThreads();
  size_t numAccumThreads = (size_t) ceil((double)numProdThreads / 2.0);
  std::string directory = matMulArgs.getDirectory();
  std::string outputDirectory = matMulArgs.getOutputDir();
  bool runSequential = matMulArgs.isRunSequential();
  bool validate = matMulArgs.isValidateResults();

  size_t numGPUs = matMulArgs.getNumGPUs();
  int gpuIds[numGPUs];

  matMulArgs.copyGpuIds(gpuIds);


  CUcontext *contexts = initCuda(numGPUs, gpuIds);

  std::string runtimeFileStr("runtimes");

  int numRetry = 1;

  std::ofstream runtimeFile(runtimeFileStr, std::ios::app);
  double *matrixA = new double[matrixAHeight * sharedDim];
  double *matrixB = new double[matrixBWidth * sharedDim];
  double *matrixC = new double[matrixAHeight * matrixBWidth];

  initMatrix(matrixA, sharedDim, matrixAHeight, true);
  initMatrix(matrixB, matrixBWidth, sharedDim, true);

  for (int numTry = 0; numTry < numRetry; numTry++) {
    SimpleClock clk;
    SimpleClock endToEnd;

    if (runSequential) {
      endToEnd.start();
      initMatMul(numProdThreads);

      cublasXtHandle_t handle;

      cublasXtCreate(&handle);

      cublasXtDeviceSelect(handle, numGPUs, gpuIds);
      cublasXtSetBlockDim(handle, blockSize);

      clk.start();
      computeSequentialMatMul(matrixA, matrixB, matrixC, (size_t) matrixAHeight, (size_t) sharedDim,
                              (size_t) matrixBWidth, handle);
      clk.stopAndIncrement();

      cublasXtDestroy(handle);

      endToEnd.stopAndIncrement();
    }
    else {
      endToEnd.start();
      initMatMul(1);

      LoadMatrixTask *readAMatTask =
          new LoadMatrixTask(matrixA,
                             numReadThreads,
                             MatrixType::MatrixA,
                             blockSize,
                             sharedDim,
                             matrixAHeight,
                             true);

      LoadMatrixTask *readBMatTask =
          new LoadMatrixTask(matrixB,
                             numReadThreads,
                             MatrixType::MatrixB,
                             blockSize,
                             matrixBWidth,
                             sharedDim,
                             true);

      MatrixMulBlkCudaTask *mmulTask = new MatrixMulBlkCudaTask(contexts, gpuIds, numGPUs);
      MatMulAccumTask *accumTask = new MatMulAccumTask(numAccumThreads, true);

      MatMulOutputTask *outputTask = new MatMulOutputTask(matrixC, matrixAHeight, blockSize, true);

      size_t blkHeightMatB = readBMatTask->getNumBlocksRows();
      size_t blkWidthMatB = readBMatTask->getNumBlocksCols();

      size_t blkHeightMatA = readAMatTask->getNumBlocksRows();
      size_t blkWidthMatA = readAMatTask->getNumBlocksCols();

      CudaCopyInTask *cudaCopyInATask = new CudaCopyInTask(contexts, gpuIds, numGPUs, MatrixType::MatrixA, blkWidthMatB);
      CudaCopyInTask *cudaCopyInBTask = new CudaCopyInTask(contexts, gpuIds, numGPUs, MatrixType::MatrixB, blkHeightMatA);

      CudaCopyOutTask *cudaCopyOutCTask = new CudaCopyOutTask(contexts, gpuIds, numGPUs, MatrixType::MatrixC);

      MatMulDistributeRule *distributeRuleMatA = new MatMulDistributeRule(MatrixType::MatrixA);
      MatMulDistributeRule *distributeRuleMatB = new MatMulDistributeRule(MatrixType::MatrixB);

      MatMulLoadRule<htgs::m_data_t<double>> *loadRule = new MatMulLoadRule<htgs::m_data_t<double>>(blkWidthMatA, blkHeightMatA, blkWidthMatB, blkHeightMatB);
      MatMulAccumulateRule<double *> *accumulateRule = new MatMulAccumulateRule<double *>(blkWidthMatB, blkHeightMatA, blkWidthMatA);

      MatMulOutputRule *outputRule = new MatMulOutputRule(blkWidthMatB, blkHeightMatA, blkWidthMatA);

      auto distributeBk = new htgs::Bookkeeper<MatrixRequestData>();
      auto matMulBk = new htgs::Bookkeeper<MatrixBlockData<htgs::m_data_t<double>>>();
      auto matAccumBk = new htgs::Bookkeeper<MatrixBlockData<double *>>();

      auto taskGraph = new htgs::TaskGraphConf<MatrixRequestData, MatrixRequestData>();

      taskGraph->setGraphConsumerTask(distributeBk);
      taskGraph->addRuleEdge(distributeBk, distributeRuleMatA, readAMatTask);
      taskGraph->addRuleEdge(distributeBk, distributeRuleMatB, readBMatTask);


      taskGraph->addEdge(readAMatTask, cudaCopyInATask);
      taskGraph->addEdge(readBMatTask, cudaCopyInBTask);

      taskGraph->addEdge(cudaCopyInATask, matMulBk);
      taskGraph->addEdge(cudaCopyInBTask, matMulBk);

      taskGraph->addRuleEdge(matMulBk, loadRule, mmulTask);

      taskGraph->addEdge(mmulTask, cudaCopyOutCTask);

      taskGraph->addEdge(cudaCopyOutCTask, matAccumBk);

      taskGraph->addRuleEdge(matAccumBk, outputRule, outputTask);
      taskGraph->addRuleEdge(matAccumBk, accumulateRule, accumTask);

      taskGraph->addEdge(accumTask, matAccumBk);

      taskGraph->addGraphProducerTask(outputTask);


      taskGraph->addCudaMemoryManagerEdge(matrixTypeToString(MatrixType::MatrixA) + "Copy",
                                          cudaCopyInATask,
                                          new CudaAllocator(blockSize, blockSize),
                                          blkWidthMatB+1,
                                          htgs::MMType::Static,
                                          contexts);
      taskGraph->addCudaMemoryManagerEdge(matrixTypeToString(MatrixType::MatrixB) + "Copy",
                                          cudaCopyInBTask,
                                          new CudaAllocator(blockSize, blockSize),
                                          blkHeightMatA+1,
                                          htgs::MMType::Static,
                                          contexts);

      taskGraph->addCudaMemoryManagerEdge(matrixTypeToString(MatrixType::MatrixC),
                                          mmulTask,
                                          new CudaAllocator(blockSize, blockSize),
                                          4,
                                          htgs::MMType::Static,
                                          contexts);

      htgs::TaskGraphRuntime *runtime = new htgs::TaskGraphRuntime(taskGraph);

      clk.start();

      runtime->executeRuntime();

      for (size_t col = 0; col < blkWidthMatA; col++) {
        for (size_t row = 0; row < blkHeightMatA; row++) {

          MatrixRequestData *matA = new MatrixRequestData(row, col, MatrixType::MatrixA);
          taskGraph->produceData(matA);
        }
      }

      for (size_t row = 0; row < blkHeightMatB; row++) {
        for (size_t col = 0; col < blkWidthMatB; col++) {
          MatrixRequestData *matB = new MatrixRequestData(row, col, MatrixType::MatrixB);
          taskGraph->produceData(matB);

        }
      }

      taskGraph->finishedProducingData();

      while (!taskGraph->isOutputTerminated()) {
        auto data = taskGraph->consumeData();
        if (data != nullptr) {
//          std::cout << data->getRow() << ", " << data->getCol() << std::endl;
        }
      }

      runtime->waitForRuntime();


//      taskGraph->writeDotToFile("profile-graph.dot");
      taskGraph->writeDotToFile("profile-all-threads-graph.dot", DOTGEN_FLAG_SHOW_ALL_THREADING);
      taskGraph->writeDotToFile("profile-graph.dot", DOTGEN_COLOR_COMP_TIME);

      clk.stopAndIncrement();

      delete runtime;
      endToEnd.stopAndIncrement();
    }

    if (validate) {
      double *matrixCTest = new double[matrixAHeight * matrixBWidth];
      initMatMul(numProdThreads);

      cublasXtHandle_t handle;

      cublasXtCreate(&handle);

      cublasXtDeviceSelect(handle, (int)numGPUs, gpuIds);
      cublasXtSetBlockDim(handle, (int)blockSize);

      computeSequentialMatMul(matrixA, matrixB, matrixCTest, (size_t) matrixAHeight, (size_t) sharedDim,
                              (size_t) matrixBWidth, handle);

      cublasXtDestroy(handle);


      int res = validateResults(matrixC, matrixCTest, matrixAHeight, matrixBWidth);
      if (res != 0) {
        std::cout << "Error validating test failed!" << std::endl;
      }
      else
      {
        std::cout << "Test PASSED" << std::endl;
      }

      delete []matrixCTest;

    }

    std::cout << (runSequential ? "sequential" : "htgs") << ", " << numProdThreads
              << ", accum-threads: " << numAccumThreads << ", width-b: " << matrixBWidth << ", height-a: " << matrixAHeight
              << ", shared-dim: " << sharedDim
              << ", blockSize: " << blockSize
              << ", time:" << clk.getAverageTime(TimeVal::MILLI)
              << ", end-to-end:" << endToEnd.getAverageTime(TimeVal::MILLI)

              << std::endl;

    runtimeFile << (runSequential ? "sequential" : "htgs") << ", " << numProdThreads
                << ", " << numAccumThreads << ", "
                << matrixBWidth << ", " << matrixAHeight
                << ", " << sharedDim << ", " << blockSize << ", " << clk.getAverageTime(TimeVal::MILLI)
                << ", " << endToEnd.getAverageTime(TimeVal::MILLI)
                << std::endl;

  }

  delete[] matrixA;
  delete[] matrixB;
  delete[] matrixC;
}