//
// Created by tjb3 on 2/23/16.
//
//#define DEBUG_FLAG
//#define DEBUG_LEVEL_VERBOSE

#include <htgs/api/TaskGraph.hpp>
#include <htgs/api/Runtime.hpp>
#include <htgs/api/ExecutionPipeline.hpp>
#include <cblas.h>
#include <cublasXt.h>
#include <cuda_runtime_api.h>

#define gpuErrorChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

#include "data/MatrixRequestData.h"
#include "data/MatrixBlockData.h"
#include "tasks/ReadMatrixTask.h"
#include "rules/MatrixLoadRule.h"
#include "tasks/MatrixMulBlkTask.h"
#include "tasks/MatrixAccumTask.h"
#include "tasks/OutputTask.h"
#include "rules/MatrixAccumulateRule.h"
#include "rules/MatrixDistributeRule.h"
#include "rules/MatrixOutputRule.h"
#include "rules/MatrixDecompositionRule.h"
#include "../../tutorial-utils/SimpleClock.h"

#include "tasks/MatrixCopyInTask.h"
#include "tasks/MatrixCopyOutTask.h"
#include "memory/CudaMatrixAllocator.h"
#include "../../tutorial-utils/util-cuda.h"

int validateResults(double *matrixC, double *matrixC_HTGS, int fullMatrixAHeight, int fullMatrixBWidth) {
  int count = 0;

  for (int r = 0; r < fullMatrixAHeight; r++) {
    for (int c = 0; c < fullMatrixBWidth; c++) {
//      if (matrixC[r*fullMatrixBWidth+c] != matrixC_HTGS[IDX2C(r, c, fullMatrixAHeight)])
      if (matrixC[IDX2C(r, c, fullMatrixAHeight)] != matrixC_HTGS[IDX2C(r, c, fullMatrixAHeight)]) {
        if (count <= 20) {
          std::cout << std::fixed << r * fullMatrixBWidth + c << ": cMem = " << matrixC[r * fullMatrixBWidth + c]
                    << " cMemPar = " << matrixC_HTGS[IDX2C(r, c, fullMatrixAHeight)] << std::endl;
        }
        count++;
      }
    }
  }
//  for (int i = 0; i < fullMatrixAHeight*fullMatrixBWidth; i++)
//  {
//    if (matrixC[i] != matrixC_HTGS[i])
//    {
//      if (count <= 20) {
//        std::cout << std::fixed << i << ": cMem = " << matrixC[i] << " cMemPar = " << matrixC_HTGS[i] << std::endl;
//      }
//      count++;
//    }
//  }

  if (count > 0)
    std::cout << "Total incorrect = " << count << std::endl;

  if (count > 0)
    return 1;
  return 0;
}

void computeSequentialMatMul(double *matrixA,
                             double *matrixB,
                             double *matrixC,
                             size_t fullMatrixAHeight,
                             size_t fullMatrixAWidth,
                             size_t fullMatrixBWidth,
                             int blockDim,
                             int numGpus,
                             int *devices) {

  cublasXtHandle_t handle;

  cublasXtCreate(&handle);

  cublasXtDeviceSelect(handle, numGpus, devices);
  cublasXtSetBlockDim(handle, blockDim);

  double alpha = 1.0;
  double beta = 0.0;

  cublasXtDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, fullMatrixAHeight, fullMatrixBWidth, fullMatrixAWidth, &alpha,
                matrixA, fullMatrixAHeight,
                matrixB, fullMatrixAHeight,
                &beta, matrixC, fullMatrixAHeight);


//  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, fullMatrixAHeight, fullMatrixBWidth, fullMatrixAWidth, 1.0, matrixA, fullMatrixAHeight,
//              matrixB, fullMatrixAWidth, 0.0, matrixC, fullMatrixAHeight);


  cublasXtDestroy(handle);
}

int main(int argc, char *argv[]) {
  long matrixAHeight = 1024;
  long matrixBWidth = 1024;
  long sharedDim = 1024;

  int blockSize = 512;
  int numReadThreads = 1;
  int numProdThreads = 26;
  int numBlasThreads = 40;
  bool runSequential = false;
  bool validate = false;

  int numGpus = 1;

  std::string runtimeFileStr("runtimes");

  int numRetry = 1;

  if (argc > 1) {
    for (int arg = 1; arg < argc; arg++) {
      std::string argvs(argv[arg]);

      if (argvs == "--width-b") {
        arg++;
        matrixBWidth = atoi(argv[arg]);
      }

      if (argvs == "--height-a") {
        arg++;
        matrixAHeight = atoi(argv[arg]);
      }

      if (argvs == "--shared-dim") {
        arg++;
        sharedDim = atoi(argv[arg]);
      }

      if (argvs == "--run-sequential") {
        runSequential = true;
      }

      if (argvs == "--num-retry" && arg + 1 < argc) {
        arg++;
        numRetry = atoi(argv[arg]);
      }

      if (argvs == "--block-size") {
        arg++;
        blockSize = atoi(argv[arg]);
      }

      if (argvs == "--num-threads-htgs" && arg + 1 < argc) {
        numProdThreads = atoi(argv[arg + 1]);
        arg++;
      }

      if (argvs == "--num-threads-blas" && arg + 1 < argc) {
        numBlasThreads = atoi(argv[arg + 1]);
        arg++;
      }

      if (argvs == "--runtime-file" && arg + 1 < argc) {
        runtimeFileStr = argv[arg + 1];
        arg++;
      }

      if (argvs == "--validate-results") {
        validate = true;
      }

      if (argvs == "--num-gpus") {
        numGpus = atoi(argv[arg + 1]);
        arg++;
      }

      if (argvs == "--help") {
        std::cout << argv[0]
                  << " args: [--width-b <#>] [--height-a <#>] [--shared-dim <#>] [--block-size <#>] [--num-retry <#>] [--num-threads-htgs <#>] [--num-threads-blas <#>] [--runtime-file <filename>] [--dir <dir>] [--output-dir <dir>] [--validate-results] [--run-sequential] [--help]"
                  << std::endl;
        exit(0);
      }
    }
  }


  // Initialize GPUs
  int *cudaIds = new int[2]{2, 1};

  std::ofstream runtimeFile(runtimeFileStr, std::ios::app);
  double *matrixA = new double[matrixAHeight * sharedDim];
  double *matrixB = new double[matrixBWidth * sharedDim];
  double *matrixC = new double[matrixAHeight * matrixBWidth];
  double *matrixC_HTGS = new double[matrixAHeight * matrixBWidth];

  initMatrix(matrixA, sharedDim, matrixAHeight, true);
  initMatrix(matrixB, matrixBWidth, sharedDim, true);

  for (int numTry = 0; numTry < numRetry; numTry++) {
    SimpleClock clk;

    if (runSequential) {
      openblas_set_num_threads(numBlasThreads);

      clk.start();
      computeSequentialMatMul(matrixA, matrixB, matrixC, (size_t) matrixAHeight, (size_t) sharedDim,
                              (size_t) matrixBWidth, blockSize, numGpus, cudaIds);
      clk.stopAndIncrement();
    }
    else {
      CUcontext *contexts = initCuda(numGpus, cudaIds);

      openblas_set_num_threads(1);

      ReadMatrixTask
          *readAMatTask = new ReadMatrixTask(numReadThreads, MatrixType::MatrixA, blockSize,
                                             sharedDim, matrixAHeight, matrixA, "A");
      ReadMatrixTask
          *readBMatTask = new ReadMatrixTask(numReadThreads, MatrixType::MatrixB, blockSize,
                                             matrixBWidth, sharedDim, matrixB, "B");

      OutputTask *outputTask = new OutputTask(matrixC_HTGS, matrixBWidth, matrixAHeight, blockSize);

      int blkHeightMatB = readBMatTask->getNumBlocksRows();
      int blkWidthMatB = readBMatTask->getNumBlocksCols();

      int blkHeightMatA = readAMatTask->getNumBlocksRows();
      int blkWidthMatA = readAMatTask->getNumBlocksCols();

      MatrixCopyInTask *copyInA =
          new MatrixCopyInTask("MatrixA", blockSize, blkWidthMatB, contexts, cudaIds, numGpus, matrixAHeight);
      MatrixCopyInTask
          *copyInB = new MatrixCopyInTask("MatrixB", blockSize, blkHeightMatA, contexts, cudaIds, numGpus, sharedDim);

      MatrixCopyOutTask *copyOutC = new MatrixCopyOutTask("MatrixC", blockSize, contexts, cudaIds, numGpus);

      MatrixMulBlkTask *mmulTask = new MatrixMulBlkTask(contexts,
                                                        cudaIds,
                                                        numGpus,
                                                        sharedDim,
                                                        matrixAHeight,
                                                        matrixBWidth,
                                                        sharedDim,
                                                        blockSize);

      MatrixAccumTask *accumTask = new MatrixAccumTask((int) ceil((double) numProdThreads / 2.0));

      MatrixDistributeRule *distributeRuleMatA = new MatrixDistributeRule(MatrixType::MatrixA);
      MatrixDistributeRule *distributeRuleMatB = new MatrixDistributeRule(MatrixType::MatrixB);

      MatrixLoadRule *loadRule = new MatrixLoadRule(blkWidthMatA, blkHeightMatA, blkWidthMatB, blkHeightMatB);
      MatrixAccumulateRule *accumulateRule = new MatrixAccumulateRule(blkWidthMatB, blkHeightMatA, blkWidthMatA);

//      MatrixLoopRule *loopRuleMatA = new MatrixLoopRule(0);//blkWidthMatA);
//      MatrixLoopRule *loopRuleMatB = new MatrixLoopRule(0);//blkHeightMatB);

      MatrixOutputRule *outputRule = new MatrixOutputRule(blkWidthMatB, blkHeightMatA, blkWidthMatA);

      auto distributeBk = new htgs::Bookkeeper<MatrixRequestData>();
      auto matMulBk = new htgs::Bookkeeper<MatrixBlockData<MatrixMemoryData_t>>();
      auto matAccumBk = new htgs::Bookkeeper<MatrixBlockData<double *>>();

      auto taskGraph = new htgs::TaskGraph<MatrixRequestData, MatrixRequestData>();
      auto subTaskGraph = new htgs::TaskGraph<MatrixRequestData, MatrixBlockData<double *>>();

      subTaskGraph->addGraphInputConsumer(distributeBk);
      subTaskGraph->addRule(distributeBk, readAMatTask, distributeRuleMatA);
      subTaskGraph->addRule(distributeBk, readBMatTask, distributeRuleMatB);

      subTaskGraph->addEdge(readAMatTask, copyInA);
      subTaskGraph->addEdge(readBMatTask, copyInB);

      subTaskGraph->addEdge(copyInA, matMulBk);
      subTaskGraph->addEdge(copyInB, matMulBk);

      subTaskGraph->addRule(matMulBk, mmulTask, loadRule);
//      taskGraph->addRule(matMulBk, readAMatTask, loopRuleMatA);
//      taskGraph->addRule(matMulBk, readBMatTask, loopRuleMatB);

      subTaskGraph->addEdge(mmulTask, copyOutC);
      subTaskGraph->addGraphOutputProducer(copyOutC);

      subTaskGraph->addCudaMemoryManagerEdge("MatrixACopy",
                                             copyInA,
                                             mmulTask,
                                             new CudaMatrixAllocator(blockSize, blockSize),
                                             blkWidthMatB + 4,
                                             htgs::MMType::Static,
                                             contexts);

      subTaskGraph->addCudaMemoryManagerEdge("MatrixBCopy",
                                             copyInB,
                                             mmulTask,
                                             new CudaMatrixAllocator(blockSize, blockSize),
                                             blkHeightMatA + 4,
                                             htgs::MMType::Static,
                                             contexts);

      subTaskGraph->addCudaMemoryManagerEdge("MatrixC",
                                             mmulTask,
                                             copyOutC,
                                             new CudaMatrixAllocator(blockSize, blockSize),
                                             4,
                                             htgs::MMType::Static,
                                             contexts);

      auto execPipeline =
          new htgs::ExecutionPipeline<MatrixRequestData, MatrixBlockData<double *>>(numGpus, subTaskGraph);
      auto decompositionRule = new MatrixDecompositionRule(numGpus);

      execPipeline->addInputRule(decompositionRule);

      taskGraph->addGraphInputConsumer(execPipeline);
      taskGraph->addEdge(execPipeline, matAccumBk);
      taskGraph->addRule(matAccumBk, accumTask, accumulateRule);
      taskGraph->addEdge(accumTask, matAccumBk);

      taskGraph->addRule(matAccumBk, outputTask, outputRule);
      taskGraph->addGraphOutputProducer(outputTask);

      taskGraph->incrementGraphInputProducer();

      taskGraph->writeDotToFile("cuda-graph.dot");

      htgs::Runtime *runtime = new htgs::Runtime(taskGraph);

      clk.start();

      runtime->executeRuntime();

      for (int col = 0; col < blkWidthMatA; col++) {
        for (int row = 0; row < blkHeightMatA; row++) {

          MatrixRequestData *matA = new MatrixRequestData(row, col, MatrixType::MatrixA);
          taskGraph->produceData(matA);

        }
      }

      for (int row = 0; row < blkHeightMatB; row++) {
        for (int col = 0; col < blkWidthMatB; col++) {

          MatrixRequestData *matB = new MatrixRequestData(row, col, MatrixType::MatrixB);
          taskGraph->produceData(matB);

        }
      }

      taskGraph->finishedProducingData();

      while (!taskGraph->isOutputTerminated()) {
        auto data = taskGraph->consumeData();
        if (data != nullptr) {
        }
      }

      runtime->waitForRuntime();
      clk.stopAndIncrement();

      delete runtime;
    }

    if (validate) {
      int res = validateResults(matrixC, matrixC_HTGS, matrixAHeight, matrixBWidth);
      if (res != 0) {
        std::cout << "Error validating test failed!" << std::endl;
      }
      else {
        std::cout << "Test PASSED" << std::endl;
      }

    }

    std::cout << (runSequential ? "sequential" : "htgs") << ", " << (runSequential ? numBlasThreads : numProdThreads)
              << ", width-b: " << matrixBWidth << ", height-a: " << matrixAHeight
              << ", shared-dim: " << sharedDim
              << ", " << ", blockSize: " << blockSize << ", time:" << clk.getAverageTime(TimeVal::MILLI)
              << std::endl;

    runtimeFile << (runSequential ? "sequential" : "htgs") << ", " << (runSequential ? numBlasThreads : numProdThreads)
                << ", "
                << matrixBWidth << ", " << matrixAHeight
                << ", " << sharedDim << ", " << blockSize << ", " << clk.getAverageTime(TimeVal::MILLI)
                << std::endl;

  }

  delete[] matrixA;
  delete[] matrixB;
  delete[] matrixC;

}