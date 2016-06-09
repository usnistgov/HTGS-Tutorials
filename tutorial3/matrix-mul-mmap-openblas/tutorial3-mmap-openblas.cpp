//
// Created by tjb3 on 2/23/16.
//
//#define DEBUG_FLAG
//#define DEBUG_LEVEL_VERBOSE
#include <htgs/api/TaskGraph.hpp>
#include <htgs/api/Runtime.hpp>
#include <string.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <fcntl.h>
#include <err.h>
#include <unistd.h>
#include <cblas.h>

#include "data/MatrixRequestData.h"
#include "data/MatrixBlockData.h"
#include "tasks/ReadMatrixTask.h"
#include "memory/MatrixAllocator.h"
#include "rules/MatrixLoadRule.h"
#include "tasks/MatrixMulBlkTask.h"
#include "tasks/MatrixAccumTask.h"
#include "tasks/OutputTask.h"
#include "../api/SimpleClock.h"
#include "rules/MatrixAccumulateRule.h"
#include "rules/MatrixDistributeRule.h"
#include "rules/MatrixLoopRule.h"
#include "rules/MatrixOutputRule.h"

int create_dir(std::string path) {
#ifdef __linux__
  int val = mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#else
  std::wstring wpath = std::wstring(path.begin(), path.end());
  std::wcout << " Creating folder " << wpath << std::endl;
  int val = _wmkdir(wpath.c_str());
#endif
  if (val == 0) {
    std::cout << path << " created successfully " << std::endl;
    return 0;
  }
  else {
    if (errno == EEXIST)
      return 0;

    std::cout << "Unable to create directory " << path <<": " << strerror(errno) << std::endl; // << val << " " << path << std::endl;
    return val;
  }

}
int validateResults(std::string baseDirectory, int fullMatrixAHeight, int fullMatrixBWidth)
{
  std::string fileName(baseDirectory + "/matrixC");
  std::string fileNamePar(baseDirectory + "/matrixCPar");

  double *cMem = new double[fullMatrixAHeight*fullMatrixBWidth];
  double *cMemPar = new double[fullMatrixAHeight*fullMatrixBWidth];

  std::ifstream c(fileName, std::ios::binary);
  std::ifstream cPar(fileNamePar, std::ios::binary);

  c.read((char *)cMem, sizeof(double)*fullMatrixAHeight*fullMatrixBWidth);
  cPar.read((char *)cMemPar, sizeof(double)*fullMatrixAHeight*fullMatrixBWidth);
  int count = 0;

  for (int i = 0; i < fullMatrixAHeight*fullMatrixBWidth; i++)
  {
    if (cMem[i] != cMemPar[i])
    {
      if (count <= 20) {
        std::cout << std::fixed << i << ": cMem = " << cMem[i] << " cMemPar = " << cMemPar[i] << std::endl;
      }
      count++;
    }
  }

  if (count > 0)
    std::cout << "Total incorrect = " << count <<std::endl;
  c.close();
  cPar.close();
  delete [] cMem;
  delete[] cMemPar;


  if (count > 0)
    return 1;
  return 0;
}

int validateResults2(std::string baseDirectory, int fullMatrixAHeight, int fullMatrixBWidth, double *result)
{
  std::string fileName(baseDirectory + "/matrixC");

  double *cMem = new double[fullMatrixAHeight*fullMatrixBWidth];

  std::ifstream c(fileName, std::ios::binary);

  c.read((char *)cMem, sizeof(double)*fullMatrixAHeight*fullMatrixBWidth);
  int count = 0;

  for (int i = 0; i < fullMatrixAHeight*fullMatrixBWidth; i++)
  {
    if (cMem[i] != result[i])
    {
      if (count <= 20) {
        std::cout << std::fixed << i << ": cMem = " << cMem[i] << " cMemPar = " << result[i] << std::endl;
      }
      count++;
    }
  }

  if (count > 0)
    std::cout << "Total incorrect = " << count <<std::endl;
  c.close();
  delete [] cMem;


  if (count > 0)
    return 1;
  return 0;
}

void computeSequentialMatMulNoMMap(std::string directory, int fullMatrixAHeight, int fullMatrixAWidth, int fullMatrixBWidth) {
  std::string matrixCFilename(directory + "/matrixC");
  std::string matrixAFilename(directory + "/matrixA");
  std::string matrixBFilename(directory + "/matrixB");

  std::ifstream matrixAFile(matrixAFilename, std::ios::binary);
  std::ifstream matrixBFile(matrixBFilename, std::ios::binary);
  std::ofstream matrixCFile(matrixCFilename, std::ios::binary);

  double *matrixA = new double[fullMatrixAHeight*fullMatrixAWidth];
  double *matrixB = new double[fullMatrixBWidth*fullMatrixAWidth];
  double *matrixC = new double[fullMatrixAHeight*fullMatrixBWidth];

  matrixAFile.read((char *)matrixA, sizeof(double)*fullMatrixAHeight*fullMatrixAWidth);
  matrixBFile.read((char *)matrixB, sizeof(double)*fullMatrixAWidth*fullMatrixBWidth);

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, fullMatrixAHeight, fullMatrixBWidth, fullMatrixAWidth, 1.0, matrixA, fullMatrixAWidth,
  matrixB, fullMatrixBWidth, 0.0, matrixC, fullMatrixBWidth);

//  for (int i = 0; i < fullMatrixAHeight; i++) {
//    for (int j = 0; j < fullMatrixBWidth; j++) {
//      double sum = 0.0;
//      for (int k = 0; k < fullMatrixAWidth; k++) {
//        sum += matrixA[i * fullMatrixAWidth + k] * matrixB[k * fullMatrixBWidth + j];
//      }
//      matrixC[i * fullMatrixBWidth + j] = sum;
//    }
//  }

  matrixCFile.write((char *)matrixC, sizeof(double)*fullMatrixAHeight*fullMatrixBWidth);

  matrixCFile.flush();

  delete [] matrixA;
  delete [] matrixB;
  delete [] matrixC;

  matrixAFile.close();
  matrixBFile.close();
  matrixCFile.close();
}

void computeSequentialMatMul(std::string directory, int fullMatrixAHeight, int fullMatrixAWidth, int fullMatrixBWidth) {
  std::string matrixCFilename(directory + "/matrixC");
  std::string matrixAFilename(directory + "/matrixA");
  std::string matrixBFilename(directory + "/matrixB");

  int fda = -1;
  int fdb = -1;
  int fdc = -1;

  if ((fda = open(matrixAFilename.c_str(), O_RDONLY)) == -1) {
    err(1, "open failed");
  }

  if ((fdb = open(matrixBFilename.c_str(), O_RDONLY)) == -1) {
    err(1, "open failed");
  }

  if ((fdc = open(matrixCFilename.c_str(), O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600)) == -1) {
    err(1, "write open failed");
  }

  // stretch the file to the size of the mmap
  if (lseek(fdc, fullMatrixAHeight*fullMatrixBWidth*sizeof(double)-1, SEEK_SET) == -1)
  {
    close(fdc);
    err(2, "Error using lseek to stretch the file");
  }

  // Write at end to ensure file has the correct size
  if (write(fdc, "", 1) == -1)
  {
    close(fdc);
    err(3, "Error writing to complete stretching the file");
  }

  double *matrixA =
      (double *) mmap(NULL, fullMatrixAHeight * fullMatrixAWidth * sizeof(double), PROT_READ, MAP_SHARED, fda, 0);
  double *matrixB =
      (double *) mmap(NULL, fullMatrixAWidth * fullMatrixBWidth * sizeof(double), PROT_READ, MAP_SHARED, fdb, 0);
  double *matrixC =
      (double *)mmap(NULL, fullMatrixAHeight*fullMatrixBWidth * sizeof(double), PROT_WRITE, MAP_SHARED, fdc, 0);

//  memset(matrixC, 0.0, sizeof(double)*fullMatrixAHeight*fullMatrixBWidth);

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, fullMatrixAHeight, fullMatrixBWidth, fullMatrixAWidth, 1.0, matrixA, fullMatrixAWidth,
              matrixB, fullMatrixBWidth, 0.0, matrixC, fullMatrixBWidth);

  if (msync(matrixC, fullMatrixAHeight*fullMatrixBWidth*sizeof(double), MS_SYNC) == -1)
  {
    err(5, "Could not sync the file to disk");
  }

  munmap(matrixA, fullMatrixAHeight*fullMatrixAWidth*sizeof(double));
  munmap(matrixB, fullMatrixBWidth*fullMatrixAWidth*sizeof(double));
  munmap(matrixC, fullMatrixAHeight*fullMatrixBWidth*sizeof(double));

  close(fda);
  close(fdb);
  close(fda);

}

int main(int argc, char *argv[])
{
  int dim = 2048;
  int blockSize = 256;
  int numReadThreads = 1;
  int numProdThreads = 40;
  int numBlasThreads = 40;
  bool runSequential = false;
  std::string runtimeFileStr("runtimes");
  int numRetry = 1;

  if (argc > 1)
  {
    for (int arg = 1; arg < argc; arg++)
    {
      std::string argStr(argv[arg]);

      if (argStr == "--dim" && arg+1 < argc)
      {
        dim = atoi(argv[arg+1]);
        arg++;
      }

      if (argStr == "--seq")
      {
        runSequential = true;
      }

      if (argStr == "--num-retry" && arg+1 < argc)
      {
        numRetry = atoi(argv[arg+1]);
        arg++;
      }

      if (argStr == "--blk-size" && arg+1 < argc)
      {
        blockSize = atoi(argv[arg+1]);
        arg++;
      }

      if (argStr == "--num-threads-htgs" && arg+1 < argc)
      {
        numProdThreads = atoi(argv[arg+1]);
        arg++;
      }

      if (argStr == "--num-threads-blas" && arg+1 < argc)
      {
        numBlasThreads = atoi(argv[arg+1]);
        arg++;
      }

      if (argStr == "--runtime-file" && arg+1 < argc)
      {
        runtimeFileStr = argv[arg+1];
        arg++;
      }

      if (argStr == "--help")
      {
        std::cout << "Usage: " << argv[0] <<" --dim <n> --seq --num-retry <n> --blk-size <n> --num-threads-htgs <n> --num-thread-blas <n> --runtime-file <file>" << std::endl;
        exit(0);
      }
    }
  }

  int matrixAHeight = dim;
  int matrixBWidth = dim;
  int matrixCDim = dim;
  std::string directory("data/tutorial2/" + std::to_string(matrixAHeight) + "x" + std::to_string(matrixCDim));

  std::ofstream runtimeFile(runtimeFileStr, std::ios::app);

    for (int numTry = 0; numTry < numRetry; numTry++) {
      SimpleClock clk;
      if (runSequential) {
        openblas_set_num_threads(numBlasThreads);

        clk.start();
        computeSequentialMatMul(directory, matrixAHeight, matrixCDim, matrixBWidth);
        clk.stopAndIncrement();
      }
      else {
        openblas_set_num_threads(1);

        ReadMatrixTask *readAMatTask =
            new ReadMatrixTask(numReadThreads,
                               MatrixType::MatrixA,
                               blockSize,
                               matrixCDim,
                               matrixAHeight,
                               directory,
                               "A");
        ReadMatrixTask *readBMatTask =
            new ReadMatrixTask(numReadThreads,
                               MatrixType::MatrixB,
                               blockSize,
                               matrixBWidth,
                               matrixCDim,
                               directory,
                               "B");
        MatrixMulBlkTask *mmulTask =
            new MatrixMulBlkTask(numProdThreads, matrixCDim, matrixAHeight, matrixBWidth, matrixCDim, blockSize);
        MatrixAccumTask *accumTask = new MatrixAccumTask((int)ceil((double)numProdThreads / 2.0));

        OutputTask *outputTask = new OutputTask(directory, matrixBWidth, matrixAHeight, blockSize);

        int blkHeightMatB = readBMatTask->getNumBlocksRows();
        int blkWidthMatB = readBMatTask->getNumBlocksCols();

        int blkHeightMatA = readAMatTask->getNumBlocksRows();
        int blkWidthMatA = readAMatTask->getNumBlocksCols();

        MatrixDistributeRule *distributeRuleMatA = new MatrixDistributeRule(MatrixType::MatrixA);
        MatrixDistributeRule *distributeRuleMatB = new MatrixDistributeRule(MatrixType::MatrixB);

        MatrixLoadRule *loadRule = new MatrixLoadRule(blkWidthMatA, blkHeightMatA, blkWidthMatB, blkHeightMatB);
        MatrixAccumulateRule *accumulateRule = new MatrixAccumulateRule(blkWidthMatB, blkHeightMatA, blkWidthMatA);

        MatrixOutputRule *outputRule = new MatrixOutputRule(blkWidthMatB, blkHeightMatA, blkWidthMatA);

        auto distributeBk = new htgs::Bookkeeper<MatrixRequestData>();
        auto matMulBk = new htgs::Bookkeeper<MatrixBlockData<double *>>();
        auto matAccumBk = new htgs::Bookkeeper<MatrixBlockData<double *>>();

        auto taskGraph = new htgs::TaskGraph<MatrixRequestData, MatrixBlockData<double *>>();

        taskGraph->addGraphInputConsumer(distributeBk);
        taskGraph->addRule(distributeBk, readAMatTask, distributeRuleMatA);
        taskGraph->addRule(distributeBk, readBMatTask, distributeRuleMatB);

        taskGraph->addEdge(readAMatTask, matMulBk);
        taskGraph->addEdge(readBMatTask, matMulBk);

        taskGraph->addRule(matMulBk, mmulTask, loadRule);

        taskGraph->addEdge(mmulTask, matAccumBk);
        taskGraph->addRule(matAccumBk, accumTask, accumulateRule);
        taskGraph->addEdge(accumTask, matAccumBk);

        taskGraph->addRule(matAccumBk, outputTask, outputRule);
        taskGraph->addGraphOutputProducer(outputTask);

        taskGraph->incrementGraphInputProducer();

        htgs::Runtime *runtime = new htgs::Runtime(taskGraph);

        clk.start();

        runtime->executeRuntime();

        for (int row = 0; row < std::max(blkHeightMatA, blkHeightMatB); row++) {
          for (int col = 0; col < std::max(blkWidthMatA, blkWidthMatB); col++) {
            if (row < blkHeightMatA && col < blkWidthMatA) {
              MatrixRequestData *matrixA = new MatrixRequestData(row, col, MatrixType::MatrixA);
              taskGraph->produceData(matrixA);
            }

            if (row < blkHeightMatB && col < blkWidthMatB) {
              MatrixRequestData *matrixB = new MatrixRequestData(row, col, MatrixType::MatrixB);
              taskGraph->produceData(matrixB);
            }
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
        int res = validateResults(directory, matrixAHeight, matrixBWidth);

        if (res != 0)
        {
          std::cout << "Error validation test failed!" << std::endl;
        }
        delete runtime;

      }

      std::cout << (runSequential ? "sequential" : "htgs") << ", " << (runSequential ? numBlasThreads : numProdThreads) << ", " << dim << ", " << (runSequential ? 0 : blockSize) << ", " << clk.getAverageTime(TimeVal::MILLI) << std::endl;
      runtimeFile << (runSequential ? "sequential" : "htgs") << ", " << (runSequential ? numBlasThreads : numProdThreads) << ", " << dim << ", " << blockSize << ", " << clk.getAverageTime(TimeVal::MILLI) << std::endl;


    }
}