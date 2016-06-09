//
// Created by tjb3 on 2/23/16.
//
//#define DEBUG_FLAG
//#define DEBUG_LEVEL_VERBOSE

#define PROFILE

#include <htgs/api/TaskGraph.hpp>
#include <htgs/api/Runtime.hpp>
#include <string.h>
#include <sys/stat.h>
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
#include "../matrix-ops/utils/util-matrix.h"

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
      return 1;

    std::cout << "Unable to create directory " << path <<": " << strerror(errno) << std::endl; // << val << " " << path << std::endl;
    return val;
  }

}
int validateResults(std::string baseDirectory, int matrixAHeight, int matrixBWidth, int blockSize)
{
  int blkHeightMatA = (int)ceil((double)matrixAHeight / (double)blockSize);
  int blkWidthMatB = (int)ceil((double)matrixBWidth / (double)blockSize);

  for (int row = 0; row < blkHeightMatA; row++)
  {
    for (int col = 0; col < blkWidthMatB; col++)
    {
      int matrixAHeight = (row == blkHeightMatA-1 && matrixAHeight % blockSize != 0) ? matrixAHeight % blockSize : blockSize;
      int matrixBWidth = (col == blkWidthMatB-1 && matrixBWidth % blockSize != 0) ? matrixBWidth % blockSize : blockSize;
      std::string fileName(baseDirectory + "/matrixC/" + std::to_string(row) + "_" + std::to_string(col));
      std::string fileNamePar(baseDirectory + "/matrixCPar/" + std::to_string(row) + "_" + std::to_string(col));

      std::ifstream c(fileName, std::ios::binary);
      std::ifstream cPar(fileNamePar, std::ios::binary);

      double *cMem = new double[matrixAHeight*matrixBWidth];
      double *cMemPar = new double[matrixAHeight*matrixBWidth];

      c.read((char *)cMem, sizeof(double)*matrixAHeight*matrixBWidth);
      cPar.read((char *)cMemPar, sizeof(double)*matrixAHeight*matrixBWidth);

      int count = 0;
      for (int i = 0; i < matrixAHeight*matrixBWidth; i++)
      {
        if (cMem[i] != cMemPar[i]) {
          std::cout << i << ": cMem = " << cMem[i] << " cMemPar = " << cMemPar[i] << std::endl;
          if (count == 20) {
            return -1;
          }
           count++;
        }
      }

      delete [] cMem;
      delete [] cMemPar;


    }
  }

  return 0;
}

void computeSequentialMatMul(std::string directory, int fullMatrixAHeight, int fullMatrixAWidth, int fullMatrixBWidth, int blockSize)
{
  std::string matrixCDir(directory + "/matrixC");
  create_dir(matrixCDir);

  int blkHeightMatA = (int)ceil((double)fullMatrixAHeight / (double)blockSize);
  int blkWidthMatA = (int)ceil((double)fullMatrixAWidth / (double)blockSize);
  int blkHeightMatB = blkWidthMatA;
  int blkWidthMatB = (int)ceil((double)fullMatrixBWidth / (double)blockSize);

  double ***matALookup = new double**[blkHeightMatA];
  for (int i = 0; i < blkHeightMatA; i++)
  {
    matALookup[i] = new double*[blkWidthMatA];
  }

  double ***matBLookup = new double**[blkHeightMatB];
  for (int i = 0; i < blkHeightMatB; i++)
  {
    matBLookup[i] = new double*[blkWidthMatB];
  }

  for (int i = 0; i < blkHeightMatA; i++)
  {
    for (int j = 0; j < blkWidthMatA; j++)
    {
      matALookup[i][j] = nullptr;
    }
  }

  for (int i = 0; i < blkHeightMatB; i++)
  {
    for (int j = 0; j < blkWidthMatB; j++)
    {
      matBLookup[i][j] = nullptr;
    }
  }



  for (int blkRowA = 0; blkRowA < blkHeightMatA; blkRowA++)
  {
    for (int blkColB = 0; blkColB < blkWidthMatB; blkColB++)
    {
      int matrixAHeight = (blkRowA == blkHeightMatA-1 && fullMatrixAHeight % blockSize != 0) ? fullMatrixAHeight % blockSize : blockSize;
      int matrixBWidth = (blkColB == blkWidthMatB-1 && fullMatrixBWidth % blockSize != 0) ? fullMatrixBWidth % blockSize : blockSize;

      std::string matrixCFile(matrixCDir + "/" + std::to_string(blkRowA) + "_" + std::to_string(blkColB));

      double *finalResultC = new double[matrixAHeight*matrixBWidth];
      memset(finalResultC, 0, sizeof(double) *matrixAHeight * matrixBWidth);
      // matrix C . . .
      for (int blk = 0; blk < blkWidthMatA; blk++)
      {
        // Read A and B
        int matrixAWidth = (blk == blkWidthMatA-1 && fullMatrixAWidth % blockSize != 0) ? fullMatrixAWidth % blockSize : blockSize;

        double *matrixA;
        double *matrixB;
        if (matALookup[blkRowA][blk] == nullptr)
        {
          matrixA = new double[matrixAHeight*matrixAWidth];
          std::string matrixAFile(directory + "/matrixA/" + std::to_string(blkRowA) + "_" + std::to_string(blk));
          std::ifstream fileA(matrixAFile, std::ios::binary);
          fileA.read((char *)matrixA, sizeof(double) * matrixAHeight * matrixAWidth);
          matALookup[blkRowA][blk] = matrixA;
        }
        else
        {
          matrixA = matALookup[blkRowA][blk];
        }


        if (matBLookup[blk][blkColB] == nullptr)
        {
          matrixB = new double[matrixBWidth*matrixAWidth];
          std::string matrixBFile(directory + "/matrixB/" + std::to_string(blk) + "_" + std::to_string(blkColB));
          std::ifstream fileB(matrixBFile, std::ios::binary);
          fileB.read((char *)matrixB, sizeof(double) * matrixBWidth * matrixAWidth);
          matBLookup[blk][blkColB] = matrixB;
        }
        else
        {
          matrixB = matBLookup[blk][blkColB];
        }

//        std::cout << "Seq Computing A(" << blkRowA << ", " << blk << ") x B(" << blk << ", " << blkColB << ") = C(" <<blkRowA << ", "<< blkColB << ")" <<std::endl;

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, matrixAHeight, matrixBWidth, matrixAWidth, 1.0, matrixA, matrixAWidth,
                    matrixB, matrixBWidth, 1.0, finalResultC, matrixBWidth);

      }

      std::ofstream fileC(matrixCFile, std::ios::binary);
      fileC.write((char *)finalResultC, sizeof(double) * matrixAHeight * matrixBWidth);

      delete [] finalResultC;
    }
  }

  for (int i = 0; i < blkHeightMatA; i++)
  {
    for (int j = 0; j < blkWidthMatA; j++)
    {
      delete [] matALookup[i][j];
    }
    delete [] matALookup[i];
  }

  delete []matALookup;

  for (int i = 0; i < blkHeightMatB; i++)
  {
    for (int j = 0; j < blkWidthMatB; j++)
    {
      delete [] matBLookup[i][j];
    }
    delete [] matBLookup[i];
  }

  delete []matBLookup;
}

int main(int argc, char *argv[])
{
  int dim = 16384;
  int blockSize = 4096;
  int numReadThreads = 1;
  int numProdThreads = 16;
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
  int matrixAWidth = dim;

  std::string baseDirectory("data/blocked-matrix");

  std::string directory = generateDirectoryName(baseDirectory, matrixAHeight,matrixBWidth, blockSize);

  generateMatrixFiles2(baseDirectory, BlockType::MatrixA, matrixAWidth, matrixAHeight, blockSize);
  generateMatrixFiles2(baseDirectory, BlockType::MatrixB, matrixBWidth, matrixAWidth, blockSize);

  std::string outputDirectory(directory + "/matrixCPar");

  create_dir(outputDirectory);
  std::ofstream runtimeFile(runtimeFileStr, std::ios::app);

  for (int numTry = 0; numTry < numRetry; numTry++) {
    SimpleClock clk;

    if (runSequential) {

      openblas_set_num_threads(numBlasThreads);


      clk.start();
      computeSequentialMatMul(directory, matrixAHeight, matrixAWidth, matrixBWidth, blockSize);
      clk.stopAndIncrement();
    }
    else {
      openblas_set_num_threads(1);

      ReadMatrixTask
          *readAMatTask = new ReadMatrixTask(numReadThreads, blockSize, matrixAWidth, matrixAHeight, directory, "A");
      ReadMatrixTask
          *readBMatTask = new ReadMatrixTask(numReadThreads, blockSize, matrixBWidth, matrixAWidth, directory, "B");
      MatrixMulBlkTask *mmulTask = new MatrixMulBlkTask(numProdThreads);
      MatrixAccumTask *accumTask = new MatrixAccumTask((int) ceil((double) numProdThreads / 2.0));

      OutputTask *outputTask = new OutputTask(outputDirectory);

      int blkHeightMatB = readBMatTask->getNumBlocksRows();
      int blkWidthMatB = readBMatTask->getNumBlocksCols();

      int blkHeightMatA = readAMatTask->getNumBlocksRows();
      int blkWidthMatA = readAMatTask->getNumBlocksCols();

      MatrixDistributeRule *distributeRuleMatA = new MatrixDistributeRule(MatrixType::MatrixA);
      MatrixDistributeRule *distributeRuleMatB = new MatrixDistributeRule(MatrixType::MatrixB);

      MatrixLoadRule *loadRule = new MatrixLoadRule(blkWidthMatA, blkHeightMatA, blkWidthMatB, blkHeightMatB);
      MatrixAccumulateRule *accumulateRule = new MatrixAccumulateRule(blkWidthMatB, blkHeightMatA, blkWidthMatA);

      MatrixLoopRule *loopRuleMatA = new MatrixLoopRule(0);//blkWidthMatA);
      MatrixLoopRule *loopRuleMatB = new MatrixLoopRule(0);//blkHeightMatB);

      MatrixOutputRule *outputRule = new MatrixOutputRule(blkWidthMatB, blkHeightMatA, blkWidthMatA);

      auto distributeBk = new htgs::Bookkeeper<MatrixRequestData>();
      auto matMulBk = new htgs::Bookkeeper<MatrixBlockData<MatrixMemoryData_t>>();
      auto matAccumBk = new htgs::Bookkeeper<MatrixBlockData<double *>>();


      auto taskGraph = new htgs::TaskGraph<MatrixRequestData, MatrixRequestData>();

      taskGraph->addGraphInputConsumer(distributeBk);
      taskGraph->addRule(distributeBk, readAMatTask, distributeRuleMatA);
      taskGraph->addRule(distributeBk, readBMatTask, distributeRuleMatB);

      taskGraph->addEdge(readAMatTask, matMulBk);
      taskGraph->addEdge(readBMatTask, matMulBk);

      taskGraph->addRule(matMulBk, mmulTask, loadRule);
      taskGraph->addRule(matMulBk, readAMatTask, loopRuleMatA);
      taskGraph->addRule(matMulBk, readBMatTask, loopRuleMatB);

      taskGraph->addEdge(mmulTask, matAccumBk);
      taskGraph->addRule(matAccumBk, accumTask, accumulateRule);
      taskGraph->addEdge(accumTask, matAccumBk);

      taskGraph->addRule(matAccumBk, outputTask, outputRule);
      taskGraph->addGraphOutputProducer(outputTask);

      taskGraph->addMemoryManagerEdge("matrixA",
                                      readAMatTask,
                                      mmulTask,
                                      new MatrixAllocator(blockSize, blockSize),
                                      1000,
                                      htgs::MMType::Static);
      taskGraph->addMemoryManagerEdge("matrixB",
                                      readBMatTask,
                                      mmulTask,
                                      new MatrixAllocator(blockSize, blockSize),
                                      1000,
                                      htgs::MMType::Static);

      taskGraph->incrementGraphInputProducer();

      htgs::Runtime *runtime = new htgs::Runtime(taskGraph);

      clk.start();

      runtime->executeRuntime();

      for (int row = 0; row < blkHeightMatB; row++) {
        for (int col = 0; col < blkWidthMatB; col++) {
          MatrixRequestData *matrixB = new MatrixRequestData(row, col, MatrixType::MatrixB);
          taskGraph->produceData(matrixB);
        }
      }

      for (int col = 0; col < blkWidthMatA; col++)
      {
        for (int row = 0; row < blkHeightMatA; row++)
        {
          MatrixRequestData *matrixA = new MatrixRequestData(row, col, MatrixType::MatrixA);
          taskGraph->produceData(matrixA);
        }
      }


      taskGraph->finishedProducingData();

      while (!taskGraph->isOutputTerminated()) {
        auto data = taskGraph->consumeData();
        if (data != nullptr) {
//      std::cout << "Result received: " << data->getRow() << ", " << data->getCol() <<std::endl;
        }
      }

      runtime->waitForRuntime();

      clk.stopAndIncrement();

      int res = validateResults(directory, matrixAHeight, matrixBWidth, blockSize);
      if (res != 0) {
        std::cout << "Error validation test failed!" << std::endl;
      }
      delete runtime;

    }


    std::cout << (runSequential ? "sequential" : "htgs") << ", " << (runSequential ? numBlasThreads : numProdThreads)
        << ", " << dim << ", " << (runSequential ? 0 : blockSize) << ", " << clk.getAverageTime(TimeVal::MILLI)
        << std::endl;
    runtimeFile << (runSequential ? "sequential" : "htgs") << ", " << (runSequential ? numBlasThreads : numProdThreads) << ", " << dim << ", " << blockSize << ", " << clk.getAverageTime(TimeVal::MILLI) << std::endl;

  }

//  std::cout << "Finished (" << (res != 0 ? "FAILED" : "PASSED") << ") running parallel in " << clk.getAverageTime(TimeVal::MILLI) << " ms   sequential in " << clkSeq.getAverageTime(TimeVal::MILLI) << "ms " << (clkSeq.getAverageTime(TimeVal::MILLI) / clk.getAverageTime(TimeVal::MILLI)) << "x speedup" << std::endl;



}