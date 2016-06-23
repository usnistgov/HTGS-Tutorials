
// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
// NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
// You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

//
// Created by tjb3 on 2/23/16.
//
//#define DEBUG_FLAG
//#define DEBUG_LEVEL_VERBOSE
#include <htgs/api/TaskGraph.hpp>
#include <htgs/api/Runtime.hpp>
#include <sys/mman.h>
#include <fcntl.h>
#include <err.h>
#include <unistd.h>

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
#include "../../tutorial-utils/SimpleClock.h"
#include "../../tutorial-utils/util-matrix.h"
#include "../../tutorial-utils/util-filesystem.h"

int validateResults(std::string baseDirectory, int fullMatrixAHeight, int fullMatrixBWidth) {
  std::string fileName(baseDirectory + "/matrixC");
  std::string fileNamePar(baseDirectory + "/matrixC_HTGS");

  double *cMem = new double[fullMatrixAHeight * fullMatrixBWidth];
  double *cMemPar = new double[fullMatrixAHeight * fullMatrixBWidth];

  std::ifstream c(fileName, std::ios::binary);
  std::ifstream cPar(fileNamePar, std::ios::binary);

  c.read((char *) cMem, sizeof(double) * fullMatrixAHeight * fullMatrixBWidth);
  cPar.read((char *) cMemPar, sizeof(double) * fullMatrixAHeight * fullMatrixBWidth);
  int count = 0;

  for (int i = 0; i < fullMatrixAHeight * fullMatrixBWidth; i++) {
    if (cMem[i] != cMemPar[i]) {
      std::cout << std::fixed << i << ": cMem = " << cMem[i] << " cMemPar = " << cMemPar[i] << std::endl;
      if (count == 20) {
        return -1;
      }
      count++;
    }
  }

  return 0;
}

void computeSequentialMatMulNoMMap(std::string directoryA,
                                   std::string directoryB,
                                   std::string outputDirectory,
                                   int fullMatrixAHeight,
                                   int fullMatrixAWidth,
                                   int fullMatrixBWidth) {
  std::string matrixCFilename(outputDirectory + "/matrixC");
  std::string matrixAFilename(directoryA + "/MatrixA");
  std::string matrixBFilename(directoryB + "/MatrixB");

  std::ifstream matrixAFile(matrixAFilename, std::ios::binary);
  std::ifstream matrixBFile(matrixBFilename, std::ios::binary);
  std::ofstream matrixCFile(matrixCFilename, std::ios::binary);

  double *matrixA = new double[fullMatrixAHeight * fullMatrixAWidth];
  double *matrixB = new double[fullMatrixBWidth * fullMatrixAWidth];
  double *matrixC = new double[fullMatrixAHeight * fullMatrixBWidth];

  matrixAFile.read((char *) matrixA, sizeof(double) * fullMatrixAHeight * fullMatrixAWidth);
  matrixBFile.read((char *) matrixB, sizeof(double) * fullMatrixAWidth * fullMatrixBWidth);

  for (int i = 0; i < fullMatrixAHeight; i++) {
    for (int j = 0; j < fullMatrixBWidth; j++) {
      double sum = 0.0;
      for (int k = 0; k < fullMatrixAWidth; k++) {
        sum += matrixA[i * fullMatrixAWidth + k] * matrixB[k * fullMatrixBWidth + j];
      }
      matrixC[i * fullMatrixBWidth + j] = sum;
    }
  }

  matrixCFile.write((char *) matrixC, sizeof(double) * fullMatrixAHeight * fullMatrixBWidth);

}

void computeSequentialMatMul(std::string directoryA,
                             std::string directoryB,
                             std::string outputDirectory,
                             int fullMatrixAHeight,
                             int fullMatrixAWidth,
                             int fullMatrixBWidth) {
  std::string matrixCFilename(outputDirectory + "/matrixC");
  std::string matrixAFilename(directoryA + "/MatrixA");
  std::string matrixBFilename(directoryB + "/MatrixB");

  int fda = -1;
  int fdb = -1;
  int fdc = -1;

  if ((fda = open(matrixAFilename.c_str(), O_RDONLY)) == -1) {
    err(1, "open failed");
  }

  if ((fdb = open(matrixBFilename.c_str(), O_RDONLY)) == -1) {
    err(1, "open failed");
  }

  if ((fdc = open(matrixCFilename.c_str(), O_RDWR | O_CREAT | O_TRUNC, (mode_t) 0600)) == -1) {
    err(1, "write open failed");
  }

  // stretch the file to the size of the mmap
  if (lseek(fdc, fullMatrixAHeight * fullMatrixBWidth * sizeof(double) - 1, SEEK_SET) == -1) {
    close(fdc);
    err(2, "Error using lseek to stretch the file");
  }

  // Write at end to ensure file has the correct size
  if (write(fdc, "", 1) == -1) {
    close(fdc);
    err(3, "Error writing to complete stretching the file");
  }

  double *matrixA =
      (double *) mmap(NULL, fullMatrixAHeight * fullMatrixAWidth * sizeof(double), PROT_READ, MAP_SHARED, fda, 0);
  double *matrixB =
      (double *) mmap(NULL, fullMatrixAWidth * fullMatrixBWidth * sizeof(double), PROT_READ, MAP_SHARED, fdb, 0);
  double *matrixC =
      (double *) mmap(NULL, fullMatrixAHeight * fullMatrixBWidth * sizeof(double), PROT_WRITE, MAP_SHARED, fdc, 0);

  for (int i = 0; i < fullMatrixAHeight; i++) {
    for (int j = 0; j < fullMatrixBWidth; j++) {
      double sum = 0.0;
      for (int k = 0; k < fullMatrixAWidth; k++) {
        sum += matrixA[i * fullMatrixAWidth + k] * matrixB[k * fullMatrixBWidth + j];
      }
      matrixC[i * fullMatrixBWidth + j] = sum;
    }
  }

  if (msync(matrixC, fullMatrixAHeight * fullMatrixBWidth * sizeof(double), MS_SYNC) == -1) {
    err(5, "Could not sync the file to disk");
  }

  munmap(matrixA, fullMatrixAHeight * fullMatrixAWidth * sizeof(double));
  munmap(matrixB, fullMatrixBWidth * fullMatrixAWidth * sizeof(double));
  munmap(matrixC, fullMatrixAHeight * fullMatrixBWidth * sizeof(double));

}

int main(int argc, char *argv[]) {
  int matrixAHeight = 1024;
  int matrixBWidth = 1024;
  int sharedDim = 1024;
  int blockSize = 256;
  int numReadThreads = 1;
  int numProdThreads = 20;
  std::string directory("data");
  std::string outputDirectory(directory);
  bool validate = false;
  bool runSequential = false;

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

    if (argvs == "--block-size") {
      arg++;
      blockSize = atoi(argv[arg]);
    }

    if (argvs == "--num-readers") {
      arg++;
      numReadThreads = atoi(argv[arg]);
    }

    if (argvs == "--num-workers") {
      arg++;
      numProdThreads = atoi(argv[arg]);
    }

    if (argvs == "--dir") {
      arg++;
      directory = argv[arg];
    }

    if (argvs == "--output-dir") {
      arg++;
      outputDirectory = argv[arg];
    }

    if (argvs == "--validate-results") {
      validate = true;
    }

    if (argvs == "--run-sequential") {
      runSequential = true;
    }

    if (argvs == "--help") {
      std::cout << argv[0]
                << " args: [--width-b <#>] [--height-a <#>] [--shared-dim <#>] [--block-size <#>] [--num-readers <#>] [--num-workers <#>] [--dir <dir>] [--output-dir <dir>] [--validate-results] [--run-sequential] [--help]"
                << std::endl;
      exit(0);
    }
  }

  if (!has_dir(outputDirectory))
    create_dir(outputDirectory);

  checkAndValidateMatrixFiles(directory, sharedDim, matrixAHeight, matrixBWidth, sharedDim);

  std::string inputDirectoryA = generateDirectoryName(directory, sharedDim, matrixAHeight);
  std::string inputDirectoryB = generateDirectoryName(directory, matrixBWidth, sharedDim);
  outputDirectory = generateDirectoryName(outputDirectory, matrixBWidth, matrixAHeight);

  if (!has_dir(outputDirectory))
    create_dir(outputDirectory);

  SimpleClock clk;

  if (runSequential) {
    clk.start();
    computeSequentialMatMulNoMMap(inputDirectoryA,
                                  inputDirectoryB,
                                  outputDirectory,
                                  matrixAHeight,
                                  sharedDim,
                                  matrixBWidth);
    clk.stopAndIncrement();
  }
  else {

    ReadMatrixTask *readAMatTask =
        new ReadMatrixTask(numReadThreads,
                           MatrixType::MatrixA,
                           blockSize,
                           sharedDim,
                           matrixAHeight,
                           inputDirectoryA,
                           "A");
    ReadMatrixTask *readBMatTask =
        new ReadMatrixTask(numReadThreads,
                           MatrixType::MatrixB,
                           blockSize,
                           matrixBWidth,
                           sharedDim,
                           inputDirectoryB,
                           "B");
    MatrixMulBlkTask
        *mmulTask = new MatrixMulBlkTask(numProdThreads, sharedDim, matrixAHeight, matrixBWidth, sharedDim, blockSize);
    MatrixAccumTask *accumTask = new MatrixAccumTask(numProdThreads / 2);

    OutputTask *outputTask = new OutputTask(outputDirectory, matrixBWidth, matrixAHeight, blockSize);

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

    auto taskGraph = new htgs::TaskGraph<MatrixRequestData, MatrixRequestData>();

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

    taskGraph->writeDotToFile("matMul.dot");

    taskGraph->incrementGraphInputProducer();

    htgs::Runtime *runtime = new htgs::Runtime(taskGraph);

    clk.start();

    runtime->executeRuntime();

    for (int row = 0; row < blkHeightMatA; row++) {
      for (int col = 0; col < blkWidthMatA; col++) {

        MatrixRequestData *matrixA = new MatrixRequestData(row, col, MatrixType::MatrixA);
        taskGraph->produceData(matrixA);

      }
    }

    for (int row = 0; row < blkHeightMatB; row++) {
      for (int col = 0; col < blkWidthMatB; col++) {

        MatrixRequestData *matrixB = new MatrixRequestData(row, col, MatrixType::MatrixB);
        taskGraph->produceData(matrixB);

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

    delete runtime;
  }

  if (validate) {
    int res = validateResults(outputDirectory, matrixAHeight, matrixBWidth);
    std::cout << "Finished (" << (res != 0 ? "FAILED - must run sequential" : "PASSED") << ") ";
  }

  std::cout << (runSequential ? "Sequential, " : "Parallel, ")
            << "width-b: " << matrixBWidth << ", height-a: " << matrixAHeight
            << ", shared-dim: " << sharedDim << ", blocksize: " << blockSize << ", time: " <<
            clk.getAverageTime(TimeVal::MILLI)
            << std::endl;

}