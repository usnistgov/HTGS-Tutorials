
// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
// NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
// You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

//
// Created by tjb3 on 2/23/16.
//
//#define DEBUG_FLAG
//#define DEBUG_LEVEL_VERBOSE
#include <htgs/api/TaskGraphConf.hpp>
#include <htgs/api/TaskGraphRuntime.hpp>
#include <sys/mman.h>
#include <fcntl.h>
#include <err.h>
#include <unistd.h>

#include "../../tutorial-utils/SimpleClock.h"
#include "../../tutorial-utils/util-matrix.h"
#include "../../tutorial-utils/util-filesystem.h"
#include "../../tutorial-utils/matrix-library/operations/matmul.h"
#include "../../tutorial-utils/matrix-library/args/MatMulArgs.h"
#include "tasks/ReadMatrixTaskMMap.h"
#include "../tasks/MatMulBlkTask.h"
#include "../tasks/MatMulAccumTask.h"
#include "tasks/MatMulOutputTaskWithMMap.h"
#include "../rules/MatMulDistributeRule.h"
#include "../rules/MatMulLoadRule.h"
#include "../rules/MatMulAccumulateRule.h"

int validateResults(std::string baseDirectory, size_t fullMatrixAHeight, size_t fullMatrixBWidth) {
  std::string fileName(baseDirectory + "/matrixC");
  std::string fileNamePar(baseDirectory + "/matrixC_HTGS");

  double *cMem = new double[fullMatrixAHeight * fullMatrixBWidth];
  double *cMemPar = new double[fullMatrixAHeight * fullMatrixBWidth];

  std::ifstream c(fileName, std::ios::binary);
  std::ifstream cPar(fileNamePar, std::ios::binary);

  c.read((char *) cMem, sizeof(double) * fullMatrixAHeight * fullMatrixBWidth);
  cPar.read((char *) cMemPar, sizeof(double) * fullMatrixAHeight * fullMatrixBWidth);

  if (!validateMatMulResults(20, cMem, cMemPar, fullMatrixAHeight*fullMatrixBWidth))
  {
    delete []cMem;
    delete []cMemPar;
    return -1;
  }


  delete []cMem;
  delete []cMemPar;
  return 0;
}

void computeSequentialMatMulNoMMap(std::string directoryA,
                                   std::string directoryB,
                                   std::string outputDirectory,
                                   size_t fullMatrixAHeight,
                                   size_t fullMatrixAWidth,
                                   size_t fullMatrixBWidth) {
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

  computeMatMul(fullMatrixAHeight, fullMatrixBWidth, fullMatrixAWidth, 1.0, matrixA, fullMatrixAWidth, matrixB, fullMatrixBWidth, 0.0, matrixC, fullMatrixBWidth, false);

  matrixCFile.write((char *) matrixC, sizeof(double) * fullMatrixAHeight * fullMatrixBWidth);

}

void computeSequentialMatMul(std::string directoryA,
                             std::string directoryB,
                             std::string outputDirectory,
                             size_t fullMatrixAHeight,
                             size_t fullMatrixAWidth,
                             size_t fullMatrixBWidth) {
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

  computeMatMul(fullMatrixAHeight, fullMatrixBWidth, fullMatrixAWidth, 1.0, matrixA, fullMatrixAWidth, matrixB, fullMatrixBWidth, 0.0, matrixC, fullMatrixBWidth, false);

  if (msync(matrixC, fullMatrixAHeight * fullMatrixBWidth * sizeof(double), MS_SYNC) == -1) {
    err(5, "Could not sync the file to disk");
  }

  munmap(matrixA, fullMatrixAHeight * fullMatrixAWidth * sizeof(double));
  munmap(matrixB, fullMatrixBWidth * fullMatrixAWidth * sizeof(double));
  munmap(matrixC, fullMatrixAHeight * fullMatrixBWidth * sizeof(double));

}

int main(int argc, char *argv[]) {
  MatMulArgs matMulArgs;
  matMulArgs.processArgs(argc, argv);

  size_t matrixAHeight = matMulArgs.getMatrixAHeight();
  size_t matrixBWidth = matMulArgs.getMatrixBWidth();
  size_t sharedDim = matMulArgs.getSharedDim();

  size_t blockSize = matMulArgs.getBlockSize();
  size_t numReadThreads = matMulArgs.getNumReadThreads();
  size_t numProdThreads = matMulArgs.getNumMatMulThreads();
  std::string directory = matMulArgs.getDirectory();
  std::string outputDirectory = matMulArgs.getOutputDir();
  bool runSequential = matMulArgs.isRunSequential();
  bool validate = matMulArgs.isValidateResults();

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

    ReadMatrixTaskMMap *readAMatTask =
        new ReadMatrixTaskMMap(numReadThreads,
                           MatrixType::MatrixA,
                           blockSize,
                           sharedDim,
                           matrixAHeight,
                           inputDirectoryA,
                           false);

    ReadMatrixTaskMMap *readBMatTask =
        new ReadMatrixTaskMMap(numReadThreads,
                           MatrixType::MatrixB,
                           blockSize,
                           matrixBWidth,
                           sharedDim,
                           inputDirectoryB,
                           false);
    MatMulBlkTask *mmulTask = new MatMulBlkTask(numProdThreads, false);
    MatMulAccumTask *accumTask = new MatMulAccumTask(numProdThreads / 2, false);

    MatMulOutputTaskWithMMap *outputTask = new MatMulOutputTaskWithMMap(outputDirectory, matrixBWidth, matrixAHeight, blockSize, false);

    size_t blkHeightMatB = readBMatTask->getNumBlocksRows();
    size_t blkWidthMatB = readBMatTask->getNumBlocksCols();

    size_t blkHeightMatA = readAMatTask->getNumBlocksRows();
    size_t blkWidthMatA = readAMatTask->getNumBlocksCols();

    MatMulDistributeRule *distributeRuleMatA = new MatMulDistributeRule(MatrixType::MatrixA);
    MatMulDistributeRule *distributeRuleMatB = new MatMulDistributeRule(MatrixType::MatrixB);

    MatMulLoadRule<double *> *loadRule = new MatMulLoadRule<double *>(blkWidthMatA, blkHeightMatA, blkWidthMatB, blkHeightMatB);
    MatMulAccumulateRule<double *> *accumulateRule = new MatMulAccumulateRule<double *>(blkWidthMatB, blkHeightMatA, blkWidthMatA);

    MatMulOutputRule *outputRule = new MatMulOutputRule(blkWidthMatB, blkHeightMatA, blkWidthMatA);

    auto distributeBk = new htgs::Bookkeeper<MatrixRequestData>();
    auto matMulBk = new htgs::Bookkeeper<MatrixBlockData<double *>>();
    auto matAccumBk = new htgs::Bookkeeper<MatrixBlockData<double *>>();

    auto taskGraph = new htgs::TaskGraphConf<MatrixRequestData, MatrixRequestData>();

    taskGraph->setGraphConsumerTask(distributeBk);
    taskGraph->addRuleEdge(distributeBk, distributeRuleMatA, readAMatTask);
    taskGraph->addRuleEdge(distributeBk, distributeRuleMatB, readBMatTask);

    taskGraph->addEdge(readAMatTask, matMulBk);
    taskGraph->addEdge(readBMatTask, matMulBk);

    taskGraph->addRuleEdge(matMulBk, loadRule, mmulTask);

    taskGraph->addEdge(mmulTask, matAccumBk);
    taskGraph->addRuleEdge(matAccumBk, accumulateRule, accumTask);
    taskGraph->addEdge(accumTask, matAccumBk);

    taskGraph->addRuleEdge(matAccumBk, outputRule, outputTask);
    taskGraph->addGraphProducerTask(outputTask);

    taskGraph->writeDotToFile("matMul.dot");


    htgs::TaskGraphRuntime *runtime = new htgs::TaskGraphRuntime(taskGraph);

    clk.start();

    runtime->executeRuntime();

    for (size_t row = 0; row < blkHeightMatA; row++) {
      for (size_t col = 0; col < blkWidthMatA; col++) {

        MatrixRequestData *matrixA = new MatrixRequestData(row, col, MatrixType::MatrixA);
        taskGraph->produceData(matrixA);

      }
    }

    for (size_t col = 0; col < blkWidthMatB; col++) {
      for (size_t row = 0; row < blkHeightMatB; row++) {
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