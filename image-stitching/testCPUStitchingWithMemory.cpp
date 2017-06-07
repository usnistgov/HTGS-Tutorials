//
// Created by tjb3 on 11/17/15.
//

//#define DEBUG_FLAG
//#define DEBUG_LEVEL_VERBOSE


#include <iostream>
#include <util-stitching.h>

#include <htgs/api/TaskGraphRuntime.hpp>
#include <htgs/api/TaskGraphConf.hpp>


#include "cpu/tasks/ReadTask.h"
#include "cpu/tasks/FFTTask.h"
#include "cpu/rules/StitchingRule.h"
#include "cpu/tasks/PCIAMTask.h"
#include "cpu/memory/FFTWMemory.h"
#include "cpu/memory/ReadMemory.h"
#include "StitchingParams.h"

using namespace std;
using namespace htgs;

namespace is = ImageStitching;


int main(int argc, char **argv) {

  StitchingParams params(argc, argv);

  if (params.parseArgs() < 0)
  {
    return -1;
  }

  if (params.isSaveParams()) {
    params.saveArgs(params.getOutputParamFile());
  }


  std::cout << "Testing Runtime with Memory" << std::endl;

  int startRow = params.getStartRow();
  int startCol = params.getStartCol();
  int extentWidth = params.getExtentWidth();
  int extentHeight = params.getExtentHeight();
  int numThreadsFFT = params.getNumThreadsFFT();
  int numThreadsPCIAM = params.getNumThreadsPCIAM();

  DEBUG("Building Grid");
  TileGrid<is::FFTWImageTile> *grid = new TileGrid<is::FFTWImageTile>(startRow,
                                                                      startCol,
                                                                      extentWidth,
                                                                      extentHeight,
                                                                      params.getGridWidth(),
                                                                      params.getGridHeight(),
                                                                      params.getOrigin(),
                                                                      params.getNumbering(),
                                                                      params.getStartTile(),
                                                                      params.getImageDir(),
                                                                      params.getFilenamePattern(),
                                                                      is::ImageTileType::FFTW);

  is::FFTWImageTile *tile = grid->getSubGridTilePtr(0, 0);
  tile->readTile();
  is::FFTWImageTile::initPlans(tile->getWidth(), tile->getHeight(), getFftwMode(params.getFftwMode()), params.isLoadPlan(), params.getPlanFile());

  if (params.isSavePlan())
    is::FFTWImageTile::savePlan(params.getPlanFile());

  TileGridTraverser<is::FFTWImageTile> *traverser = createTraverser(grid, Traversal::DiagonalTraversal);

  DEBUG("Setting up tasks");
  // Create ITasks
  ReadTask *readTask =
      new ReadTask(grid->getStartCol(), grid->getStartRow(), grid->getExtentWidth(), grid->getExtentHeight());
  FFTTask *fftTask =
      new FFTTask(numThreadsFFT, tile, grid->getStartCol(), grid->getStartRow(), grid->getExtentWidth(), grid->getExtentHeight());
  Bookkeeper<FFTData> *bookkeeper = new Bookkeeper<FFTData>();
  PCIAMTask *pciamTask = new PCIAMTask(numThreadsPCIAM, tile);

  // Create rules
  StitchingRule *stitchingRule = new StitchingRule(grid);

  // Create task graph
  DEBUG("Creating task graph");
  TaskGraphConf<FFTData, FFTData> *taskGraph = new TaskGraphConf<FFTData, FFTData>();

  // Setup connections
  DEBUG("Adding edges");
  taskGraph->addEdge(readTask, fftTask);
  taskGraph->addEdge(fftTask, bookkeeper);
  taskGraph->addRuleEdge(bookkeeper, stitchingRule, pciamTask);
  taskGraph->setGraphConsumerTask(readTask);

  ReadMemory *readMemAlloc = new ReadMemory((size_t)tile->getSize());
  FFTWMemory *fftwMemAlloc = new FFTWMemory((size_t)tile->fftSize);

  int memoryPoolSize = min(extentWidth, extentHeight) + 1;

  if (params.getMemoryPoolSize() > memoryPoolSize) {
    std::cout << "Using memory pool size argument" << std::endl;
    memoryPoolSize = params.getMemoryPoolSize();
  }
  else
  {
    std::cout << "Ignoring memory pool size argument" << std::endl;
  }


//  taskGraph->addMemoryManagerEdge("read", readTask, readMemAlloc, memoryPoolSize, MMType::Static);
//  taskGraph->addMemoryManagerEdge("fft", readTask, fftwMemAlloc, memoryPoolSize, MMType::Static);

  TaskGraphRuntime *runTime = new TaskGraphRuntime(taskGraph);
//    Runtime *runTime = new Runtime(taskGraph);

  DEBUG("Producing data for graph edge");
  int count = 0;
  while (traverser->hasNext()) {
    FFTData *data = new FFTData(traverser->nextPtr(), count);
    taskGraph->produceData(data);
    count++;
  }

  taskGraph->finishedProducingData();

  auto start = std::chrono::high_resolution_clock::now();

  runTime->executeAndWaitForRuntime();

  auto finish = std::chrono::high_resolution_clock::now();
  std::cout << "Execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count()
      << " ms" << std::endl;

  std::stringstream outputFile;
  outputFile << params.getOutputDir() << "/" << params.getOutputFilePrefix() << "-pre-optimization-translations-fftw-with-memorypool" << params.getExtentWidth() << "-" << params.getExtentHeight() << ".txt";
  writeTranslationsToFile(grid, outputFile.str());

  std::cout << "Finished Runtime with Memory Test" << std::endl;


  return 0;
}
