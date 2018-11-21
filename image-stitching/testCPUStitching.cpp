//
// Created by tjb3 on 11/17/15.
//

//#define DEBUG_FLAG
//#define DEBUG_LEVEL_VERBOSE


#include <iostream>
#include <util-stitching.h>
#include <htgs/api/TaskGraphConf.hpp>
#include <htgs/api/TaskGraphRuntime.hpp>

#include "cpu/tasks/ReadTask.h"
#include "cpu/tasks/FFTTask.h"
#include "cpu/rules/StitchingRule.h"
#include "cpu/tasks/PCIAMTask.h"
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



  std::cout << "Testing Runtime" << std::endl;

  int startRow = params.getStartRow();
  int startCol = params.getStartCol();
  int extentWidth = params.getExtentWidth();
  int extentHeight = params.getExtentHeight();
  int numThreadsFFT = params.getNumThreadsFFT();
  int numThreadsPCIAM = params.getNumThreadsPCIAM();

  HTGS_DEBUG("Building Grid");
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

  HTGS_DEBUG("Setting up tasks");
  // Create ITasks
  ReadTask *readTask =
      new ReadTask(grid->getStartCol(), grid->getStartRow(), grid->getExtentWidth(), grid->getExtentHeight());
  FFTTask *fftTask =
      new FFTTask(numThreadsFFT, tile, grid->getStartCol(), grid->getStartRow(), grid->getExtentWidth(), grid->getExtentHeight());
  Bookkeeper<FFTData> *bookkeeper = new Bookkeeper<FFTData>();
  PCIAMTask *pciamTask = new PCIAMTask(numThreadsPCIAM, tile);

  // Create rules
  StitchingRule *stitchingRule = new StitchingRule(grid);

  // Create tasks
//    Task<FFTData, FFTData> *readTask = new Task<FFTData, FFTData>(readITask, 1, false, 0, 1);
//    Task<FFTData, FFTData> *fftTask = new Task<FFTData, FFTData>(fftITask, 10, false, 0, 1);
//    Task<FFTData, VoidData> *bkTask = new Task<FFTData, VoidData>(bookkeeper, 1, false, 0, 1);
//    Task<PCIAMData, PCIAMData> *pciamTask= new Task<PCIAMData, PCIAMData>(pciamITask, 40, false, 0, 1);

  // Create task graph
  HTGS_DEBUG("Creating task graph");
  TaskGraphConf<FFTData, FFTData> *taskGraph = new TaskGraphConf<FFTData, FFTData>();

  // Setup connections
  HTGS_DEBUG("Adding edges");
  taskGraph->addEdge(readTask, fftTask);
  taskGraph->addEdge(fftTask, bookkeeper);
  taskGraph->addRuleEdge(bookkeeper, stitchingRule, pciamTask);
  taskGraph->setGraphConsumerTask(readTask);

//  TaskGraph<FFTData, FFTData> *copy = taskGraph->copy(0, 1);
//  copy->incrementGraphInputProducer();

  TaskGraphRuntime *runTime = new TaskGraphRuntime(taskGraph);

  HTGS_DEBUG("Producing data for graph edge");
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

  delete runTime;

  std::stringstream outputFile;
  outputFile << params.getOutputDir() << "/" << params.getOutputFilePrefix() << "-pre-optimization-translations-fftw-no-memorypool" << params.getExtentWidth() << "-" << params.getExtentHeight() << ".txt";
  writeTranslationsToFile(grid, outputFile.str());

  std::cout << "Finished Runtime Test" << std::endl;


  return 0;
}
