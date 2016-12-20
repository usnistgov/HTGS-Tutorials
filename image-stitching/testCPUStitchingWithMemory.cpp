//
// Created by tjb3 on 11/17/15.
//

//#define DEBUG_FLAG
//#define DEBUG_LEVEL_VERBOSE


#include <iostream>
#include <util-stitching.h>
#include <htgs/api/Runtime.hpp>
#include <htgs/api/TaskGraph.hpp>

#include "cpu/tasks/ReadTask.h"
#include "cpu/tasks/FFTTask.h"
#include "cpu/rules/StitchingRule.h"
#include "cpu/tasks/PCIAMTask.h"
#include "cpu/memory/FFTWMemory.h"
#include "cpu/memory/ReadMemory.h"

using namespace std;
using namespace htgs;

namespace is = ImageStitching;


int main() {


  std::cout << "Testing Runtime with Memory" << std::endl;

  int startRow = 0;
  int startCol = 0;
  int extentWidth = 23;
  int extentHeight = 30;

  DEBUG("Building Grid");
  std::string path("/home/tjb3/datasets/image-stitching/1h_Wet_10Perc");
  TileGrid<is::FFTWImageTile> *grid = new TileGrid<is::FFTWImageTile>(startRow,
                                                                      startCol,
                                                                      extentWidth,
                                                                      extentHeight,
                                                                      23,
                                                                      30,
                                                                      GridOrigin::UpperRight,
                                                                      GridNumbering::Column,
                                                                      1,
                                                                      path,
                                                                      "KB_2012_04_13_1hWet_10Perc_IR_0{pppp}.tif",
                                                                      is::ImageTileType::FFTW);

  is::FFTWImageTile *tile = grid->getSubGridTilePtr(0, 0);
  tile->readTile();
  is::FFTWImageTile::initPlans(tile->getWidth(), tile->getHeight(), FFTW_PATIENT, true, "test.dat");
  is::FFTWImageTile::savePlan("test.dat");
  TileGridTraverser<is::FFTWImageTile> *traverser = createTraverser(grid, Traversal::DiagonalTraversal);

  DEBUG("Setting up tasks");
  // Create ITasks
  ReadTask *readTask =
      new ReadTask(grid->getStartCol(), grid->getStartRow(), grid->getExtentWidth(), grid->getExtentHeight());
  FFTTask *fftTask =
      new FFTTask(10, tile, grid->getStartCol(), grid->getStartRow(), grid->getExtentWidth(), grid->getExtentHeight());
  Bookkeeper<FFTData> *bookkeeper = new Bookkeeper<FFTData>();
  PCIAMTask *pciamTask = new PCIAMTask(30, tile);

  // Create rules
  StitchingRule *stitchingRule = new StitchingRule(grid);

  // Create tasks
//    Task<FFTData, FFTData> *readTask = new Task<FFTData, FFTData>(readITask, 1, false, 0, 1);
//    Task<FFTData, FFTData> *fftTask = new Task<FFTData, FFTData>(fftITask, 10, false, 0, 1);
//    Task<FFTData, VoidData> *bkTask = new Task<FFTData, VoidData>(bookkeeper, 1, false, 0, 1);
//    Task<PCIAMData, PCIAMData> *pciamTask= new Task<PCIAMData, PCIAMData>(pciamITask, 30, false, 0, 1);

  // Create task graph
  DEBUG("Creating task graph");
  TaskGraph<FFTData, FFTData> *taskGraph = new TaskGraph<FFTData, FFTData>();

  // Setup connections
  DEBUG("Adding edges");
  taskGraph->addEdge(readTask, fftTask);
  taskGraph->addEdge(fftTask, bookkeeper);
  taskGraph->addRule(bookkeeper, pciamTask, stitchingRule);
  taskGraph->addGraphInputConsumer(readTask);
  taskGraph->incrementGraphInputProducer();

  ReadMemory *readMemAlloc = new ReadMemory(tile->getSize());
  FFTWMemory *fftwMemAlloc = new FFTWMemory(tile->fftSize);

  taskGraph->addMemoryManagerEdge("read", readTask, pciamTask, readMemAlloc, 100, MMType::Static);
  taskGraph->addMemoryManagerEdge("fft", readTask, pciamTask, fftwMemAlloc, 100, MMType::Static);

//    TaskGraph<FFTData, FFTData> *copy = taskGraph->copy(0, 1);
//    copy->incrementGraphInputProducer();

  Runtime *runTime = new Runtime(taskGraph);
//    Runtime *runTime = new Runtime(taskGraph);

  DEBUG("Producing data for graph edge");
  int count = 0;
  while (traverser->hasNext()) {
    FFTData *data = new FFTData(traverser->nextPtr(), count);
    taskGraph->produceData(data);
//        copy->produceData(data);
    count++;
  }

  taskGraph->finishedProducingData();
//    copy->finishedProducingData();

  auto start = std::chrono::high_resolution_clock::now();

  runTime->executeAndWaitForRuntime();

  auto finish = std::chrono::high_resolution_clock::now();
  std::cout << "Execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count()
      << " ms" << std::endl;

  writeTranslationsToFile(grid, "/home/tjb3/cpp-htgs-out-runtime-with-memory.txt");

  std::cout << "Finished Runtime with Memory Test" << std::endl;


  return 0;
}
