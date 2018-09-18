//
// Created by tjb3 on 12/2/15.
//

//#define DEBUG_FLAG
//#define DEBUG_LEVEL_VERBOSE

#include <tile-grid.hpp>
#include <tile-grid-traverser.hpp>
#include <util-stitching.h>

#include <htgs/api/Bookkeeper.hpp>
#include <htgs/api/TaskGraphConf.hpp>
#include <htgs/api/TaskGraphRuntime.hpp>

#include "cuda/tasks/ReadTask.h"
#include "cuda/tasks/FFTTask.h"

#include "cpu/memory/ReadMemory.h"
#include "cuda/memory/CudaMemory.h"
#include "cuda/data/PCIAMData.h"
#include "cuda/rules/StitchingRule.h"
#include "cuda/tasks/PCIAMTask.h"
#include "cuda/tasks/CCFTask.h"

using namespace htgs;

namespace is = ImageStitching;

int main() {

  std::cout << "Testing CUDA Runtime" << std::endl;

  int startRow = 0;
  int startCol = 0;
  int extentWidth = 23;
  int extentHeight = 30;
  int numGpus = 1;

  HTGS_DEBUG("Building Grid");
  std::string path("/home/tjb3/datasets/image-stitching/1h_Wet_10Perc");
  TileGrid<is::CUDAImageTile> *grid = new TileGrid<is::CUDAImageTile>(startRow,
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
                                                                      is::ImageTileType::CUDA);

  is::CUDAImageTile *tile = grid->getSubGridTilePtr(0, 0);
  tile->readTile();
  TileGridTraverser<is::CUDAImageTile> *traverser = createTraverser(grid, Traversal::DiagonalTraversal);

  HTGS_DEBUG("Initializing CUDA contexts");
  int gpuIds[1] = {0};

  CUcontext *contexts = is::CUDAImageTile::initCUDA(tile, numGpus, gpuIds);

  HTGS_DEBUG("Setting up tasks");
  // Create ITasks
  ReadTask *readTask =
      new ReadTask(grid->getStartCol(), grid->getStartRow(), grid->getExtentWidth(), grid->getExtentHeight());
  FFTTask *fftTask = new FFTTask(gpuIds,
                                 numGpus,
                                 tile,
                                 grid->getStartCol(),
                                 grid->getStartRow(),
                                 grid->getExtentWidth(),
                                 grid->getExtentHeight());
  Bookkeeper<FFTData> *bookkeeper = new Bookkeeper<FFTData>();
  PCIAMTask *pciamTask = new PCIAMTask(gpuIds, numGpus, tile);
  CCFTask *ccfTask = new CCFTask(40);

  // Create rule
  StitchingRule *stitchingRule = new StitchingRule(grid);

  // Create Tasks
//    Task<FFTData, FFTData> *readTask = new Task<FFTData, FFTData>(readITask, 1, false, 0, 1);
//    Task<FFTData, FFTData> *fftTask = new Task<FFTData, FFTData>(fftITask, 1, false, 0, 1);
//    Task<FFTData, VoidData> *bkTask = new Task<FFTData, VoidData>(bookkeeper, 1, false, 0, 1);
//    Task<PCIAMData, CCFData> *pciamTask = new Task<PCIAMData, CCFData>(pciamITask, 1, false, 0, 1);
//    Task<CCFData, VoidData> *ccfTask = new Task<CCFData, VoidData>(ccfITask, 40, false, 0, 1);

  // Create task graph
  HTGS_DEBUG("Creating task graph");
  TaskGraphConf<FFTData, FFTData> *taskGraph = new TaskGraphConf<FFTData, FFTData>();

  // Setup connections
  HTGS_DEBUG("Adding edges");
  taskGraph->addEdge(readTask, fftTask);
  taskGraph->addEdge(fftTask, bookkeeper);
  taskGraph->addRuleEdge(bookkeeper, stitchingRule, pciamTask);
  taskGraph->addEdge(pciamTask, ccfTask);
  taskGraph->setGraphConsumerTask(readTask);

  taskGraph->addMemoryManagerEdge("read", readTask, new ReadMemory(tile->getSize()), 100, MMType::Static);
  taskGraph->addCudaMemoryManagerEdge("fft",
                                      readTask,
                                      new CudaMemory(tile->fftSize),
                                      100,
                                      MMType::Static,
                                      gpuIds);

  taskGraph->writeDotToFile("/home/tjb3/cuda.dot");

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

  writeTranslationsToFile(grid, "/home/tjb3/cpp-htgs-out-cuda-runtime.txt");

  std::cout << "Finished CUDA Runtime Test " << std::endl;


  is::CUDAImageTile::destroyPlans(0);

  return 0;
}