//
// Created by tjb3 on 12/2/15.
//

//#define DEBUG_FLAG
//#define DEBUG_LEV

#include <tile-grid.hpp>
#include <tile-grid-traverser.hpp>
#include <util-stitching.h>
#include <htgs/api/TaskGraph.hpp>
#include <htgs/api/ExecutionPipeline.hpp>


#include "cuda/tasks/ReadTask.h"
#include "cuda/tasks/FFTTask.h"
#include "cpu/memory/ReadMemory.h"
#include "cuda/memory/CudaMemory.h"
#include "cuda/data/PCIAMData.h"
#include "cuda/rules/StitchingRule.h"
#include "cuda/tasks/PCIAMTask.h"
#include "cuda/tasks/CCFTask.h"
#include "cuda/rules/GridDecompositionRule.h"

using namespace htgs;

namespace is = ImageStitching;

int main() {

  std::cout << "Testing CUDA Runtime Execution Pipeline" << std::endl;

  int startRow = 0;
  int startCol = 0;
  int extentWidth = 220;
  int extentHeight = 220;
  int numGpus = 1;

  DEBUG("Building Grid");
  std::string path("/home/tjb3/datasets/image-stitching/synth_large_grid_sequential");
  TileGrid<is::CUDAImageTile> *grid = new TileGrid<is::CUDAImageTile>(startRow,
                                                                      startCol,
                                                                      extentWidth,
                                                                      extentHeight,
                                                                      220,
                                                                      220,
                                                                      GridOrigin::UpperLeft,
                                                                      GridNumbering::Row,
                                                                      1,
                                                                      path,
                                                                      "img_000{ppppp}.tif",
                                                                      is::ImageTileType::CUDA);

  is::CUDAImageTile *tile = grid->getSubGridTilePtr(0, 0);
  tile->readTile();
  TileGridTraverser<is::CUDAImageTile> *traverser = createTraverser(grid, Traversal::DiagonalTraversal);

  DEBUG("Initializing CUDA contexts");
  int gpuIds[2] = {1, 2};

  CUcontext *contexts = is::CUDAImageTile::initCUDA(tile, numGpus, gpuIds);

  DEBUG("Setting up tasks");
  // Create ITasks
  ReadTask *readTask =
      new ReadTask(grid->getStartCol(), grid->getStartRow(), grid->getExtentWidth(), grid->getExtentHeight());
  FFTTask *fftTask = new FFTTask(contexts,
                                 gpuIds,
                                 numGpus,
                                 tile,
                                 grid->getStartCol(),
                                 grid->getStartRow(),
                                 grid->getExtentWidth(),
                                 grid->getExtentHeight());
  Bookkeeper<FFTData> *bookkeeper = new Bookkeeper<FFTData>();
  PCIAMTask *pciamTask = new PCIAMTask(contexts, gpuIds, numGpus, tile);
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
  DEBUG("Creating task graph");
  TaskGraph<FFTData, CCFData> *taskGraph = new TaskGraph<FFTData, CCFData>();

  // Setup connections
  DEBUG("Adding edges");
  taskGraph->addEdge(readTask, fftTask);
  taskGraph->addEdge(fftTask, bookkeeper);
  taskGraph->addRule(bookkeeper, pciamTask, stitchingRule);
  taskGraph->addGraphInputConsumer(readTask);
  taskGraph->addGraphOutputProducer(pciamTask);

  taskGraph->addMemoryManagerEdge("read", readTask, ccfTask, new ReadMemory(tile->getSize()), 500, MMType::Static);
  taskGraph->addCudaMemoryManagerEdge("fft",
                                      readTask,
                                      pciamTask,
                                      new CudaMemory(tile->fftSize),
                                      500,
                                      MMType::Static,
                                      contexts);

  GridDecompositionRule *decompRule = new GridDecompositionRule(grid->getStartRow(),
                                                                grid->getStartCol(),
                                                                grid->getExtentWidth(),
                                                                grid->getExtentHeight(),
                                                                numGpus);
  ExecutionPipeline<FFTData, CCFData> *execPipeline = new ExecutionPipeline<FFTData, CCFData>(numGpus, taskGraph);
  execPipeline->addInputRule(decompRule);

//    Task<FFTData, CCFData> *execPipelineTask = new Task<FFTData, CCFData>(execPipelineI, 1, false, 0, 1);


  TaskGraph<FFTData, VoidData> *mainGraph = new TaskGraph<FFTData, VoidData>();
  mainGraph->addGraphInputConsumer(execPipeline);
  mainGraph->addEdge(execPipeline, ccfTask);
  mainGraph->incrementGraphInputProducer();

  mainGraph->writeDotToFile("/home/tjb3/cuda.dot");

  Runtime *runTime = new Runtime(mainGraph);

  DEBUG("Producing data for graph edge");
  int count = 0;
  while (traverser->hasNext()) {
    FFTData *data = new FFTData(traverser->nextPtr(), count);
    mainGraph->produceData(data);
    count++;
  }

  mainGraph->finishedProducingData();

  auto start = std::chrono::high_resolution_clock::now();

  runTime->executeRuntime();

  runTime->waitForRuntime();

  mainGraph->writeDotToFile("/home/tjb3/cuda-run.dot");

  auto finish = std::chrono::high_resolution_clock::now();
  std::cout << "Execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count()
      << " ms" << std::endl;

  writeTranslationsToFile(grid, "/home/tjb3/cpp-htgs-out-cuda-runtime-executionPipeline.txt");

  std::cout << "Finished CUDA Runtime Execution Pipeline Test " << std::endl;


  is::CUDAImageTile::destroyPlans(0);

  return 0;
}