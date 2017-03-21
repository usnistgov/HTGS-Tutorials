//
// Created by tjb3 on 3/21/17.
//

#include "MatMulArgs.h"
void MatMulArgs::processArgs(int argc, char **argv) {
  for (int arg = 1; arg < argc; arg++)
  {
    std::string argvs(argv[arg]);

    if (argvs == "--width-b") {
      arg++;
      matrixBWidth = (size_t)atoi(argv[arg]);
    }

    if (argvs == "--height-a") {
      arg++;
      matrixAHeight = (size_t)atoi(argv[arg]);
    }

    if (argvs == "--shared-dim") {
      arg++;
      sharedDim = (size_t)atoi(argv[arg]);
    }

    if (argvs == "--block-size") {
      arg++;
      blockSize = (size_t)atoi(argv[arg]);
    }

    if (argvs == "--num-readers") {
      arg++;
      numReadThreads = (size_t)atoi(argv[arg]);
    }

    if (argvs == "--num-workers") {
      arg++;
      numMatMulThreads = (size_t)atoi(argv[arg]);
    }

    if (argvs == "--dir") {
      arg++;
      directory = argv[arg];
    }

    if (argvs == "--output-dir") {
      arg++;
      outputDir = argv[arg];
    }

    if (argvs == "--validate-results") {
      validateResults = true;
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
}
