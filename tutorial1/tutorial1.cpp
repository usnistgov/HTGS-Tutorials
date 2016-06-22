#include <htgs/api/TaskGraph.hpp>
#include <htgs/api/Runtime.hpp>
#include "tasks/AddTask.h"

int main() {

  // Creates the Task
  AddTask *addTask = new AddTask();

  // Creates the TaskGraph
  auto taskGraph = new htgs::TaskGraph<InputData, OutputData>();

  // Declares that AddTask will be processing the input of a TaskGraph
  taskGraph->addGraphInputConsumer(addTask);

  // Declares that AddTask will be producing data for the output of a TaskGraph
  taskGraph->addGraphOutputProducer(addTask);

  // Increments the number of producers (the main thread will be producing data)
  taskGraph->incrementGraphInputProducer();

  taskGraph->writeDotToFile("tutorial1.dot");

  // Launch the taskGraph
  auto runtime = new htgs::Runtime(taskGraph);

  runtime->executeRuntime();

  int numData = 10;

  // Main thread producing data
  for (int i = 0; i < numData; i++) {
    auto inputData = new InputData(i, i);
    taskGraph->produceData(inputData);
  }

  // Indicate that the main thread has finished producing data
  taskGraph->finishedProducingData();

  runtime->waitForRuntime();


  // Process the ouput of the TaskGraph until no more data is available
  while (!taskGraph->isOutputTerminated()) {
    auto data = taskGraph->consumeData();

    int result = data->getResult();

    std::cout << "Result: " << result << std::endl;
  }
  // Wait until the runtime has finished processing data

  // Release all memory for the graph
  delete runtime;
}
