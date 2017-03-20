
// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
// NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
// You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

#include <htgs/api/TaskGraphConf.hpp>
#include <htgs/api/TaskGraphRuntime.hpp>
#include "tasks/AddTask.h"

int main() {

  // Creates the Task
  AddTask *addTask = new AddTask();

  // Creates the TaskGraph
  auto taskGraph = new htgs::TaskGraphConf<InputData, OutputData>();

  // Declares that AddTask will be processing the input of a TaskGraph
  taskGraph->setGraphConsumerTask(addTask);

  // Declares that AddTask will be producing data for the output of a TaskGraph
  taskGraph->addGraphProducerTask(addTask);

  taskGraph->writeDotToFile("tutorial1.dot");

  // Launch the taskGraph
  auto runtime = new htgs::TaskGraphRuntime(taskGraph);

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
