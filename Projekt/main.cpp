#include <mpi.h>

#include "graph.cpp"
#include "common.h"
#include "mpiHelper.cpp"
#include "shortestPath.cpp"

using namespace mpi;
using namespace graph;
using namespace shortestPath;

int main(int argc, char** argv)
{
    if(argc != 4)
    {
        std::cerr<<"Usage: \nmpirun "<<argv[0]<<
            " GRAPH_INPUT_FILE_PATH TASKS_INPUT_FILE_PATH OTPUT_FOLDER\n";
        exit(EXIT_FAILURE);
    }
    MpiHelper mpi;
    Graph* graph;
    if(mpi.isMaster())
    {
        graph = readFromFile(argv[1]);
        mpi.bcast(graph);
        std::cout <<"Master: " << graph->getEdges().size() << "\n";
    }
    else
    {
        graph = mpi.reciveGraphBcast(MpiHelper::RootId);
        auto path = new DijkstraShortestPath(graph);
        std::cout <<"Slave: " << graph->getEdges().size() << "\n";
        std::cout <<"Slave: " << path->ShortestPath(1,3) << "\n";
    }
}