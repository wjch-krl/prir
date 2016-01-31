#include "mpiHelper.h"
#include "shortestPath.h"

using namespace mpi;
using namespace graph;
using namespace shortestPath;

std::vector<std::tuple<Vertex *, Vertex *>> getTasks(const char *filePath) {
    std::vector<std::tuple<Vertex *, Vertex *>> retVal;
    std::fstream file;
    file.open(filePath, std::ios::in);
    if (!file.is_open())
    {
        throw std::string("Invalid file");
    }
    std::string unused;
    while (std::getline(file, unused))
    {
        int aId;
        int bId;
        file >> aId;
        file >> bId;
        retVal.push_back(std::make_tuple(new Vertex(aId), new Vertex(bId)));
    }
    return retVal;
}

void calculatePaths(MpiHelper* mpi) {
    auto graph = mpi->reciveGraphBcast(MpiHelper::RootId);
    auto path = new DijkstraShortestPath(graph);
    long tasksCount = mpi->recBcastGen<long>(MpiHelper::RootId);
    for (int i = 0; i < tasksCount / mpi->getWordSize(); i++)
    {
        auto task = mpi->reciveTask(MpiHelper::RootId);
        double pathCost = path->ShortestPath(std::get<0>(task)->getId(), std::get<1>(task)->getId());
        mpi->send(new SimpleBinarySerializable<double>(pathCost), MpiHelper::RootId);
    }
}

MPI_Request *SendGraphData(MpiHelper *mpi, double *costs,
                                   BinarySerializable *graph, std::vector<std::tuple<Vertex *, Vertex *>> tasks) {
    mpi->bcast(graph);
    mpi->bcastGen(tasks.size());
    auto requests = new MPI_Request[tasks.size()];
    for (int j = 0; j < mpi->getWordSize(); j++)
    {
        for (int i = j; i < tasks.size(); i += mpi->getWordSize())
        {
            mpi->sendTaskAsync(std::get<0>(tasks[i]), std::get<1>(tasks[i]), j);
            requests[i] = *mpi->getCostAsync(j,&costs[i]);
        }
    }
    return requests;
}


int main(int argc, char **argv) {
    if (argc != 3)
    {
        std::cerr << "Usage: \nmpirun " << argv[0] <<
        " GRAPH_INPUT_FILE_PATH TASKS_INPUT_FILE_PATH\n";
        exit(EXIT_FAILURE);
    }
    MpiHelper mpi;
    Graph *graph;
    auto t1 = std::chrono::high_resolution_clock::now();
    try
    {
        MPI_Request* asyncTasks;
        auto graph = readFromFile(argv[1]);
        std::vector<std::tuple<Vertex *, Vertex *>> tasks = getTasks(argv[2]);
        double* costs = new double[tasks.size()];
        if (mpi.isMaster())
        {
            asyncTasks = SendGraphData(&mpi, costs ,graph ,tasks);
        }
        calculatePaths(&mpi);
        if (mpi.isMaster())
        {
            mpi.waitForAll(asyncTasks,tasks.size());
            for(int i = 0; i < tasks.size(); i++)
            {
                std::cout<<"From " <<std::get<0>(tasks[i])->getId()<<" to "<< std::get<1>(tasks[i])->getId() << " cost " << costs[i] << "\n";
            }
        }
    }
    catch (std::string ex)
    {
        std::cerr << ex;
        // mpi.abort();
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    if (mpi.isMaster())
    {
        std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms\n";
    }
}

