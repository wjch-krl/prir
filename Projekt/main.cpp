#include "mpiHelper.h"
#include "shortestPath.h"

using namespace mpi;
using namespace graph;
using namespace shortestPath;

std::vector<std::tuple<Vertex*,Vertex*>> getTasks(const char* filePath)
{
    std::vector<std::tuple<Vertex*,Vertex*>> retVal;
    std::fstream file;
    file.open(filePath, std::ios::in);
    if (!file.is_open())
    {
        throw std::string("Invalid file");
    }
    std::string unused;
    while ( std::getline(file, unused))
    {
        int aId;
        int bId;
        file >> aId;
        file >> bId;
        retVal.push_back(std::make_tuple(new Vertex(aId),new Vertex(bId)));
    }
    return retVal;
}

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
    try {
        if(mpi.isMaster())
        {
            graph = readFromFile(argv[1]);
            mpi.bcast(graph);
            std::cout <<"Master: " << graph->getVertexCount() << "\n";
            auto tasks = getTasks(argv[2]);
            mpi.bcastGen(tasks.size());
            for(int i = 0 ; i<tasks.size() ; i++)
            {
                mpi.sendTask(std::get<0>(tasks[i]),std::get<1>(tasks[i]),1);
            }
            std::cout<<"aa";

        }
        else
        {
            graph = mpi.reciveGraphBcast(MpiHelper::RootId);
            auto path = new BellmanFordShortestPath(graph);
            std::cout <<"Slave: " << graph->getVertexCount() << "\n";
            int tasksCount = mpi.recBcastGen<int>(MpiHelper::RootId);
            std::cout<<tasksCount;
            for(int i = 0 ; i<tasksCount ; i++)
            {
                auto task = mpi.reciveTask(MpiHelper::RootId);
                std::cout <<"got task: "<< std::get<0>(task)->getId() << " " << std::get<1>(task)->getId()  <<"\n";

            }
            std::cout<<"dd";
        }
    } catch (std::string ex)
    {
        std::cerr<<ex;
       // mpi.abort();
    }
}