#include <mpi.h>
#include "common.h"

/*Exit and cleanup*/
namespace mpi 
{
    using namespace common;
    using namespace graph;
    
    void exitFailure()
    {
        MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);    
        MPI_Finalize();        
        exit(EXIT_FAILURE);
    }
    
    class MpiHelper
    {
    private:
        int worldSize;
        int worldRank;
    
    public:
        static const int RootId = 0;
        MpiHelper()
        {
            //Initialize MPI
            MPI_Init(NULL, NULL);
            // Get number of processes
            MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
            // Get process rank
            MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
        }
        
        ~MpiHelper()
        {
            MPI_Finalize();
        }
    
        bool isMaster()
        {
            return worldRank == RootId;
        }
        
        void send(BinarySerializable* object, int destination)
        {
            unsigned char* buffer;
            buffer = object->serialize();
            MPI_Send(buffer, object->SerilizedSize, MPI_UNSIGNED_CHAR,
                destination, 0, MPI_COMM_WORLD);
        }
        
        void bcast(BinarySerializable* object)
        {
            unsigned char* buffer;
            unsigned int bufferSize;
            buffer = object->serialize();
            bufferSize = object->SerilizedSize;
            MPI_Bcast(&bufferSize, 1, MPI_UNSIGNED,
                worldRank, MPI_COMM_WORLD);
            MPI_Bcast(buffer,bufferSize,MPI_UNSIGNED_CHAR,
                worldRank, MPI_COMM_WORLD);
        }
        
        Graph* reciveGraph(int source)
        {
            int bufferSize;
            MPI_Status status;
            unsigned char* buffer;
            // Probe for an incoming message from process source
            MPI_Probe(source, 0, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_INT, &bufferSize);
            buffer = new unsigned char[bufferSize];
            MPI_Recv(buffer, bufferSize, MPI_UNSIGNED_CHAR, source, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            return new Graph(buffer, bufferSize);
        }
        
        Graph* reciveGraphBcast(int root)
        {
            unsigned char* buffer;
            unsigned int bufferSize = 0;
            MPI_Bcast(&bufferSize, 1, MPI_UNSIGNED,
                root, MPI_COMM_WORLD);
            buffer = new unsigned char[bufferSize];
            MPI_Bcast(buffer,bufferSize,MPI_UNSIGNED_CHAR,
                root, MPI_COMM_WORLD);
            return new Graph(buffer, bufferSize);  
        }
        
        Vertex* reciveVertex(int source)
        {
            int bufferSize;
            MPI_Status status;
            unsigned char* buffer;
            bufferSize = sizeof(int);
            buffer = new unsigned char[bufferSize];
            MPI_Recv(buffer, bufferSize, MPI_UNSIGNED_CHAR, source, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            return new Vertex(buffer);
        }
        
        //Or not nessecery
        Graph* sendTask(Vertex* start, Vertex* stop, int destination)
        {
            send(start,destination);
            send(stop,destination);
            return reciveGraph(destination);
        }
        
        std::tuple<Vertex*,Vertex*> reciveTask(int source)
        {
            return std::make_tuple(reciveVertex(source),reciveVertex(source));
        }
    };
}