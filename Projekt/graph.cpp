#include "common.h"

namespace graph
{
    using namespace common;
        
    class Vertex : public BinarySerializable
    {
    private:
        int id;
    public: 
        Vertex(int id)
        {
            SerilizedSize = sizeof(int);
            this->id = id;
        }   
            
        Vertex(unsigned char* buffer)
        {
            SerilizedSize = sizeof(int);
            int* serializedId = (int*)(buffer); 
            id = *serializedId;            
        }
        
        void serialize(unsigned char* buffer) override
        {
            int* serializedId = (int*)(buffer); 
            *serializedId = id;
        }
    
        int getId()
        {
            return id;
        }
    };

    class Edge : public BinarySerializable
    {
    private:
        Vertex* a;
        Vertex* b; 
        double cost;
    public:
        Edge(Vertex* a, Vertex* b, double cost)
        {
            this->a = a;
            this->b = b;
            this->cost = cost;
            SerilizedSize = a->SerilizedSize + b->SerilizedSize + sizeof(double);
        }
        
        Edge(unsigned char* buffer)
        {
            this->a = new Vertex(buffer);
            this->b = new Vertex(buffer + a->SerilizedSize);
            double* serializedCost = (double*)(buffer + a->SerilizedSize + b->SerilizedSize); 
            this->cost = *serializedCost;
            SerilizedSize = a->SerilizedSize + b->SerilizedSize + sizeof(double);
        }
        
        ~Edge()
        {
            delete this->a;
            delete this->b;
        }
        
        void serialize(unsigned char* buffer) override
        {
            a->serialize(buffer);
            b->serialize(buffer + a->SerilizedSize);
            double* serializedCost = (double*)(buffer + a->SerilizedSize + b->SerilizedSize); 
            *serializedCost = cost;
        }
        
        Vertex* getA()
        {
            return a;
        }
        
        Vertex* getB()
        {
            return b;
        }
        
        double getCost()
        {
            return cost;
        }
        
        void setCost(double cost)
        {
            this->cost = cost;
        }
    };

    class Graph : public BinarySerializable
    {
        private:
            std::vector<Edge*> edges;
            int vertexCount;
        public:
            Graph(std::vector<Edge*> edges)
            {
                this->edges = edges;
                this->SerilizedSize += sizeof(int);                this->vertexCount = 0;

                bool* tmp = new bool[edges.size()];
                for(auto &edge : edges)
                {
                    this->SerilizedSize += edge->SerilizedSize;
                    tmp[edge->getA()->getId()] = true;
                    tmp[edge->getB()->getId()] = true;
                }
                for(int i = 0; i< edges.size(); i++)
                {
                    if(tmp[i])
                    {
                        vertexCount++;
                    }
                }
            }
            
            Graph(unsigned char* buffer, int buffSize)
            {
                int usedBytes = 0;
                vertexCount = *((int*)buffer);
                usedBytes+= sizeof(int);
                while(usedBytes < buffSize)
                {
                    Edge* newEdge = new Edge(buffer + usedBytes);
                    this->edges.push_back(newEdge);
                    usedBytes += newEdge->SerilizedSize;
                }
            }
            
            ~Graph()
            {
                for(auto &edge : edges)
                {
                    delete edge;
                }
            }
            
            
            void serialize(unsigned char* buffer) override {
                unsigned char *tmpBuff = buffer;
                int *tmp = (int *) (tmpBuff);
                *tmp = vertexCount;
                tmpBuff += sizeof(int);
                for (auto &edge : edges) {
                    edge->serialize(tmpBuff);
                    tmpBuff = tmpBuff + edge->SerilizedSize;
                }
            }

            std::vector<Edge*> getEdges()
            {
                return edges;
            }

        int getVertexCount(){
            return vertexCount;
        }
    };
    
    class VertexWithNeighbour
    {
    private:
        std::unordered_map<int,double> neighbours;
        int id;
    public:
        VertexWithNeighbour(int id)
        {
            this->id = id; 
        }
    
        void addNeighbour(int id, double cost)
        {
            neighbours[id] = cost;
        }
        
        std::unordered_map<int,double> getNeighbours()
        {
            return neighbours;
        }
        
        int getId()
        {
            return id;
        }
    };
    
    class VertexGraph
    {
    private:
        std::unordered_map<int,VertexWithNeighbour*> vertexes;
        
        void proccesEdge(int aId, int bId, double cost)
        {
            VertexWithNeighbour* vertex = vertexes[aId];
            if (!vertex)
            {
                vertex = new VertexWithNeighbour(aId);
                vertexes[aId] = vertex;
            }
            vertex->addNeighbour(bId, cost);
        }
        
    public:
        VertexGraph(Graph* edgeGraph)
        {
            for(auto &edge: edgeGraph->getEdges())
            {
                int aId = edge->getA()->getId();
                int bId = edge->getB()->getId();
                double cost = edge->getCost();
                proccesEdge(aId,bId,cost);
                proccesEdge(bId,aId,cost);
            }
        }
        
        std::unordered_map<int,VertexWithNeighbour*> getVertexes()
        {
            return vertexes;
        }
    };
    
    Graph* readFromFile(const char* filePath)
    {
        std::fstream file;
        file.open(filePath, std::ios::in);
        if (!file.is_open())
        {
            throw std::string("Invalid file");
        }
        std::vector<Edge*> edges;
        std::string unused;
        while ( std::getline(file, unused))
        {
            int aId;
            int bId; 
            double cost;
            file >> aId;
            file >> bId;
            file >> cost;
            edges.push_back(new 
                Edge(new Vertex(aId),new Vertex(bId),cost));
        }
        file.close();
        return new Graph(edges);
    }
}