
#include "common.h"
#include "graph.h"

namespace shortestPath
{
    using namespace graph;
    
    class DijkstraShortestPath
    {
        private:
        Graph* graph;
        public:
        DijkstraShortestPath(Graph* graph)
        {
            this->graph = graph;
        }
                
        double ShortestPath(int startId, int endId)
        {
            auto edges = graph->getEdges();
            std::vector<std::vector<Edge*>> tmpGraph (edges.size());
            for(auto &edge : edges)
            {
                tmpGraph[edge->getA()->getId()].push_back(edge);
            }
            return ShortestPath(tmpGraph,startId,endId);
        }
        
        double ShortestPath(std::vector<std::vector<Edge*>> graph, int source, int target) 
        {
            std::vector<double> min_distance( graph.size(), INT_MAX );
            min_distance[ source ] = 0;
            std::set<std::pair<double,int>> active_vertices;
            active_vertices.insert( {0,source} );
                
            while (!active_vertices.empty()) 
            {
                int where = active_vertices.begin()->second;
                if (where == target) 
                {
                    return min_distance[where];
                }
                active_vertices.erase( active_vertices.begin() );
                for (auto vertex : graph[where]) 
                {
                    if (min_distance[vertex->getB()->getId()] > min_distance[where] 
                        + vertex->getCost()) 
                    {
                        active_vertices.erase( { 
                            min_distance[vertex->getB()->getId()], vertex->getB()->getId() } );
                        min_distance[vertex->getB()->getId()] = min_distance[where]
                             + vertex->getCost();
                        active_vertices.insert( { 
                            min_distance[vertex->getB()->getId()], vertex->getB()->getId() } );
                    }
                }
            }
            return -1;
        }

    };

    // V-> Number of vertices, E-> Number of edges
    class BellmanFordShortestPath
    {
    private:
        Graph *graph;
    public:
        BellmanFordShortestPath(Graph *graph)
        {
            this->graph = graph;
        }

        double ShortestPath(int src,int dstId)
        {
            int E = graph->getEdges().size();
            int V = graph->getVertexCount();
            double dist[V];

            // Step 1: Initialize distances from src to all other vertices
            // as INFINITE
            for (int i = 0; i < V; i++)
                dist[i]   = -1;
            dist[src] = 0;

            // Step 2: Relax all edges |V| - 1 times. A simple shortest
            // path from src to any other vertex can have at-most |V| - 1
            // edges
            relaxEdges(dstId, 2,E, dist);

            // Step 3: check for negative-weight cycles.  The above step
            // guarantees shortest distances if graph doesn't contain
            // negative weight cycle.  If we get a shorter path, then there
            // is a cycle.
            for (int i = 0; i < E; i++)
            {
                int u = graph->getEdges()[i]->getA()->getId();
                int v = graph->getEdges()[i]->getB()->getId();
                double weight = graph->getEdges()[i]->getCost();
                if (dist[u] != INT_MAX && dist[u] + weight < dist[v])
                    throw std::string("Graph contains negative weight cycle");
            }

            return dist[dstId];
        }

        void relaxEdges(int dstId, int vertexCount, int edgesCount, double *dist) const {
            for (int i = 1; i <= vertexCount - 1; i++)
            {
                for (int j = 0; j < edgesCount; j++)
                {
                    int u = graph->getEdges()[j]->getA()->getId();
                    int v = graph->getEdges()[j]->getB()->getId();
                    double weight = graph->getEdges()[j]->getCost();
                    if (dist[u] != INT_MAX && dist[u] + weight < dist[v]) {
                        dist[v] = dist[u] + weight;
                    }
                    if(v == dstId || u == dstId)
                    {
                        return;
                    }
                }
            }
        }

    };
    
    
}