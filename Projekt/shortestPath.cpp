#include "common.h"

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
            return INT_MAX;
        }

    };
    
    
}