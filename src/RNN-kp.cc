#include <algorithm>
#include <queue>
#include "utils/graph.h"


// For Recurrent Neural Network implementation
// Function7 - if the breaking is correct, store corresponding info; otherwise, recover the edge and try another breaking
void ConstructUnrolledGraphForRNN(const NetProto &net_proto)
{
    Graph graph_orig;

    // 1-construct original graph which may contain a cycle, use each SNode can still access the LayerProto information
    map<string, LayerProto> protos;// store the corresponding information between layer name and layer_proto
    map<SNode, LayerProto> node_proto; // store the correponding information between SNode and LayerProto
    for (auto &layer_proto : net_proto.layer())// for each layer in the neural net, add a node in the corresponding graph
    {
        SNode node_temp;
        node_temp = graph_orig.AddNode(layer_proto.name());//use "const SNode& AddNode(string name)"
        protos[layer_proto.name()]=layer_proto;
        node_proto[node_temp] = layer_proto;
    }
    for (auto &layer_proto : net_proto.layer())// add edges in the graph
            if(layer_proto.srclayers_size())//This layer has src layers
            for(const string& src: layer_proto.srclayers())//src layers' definition in model.proto is "repeated string" which means a string array
                graph_orig.AddEdge(src, layer_proto.name());//use "void AddEdge(const string& src, const string& dst)"


    // 2-after breaking one edge in the cycle and save needed information for unroll (graph_orig & correct_breaking_edge)
    std::pair<SNode, SNode> correct_breaking_edge;
    graph_orig.DetectCycleAndSaveCycle();// detect and save the cycle for original graph
    graph_orig.ChangeCycleToEdges();// change the cycle information into edge representation, now in "std::vector<std::pair<SNode, SNode>> cycle_edges"

    for(int i = 0; i < cycle_edges.size(); i++)
    {
        graph_orig.BreakEdge(cycle_edges[i]);
        if(graph_orig.CheckCorrectBreaking() == true)
        {
            correct_breaking_edge = cycle_edges[i];//save the correct edge to be broken & the current graph_orig saves the correct acyclic graph
            break;
        }
        else
        {
            graph_orig.RecoverEdge(cycle_edges[i]);//recover the edge which has been broken
        }

    }

    // 3-unrolling
}
