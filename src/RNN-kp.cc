#include <algorithm>
#include <queue>
#include <vector>
#include "utils/graph.h"
#include "utils/singleton.h"
#include "utils/factory.h"
#include "utils/cluster.h"
#include "proto/model.pb.h"

namespace singa {
typedef std::vector<SNode> nodes_timestamp;//keep all the nodes for one timestamp

// For Recurrent Neural Network implementation
// Function7 - if the breaking is correct, store corresponding info; otherwise, recover the edge and try another breaking
void ConstructUnrolledGraphForRNN(const NetProto &net_proto)
{
    Graph graph_orig;// original graph may have a cycle

    // 1-construct original graph which may contain a cycle, use each SNode can still access the LayerProto information
    map<string, LayerProto> protos;// store the corresponding information between layer name and layer_proto
    map<int, LayerProto> nodeid_proto; // store the correponding information between SNode ID and LayerProto
    for (auto &layer_proto : net_proto.layer())// for each layer in the neural net, add a node in the corresponding graph
    {
        SNode node_temp;
        node_temp = graph_orig.AddNode(layer_proto.name());//use "const SNode& AddNode(string name)"
        protos[layer_proto.name()]=layer_proto;
        nodeid_proto[node_temp->id()] = layer_proto;
    }
    for (auto &layer_proto : net_proto.layer())// add edges in the graph
        if(layer_proto.srclayers_size() != 0)//This layer has src layers
    for(const string& src: layer_proto.srclayers())//src layers' definition in model.proto is "repeated string" which means a string array
                graph_orig.AddEdge(src, layer_proto.name());//use "void AddEdge(const string& src, const string& dst)"


    // 2-after breaking one edge in the cycle and save needed information for unroll (graph_orig (now is acyclic) & correct_breaking_edge)
    std::pair<SNode, SNode> correct_breaking_edge;//this information will be used when unrolling the cyclic graph
    graph_orig.DetectCycleAndSaveCycle();// detect and save the cycle for original graph
    graph_orig.ChangeCycleToEdges();// change the cycle information into edge representation, now in "std::vector<std::pair<SNode, SNode>> cycle_edges"

    for(int i = 0; i < graph_orig.cycle_edges().size(); i++)
    {
        //graph_orig.BreakEdge(graph_orig.cycle_edges().at(i));
        if(graph_orig.CheckCorrectBreaking(graph_orig.cycle_edges().at(i)) == true)
        {
            correct_breaking_edge = graph_orig.cycle_edges().at(i);//save the correct edge to be broken & the current graph_orig saves the correct acyclic graph
            break;
        }
        else
        {
            graph_orig.RecoverEdge(graph_orig.cycle_edges().at(i));//recover the edge which has been broken
        }
    }

    graph_orig.Sort();//topology sort for the current acyclic graph which is constructed by breaking one edge


    // 3-unrolling - constructing unrolled & acyclic graph
    int window_size = net_proto.win_size();
    Graph graph_unroll;
    nodes_timestamp nodes_timeinfo[window_size];//keep the node information grouping by each timestamp
    // Add nodes in the graph
    for(int j = 0; j < graph_orig.nodes().size(); j++)
    {
        if(nodeid_proto[graph_orig.nodes().at(j)->id()].unroll_decision() == true)// For the nodes which need to be unrolled
        {
            SNode nodes_j[window_size];
            for(int k = 0; k < window_size; k++)//This loop corresponds to different timestamps
            {
                nodes_j[k] = graph_orig.nodes().at(j);//default timestamp value is 0
                nodes_j[k]->set_orig(graph_orig.nodes().at(j));
                nodes_j[k]->set_timestamp(k);
                graph_unroll.AddNode(nodes_j[k]);
                nodes_timeinfo[k].push_back(nodes_j[k]);
            }
        }
        else// For the nodes which don't need to be unrolled
        {
            graph_orig.nodes().at(j)->set_timestamp(window_size - 1);
            graph_unroll.AddNode(graph_orig.nodes().at(j));
            nodes_timeinfo[window_size - 1].push_back(graph_orig.nodes().at(j));
        }
    }

    // Add edges in the graph - same timestamp
    for(int p = 0; p < window_size; p++)// traverse all timestamps
    {
        //for the nodes in the same timestamp
        for(int pointer1 = 0; pointer1 < nodes_timeinfo[p].size(); pointer1++)
        {
            for(int pointer2 = pointer1 + 1; pointer2 < nodes_timeinfo[p].size(); pointer2++)
            {
                if(nodes_timeinfo[p].at(pointer1)->orig()->CheckWhetherSrcNode(nodes_timeinfo[p].at(pointer2)->orig()) == true)
                    graph_unroll.AddEdge(nodes_timeinfo[p].at(pointer2), nodes_timeinfo[p].at(pointer1));

                else if(nodes_timeinfo[p].at(pointer1)->orig()->CheckWhetherDstNode(nodes_timeinfo[p].at(pointer2)->orig()) == true)
                    graph_unroll.AddEdge(nodes_timeinfo[p].at(pointer1), nodes_timeinfo[p].at(pointer2));
            }
        }
    }


    // Add edges in the graph - different timestamps
    for(int p = 0; p < window_size; p++)// traverse all timestamps
    {
        //for the nodes in the neighboring timestamps
        for(int pointer1 = 0; pointer1 < nodes_timeinfo[p].size(); pointer1++)// traverse the src node
        {
            if(pointer1 == nodes_timeinfo[p].size() - 1) break;// Not consider the last timestamp
            else if(nodes_timeinfo[p].at(pointer1)->orig() == correct_breaking_edge.first)
            {
                for(int pointer2 = pointer1 + 1; pointer2 < nodes_timeinfo[p + 1].size(); pointer2++)
                {
                    if(nodes_timeinfo[p + 1].at(pointer2)->orig() == correct_breaking_edge.second)
                        graph_unroll.AddEdge(nodes_timeinfo[p].at(pointer1), nodes_timeinfo[p + 1].at(pointer2));
                }
            }

        }
    }


    //4-Consider the important node with special proto input defined in LayerProto as "repeated int32 related_info"
    //(1)-Find this node
    SNode aggregate_node;// this node is the first node which do not need to be unrolled in topological order, its timestamp is window_size - 1
    for(int i = 0; i < graph_orig.nodes().size(); i++)
    {
        if(nodeid_proto[graph_orig.nodes().at(i)->orig()->id()].unroll_decision() == false)//the timestamp of graph_orig.nodes().at(i).orig is 0 but the timestamp for graph_orig.nodes().at(i) is now window_size - 1
        {
            aggregate_node = graph_orig.nodes().at(i);
            break;
        }
    }
    //(2)-Add edges for this node using the src node information for this node
    for(const int& i: nodeid_proto[aggregate_node->orig()->id()].related_info())
    //use "nodeid_proto[aggregate_node->orig()->id()].related_info()" as an indicator of the timestamp; for all corresponding timestamps
    {
        if(i == window_size - 1) continue;

        for(int j = 0; j < nodes_timeinfo[i].size(); j++)//for one timestamp,check the src nodes of the aggregation node
        {
            if(aggregate_node->orig()->CheckWhetherSrcNode(nodes_timeinfo[i].at(j)->orig()) == true)
            {
                graph_unroll.AddEdge(nodes_timeinfo[i].at(j), aggregate_node);
            }
        }
    }
}

}//namespace singa
