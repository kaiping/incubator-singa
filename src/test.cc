#include <gflags/gflags.h>
#include <glog/logging.h>
#include "trainer/trainer.h"
#include <algorithm>
#include <queue>
#include <vector>
#include "utils/graph.h"
#include "utils/singleton.h"
#include "utils/factory.h"
#include "utils/cluster.h"
#include "proto/model.pb.h"
#include <iostream>

/**
 * \file main.cc is the main entry of SINGA, like the driver program for Hadoop.
 *
 * 1. Users register their own implemented classes, e.g., layer, updater, etc.
 * 2. Users prepare the google protobuf object for the model configuration and
 * the cluster configuration.
 * 3. Users call trainer to start the training.
 *
 * TODO
 * 1. Add the resume function to continue training from a previously stopped
 * point.
 * 2. Add helper functions for users to configure their model and cluster
 * easily, e.g., AddLayer(layer_type, source_layers, meta_data).
 */

DEFINE_int32(procsID, 0, "Global process ID");
DEFINE_string(cluster, "examples/mnist/cluster.conf", "Cluster config file");
DEFINE_string(model, "examples/mnist/conv.conf", "Model config file");
typedef std::vector<SNode> nodes_timestamp;//keep all the nodes for one timestamp

/**
 * Register layers, and other customizable classes.
 *
 * If users want to use their own implemented classes, they should register
 * them here. Refer to the Worker::RegisterDefaultClasses()
 */
void RegisterClasses(const singa::ModelProto& proto){
}

// For Recurrent Neural Network implementation
// Function7 - if the breaking is correct, store corresponding info; otherwise, recover the edge and try another breaking
void ConstructUnrolledGraphForRNN(const singa::NetProto& net_proto)
{
    Graph graph_orig;// original graph may have a cycle

    // 1-construct original graph which may contain a cycle, use each SNode can still access the LayerProto information
    map<string, singa::LayerProto> protos;// store the corresponding information between layer name and layer_proto
    map<int, singa::LayerProto> nodeid_proto; // store the correponding information between SNode ID and LayerProto
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

    std::cout << "construct original graph which may contain a cycle" << std::endl;
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
            std::cout << "correct breaking edge: ( " << correct_breaking_edge.first->name() << " , " << correct_breaking_edge.second->name() << " )" << std::endl;
	    break;
        }
        else
        {
            graph_orig.RecoverEdge(graph_orig.cycle_edges().at(i));//recover the edge which has been broken
        }
    }
    std::cout << "Break cycle" << std::endl;
    std::cout << "topological sorting" << std::endl;
    graph_orig.Sort();//topology sort for the current acyclic graph which is constructed by breaking one edge

    // 3-unrolling - constructing unrolled & acyclic graph
    std::cout << "start unrolling..." << std::endl;
    int window_size = net_proto.win_size();
    Graph graph_unroll;
    nodes_timestamp nodes_timeinfo[window_size];//keep the node information grouping by each timestamp
    // Add nodes in the graph

    std::cout << "adding nodes..." << std::endl;
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

    std::cout << "adding edges..." << std::endl;
    std::cout << "adding edges...for the same timestamp" << std::endl;
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

    std::cout << "adding edges...for different timestamps" << std::endl;
    // Add edges in the graph - different timestamps
    for(int p = 0; p < window_size; p++)// traverse all timestamps
    {
        //for the nodes in the neighboring timestamps
	std::cout << "test - loop 3" << std::endl;
        for(int pointer1 = 0; pointer1 < nodes_timeinfo[p].size(); pointer1++)// traverse the src node
        {
	    std::cout << "test - loop 2" << std::endl;
            if(pointer1 == nodes_timeinfo[p].size() - 1) break;// Not consider the last timestamp
            else if(nodes_timeinfo[p].at(pointer1)->orig() == correct_breaking_edge.first)
            {
                for(int pointer2 = pointer1 + 1; pointer2 < nodes_timeinfo[p + 1].size(); pointer2++)
                {
		    std::cout << "test - loop 1" << std::endl;
                    if(nodes_timeinfo[p + 1].at(pointer2)->orig() == correct_breaking_edge.second)
                        graph_unroll.AddEdge(nodes_timeinfo[p].at(pointer1), nodes_timeinfo[p + 1].at(pointer2));
                }
            }

        }
    }


    //4-Consider the important node with special proto input defined in LayerProto as "repeated int32 related_info"
    //(1)-Find this node
    std::cout << "consider the important node - the aggregation node" << std::endl;
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
    std::cout << "add edges for the aggregation node..." << std::endl;
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

    std::cout << "write the graph information to string and then write to a file..." << std::endl;
  {//output the graph information to string and then write to a file
    string vis_folder=singa::Cluster::Get()->vis_folder();
    std::ofstream fout(vis_folder+"/nopartition.json", std::ofstream::out);
    fout<<graph_unroll.ToString();
    fout.flush();
    fout.close();
  }


}




int main(int argc, char **argv) {
  // TODO set log dir
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  /*No need to consider the cluster.conf file right now*/
  singa::ClusterProto cluster;
  singa::ReadProtoFromTextFile(FLAGS_cluster.c_str(), &cluster);//but can use only one master and one worker
  //std::cout << "have read cluster information..." << std:: endl;
  singa::ModelProto model;
  singa::ReadProtoFromTextFile(FLAGS_model.c_str(), &model);
  //std::cout << "have read model information..." << std::endl;
  LOG(INFO)<<"The cluster config is\n"<<cluster.DebugString();
  LOG(INFO)<<"The model config is\n"<<model.DebugString();// for checking whether the configuration file has been correctly read

  //RegisterClasses(model);
  //singa::Trainer trainer;
  //trainer.Start(model, cluster, FLAGS_procsID);

  //The input parameter for the function ConstructUnrolledGraphForRNN should be like: const NetProto &net_proto
  ConstructUnrolledGraphForRNN(model.neuralnet());// in file: model.proto, there is one field: optional NetProto neuralnet = 40;


  return 0;
}
