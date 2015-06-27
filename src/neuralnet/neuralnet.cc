#include <algorithm>
#include <queue>

#include "neuralnet/neuralnet.h"
#include "utils/singleton.h"
#include "utils/factory.h"
#include "utils/graph.h"
#include "utils/cluster.h"

#include <iostream>

namespace singa
{
#define CreateLayer(id) CreateInstance(id, Layer)

typedef std::vector<SNode> nodes_timestamp;//keep all the nodes for one timestamp

 // register different types of layers -- no need to change right now
 void NeuralNet::RegisterLayers()
{
    Factory<Layer>* factory=Singleton<Factory<Layer>>::Instance();
    factory->Register("kBridgeDst", CreateLayer(BridgeDstLayer));
    factory->Register("kBridgeSrc", CreateLayer(BridgeSrcLayer));
    factory->Register("kConvolution", CreateLayer(ConvolutionLayer));
    factory->Register("kConcate", CreateLayer(ConcateLayer));
    factory->Register("kDropout", CreateLayer(DropoutLayer));
    factory->Register("kInnerProduct", CreateLayer(InnerProductLayer));
    factory->Register("kLabel", CreateLayer(LabelLayer));
    factory->Register("kLMDBData", CreateLayer(LMDBDataLayer));
    factory->Register("kLRN", CreateLayer(LRNLayer));
    factory->Register("kMnistImage", CreateLayer(MnistImageLayer));
    factory->Register("kPooling", CreateLayer(PoolingLayer));
    factory->Register("kPrefetch", CreateLayer(PrefetchLayer));
    factory->Register("kRGBImage", CreateLayer(RGBImageLayer));
    factory->Register("kReLU", CreateLayer(ReLULayer));
    factory->Register("kShardData", CreateLayer(ShardDataLayer));
    factory->Register("kSlice", CreateLayer(SliceLayer));
    factory->Register("kSoftmaxLoss", CreateLayer(SoftmaxLossLayer));
    factory->Register("kSplit", CreateLayer(SplitLayer));
    factory->Register("kTanh", CreateLayer(TanhLayer));
}

//a generic function for setting up neuralnet -- return a shared pointer which points to a neuralnet
shared_ptr<NeuralNet> NeuralNet::SetupNeuralNet(const NetProto& np, Phase phase,
        int group_size)
{
    NetProto proto;
    proto.set_partition_type(np.partition_type());
    proto.set_win_size(np.win_size());
    // exclude layers if necessary -- for example: the training process and the testing process are mutually exclusive, can only choose one each time
for(auto& layer:np.layer())
    {
        bool include=true;
for(int x: layer.exclude())
        {
            if(x==phase)
                include=false;
        }
        if(include)
        {
            LayerProto* lp=proto.add_layer();
            lp->CopyFrom(layer);
        }
    }
    LOG(INFO)<<"NeuralNet config is "<<proto.DebugString();
    return make_shared<NeuralNet>(proto, group_size);
}

//construction function of neuralnet
NeuralNet::NeuralNet(NetProto net_proto, int group_size)
{
    group_size_=group_size;
    for(int i=0; i<net_proto.layer_size(); i++)
    {
        LayerProto * layer_proto=net_proto.mutable_layer(i);
        if(!layer_proto->has_partition_type())
            layer_proto->set_partition_type(net_proto.partition_type());
    }

    LOG(INFO)<<"Construct Neural Net...";
    // neuralnet without partitioning
    ConstructNeuralNetRNN(net_proto);
    {
        string vis_folder=Cluster::Get()->vis_folder();
        std::ofstream fout(vis_folder+"/nopartition.json", std::ofstream::out);
        fout<<ToString();
        fout.flush();
        fout.close();
    }

    // neuralnet with partitioning - not consider right now
    if(group_size_>1)
    {
        PartitionNeuralNet();
        string vis_folder=Cluster::Get()->vis_folder();
        std::ofstream fout(vis_folder+"/partition.json", std::ofstream::out);
        fout<<ToString();
        fout.flush();
        fout.close();
    }
for(auto layer: layers_)
    {
        DLOG(INFO)<<layer->name();
    }
for(auto& layer: layers_)
    {
for(shared_ptr<Param> p: layer->GetParams())
        {
            params_.push_back(p);
        }
    }
    LOG(INFO)<<"Neural Net constructed";
    // init all data members to avoid conflicts from multi-thread access
    losslayers();
    paramid2param(0);
    datalayers();
    parserlayers();
}

/*
  Implementation for Recurrent Neural Network - for recurrent neural network
- first: (layers -> node in cyclic graph)
- second: unroll the cyclic graph to acyclic graph
- third: (node in acyclic graph -> layers)

- the part in test.cc only corresponds to the first part and second part
- now focus on implementing the third part*/

// For Recurrent Neural Network implementation
// Function7 - if the breaking is correct, store corresponding info; otherwise, recover the edge and try another breaking
void NeuralNet::ConstructNeuralNetRNN(const NetProto& net_proto)
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

    std::cout << "construct original graph which may contain a cycle" << std::endl;
    // 2-after breaking one edge in the cycle and save needed information for unroll (graph_orig (now is acyclic) & correct_breaking_edge)
    std::pair<SNode, SNode> correct_breaking_edge;//this information will be used when unrolling the cyclic graph
    graph_orig.DetectCycleAndSaveCycle();// detect and save the cycle for original graph
    graph_orig.ChangeCycleToEdges();// change the cycle information into edge representation, now in "std::vector<std::pair<SNode, SNode>> cycle_edges"

    for(int i = 0; i < graph_orig.cycle_edges().size(); i++)
    {
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

    //graph_orig.Sort();//topology sort for the current acyclic graph which is constructed by breaking one edge

    // 3-unrolling - constructing unrolled & acyclic graph
    std::cout << "start unrolling..." << std::endl;
    int window_size = net_proto.win_size();
    std::cout << "window size: " << window_size << std::endl;
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
                nodes_j[k] = std::make_shared<Node>(graph_orig.node(j)->name());
                //nodes_j[k]->set_srcnodes(graph_orig.node(j)->srcnodes());
                //nodes_j[k]->set_dstnodes(graph_orig.node(j)->dstnodes());
                nodes_j[k]->set_val(graph_orig.node(j)->val());
                nodes_j[k]->set_color(graph_orig.node(j)->color());
                nodes_j[k]->set_weight(graph_orig.node(j)->weight());
                nodes_j[k]->set_shape(graph_orig.node(j)->shape());
                nodes_j[k]->set_id(graph_orig.node(j)->id());
                nodes_j[k]->set_timestamp(graph_orig.node(j)->timestamp());
                nodes_j[k]->set_orig(graph_orig.node(j)->orig());
                //std::cout << "When initilization, the name of the node is: " <<  nodes_j[k]->name() << std::endl;
                nodes_j[k]->set_orig(graph_orig.node(j));
                //std::cout << "In graph_orig: orig node's name: " << graph_orig.node(j)->name() << " timestamp: " << graph_orig.node(j)->timestamp() << std::endl;
                //std::cout << "orig node's name : " << nodes_j[k]->orig()->name() << " timestamp: " << nodes_j[k]->orig()->timestamp() << std::endl;
                nodes_j[k]->set_timestamp(k);
                nodes_j[k]->set_name(nodes_j[k]->orig()->name() + "@" +std::to_string(k));
                //std::cout << "In graph_orig: orig node's name: " << graph_orig.node(j)->name() << " timestamp: " << graph_orig.node(j)->timestamp() << std::endl;
                //std::cout << "orig node's name : " << nodes_j[k]->orig()->name() << " timestamp: " << nodes_j[k]->orig()->timestamp() << std::endl;
                //std::cout << nodes_j[k]->name() << std::endl;
                graph_.AddNode(nodes_j[k]);
		nodes_timeinfo[k].push_back(nodes_j[k]);
            }
        }
        else// For the nodes which don't need to be unrolled
        {
            SNode new_node;
            //Initialization
            new_node = std::make_shared<Node>(graph_orig.node(j)->name());
            //new_node->set_srcnodes(graph_orig.node(j)->srcnodes());
            //new_node->set_dstnodes(graph_orig.node(j)->dstnodes());
            new_node->set_val(graph_orig.node(j)->val());
            new_node->set_color(graph_orig.node(j)->color());
            new_node->set_weight(graph_orig.node(j)->weight());
            new_node->set_shape(graph_orig.node(j)->shape());
            new_node->set_id(graph_orig.node(j)->id());
            new_node->set_timestamp(graph_orig.node(j)->timestamp());
            new_node->set_orig(graph_orig.node(j)->orig());
            //Update corresponding information
            new_node->set_timestamp(window_size - 1);
            new_node->set_orig(graph_orig.node(j));
            //new_node->set_name(new_node->orig()->name() + "@" +std::to_string(window_size - 1));
            new_node->set_name(new_node->orig()->name());
	    graph_.AddNode(new_node);
	    nodes_timeinfo[window_size - 1].push_back(new_node);
        }
    }

    for(int k = 0; k < window_size; k++)
    {
        std::cout << "number of nodes for timestamp: " << k << " is: " << nodes_timeinfo[k].size() << std::endl;
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
                {
                    graph_.AddEdge(nodes_timeinfo[p].at(pointer2), nodes_timeinfo[p].at(pointer1));
		    std::cout << "adding edge from : " << nodes_timeinfo[p].at(pointer2)->name() << " to : " << nodes_timeinfo[p].at(pointer1)->name() << std::endl;
                }
                else if(nodes_timeinfo[p].at(pointer1)->orig()->CheckWhetherDstNode(nodes_timeinfo[p].at(pointer2)->orig()) == true)
                {
                    graph_.AddEdge(nodes_timeinfo[p].at(pointer1), nodes_timeinfo[p].at(pointer2));
		    std::cout << "adding edge from : " << nodes_timeinfo[p].at(pointer1)->name() << " to : " << nodes_timeinfo[p].at(pointer2)->name() << std::endl;
                }
            }
        }
    }
    std::cout << "adding edges...for different timestamps" << std::endl;
    // Add edges in the graph - different timestamps
    for(int p = 0; p < window_size - 1; p++)// traverse all timestamps
    {
        //for the nodes in the neighboring timestamps
        //std::cout << "test - loop 3" << std::endl;
        for(int pointer1 = 0; pointer1 < nodes_timeinfo[p].size(); pointer1++)// traverse the src node
        {
            //std::cout << "test - loop 2" << std::endl;
            if(nodes_timeinfo[p].at(pointer1)->orig() == correct_breaking_edge.first)
            {
                for(int pointer2 = 0; pointer2 < nodes_timeinfo[p + 1].size(); pointer2++)
                {
                    //std::cout << "test - loop 1" << std::endl;
                    if(nodes_timeinfo[p + 1].at(pointer2)->orig() == correct_breaking_edge.second)
                    {
                        graph_.AddEdge(nodes_timeinfo[p].at(pointer1), nodes_timeinfo[p + 1].at(pointer2));
			std::cout << "successfully add one edge according to correct breaking edge!" << std::endl;
                        std::cout << "adding edge from : " << nodes_timeinfo[p].at(pointer1)->name() << " to : " << nodes_timeinfo[p+1].at(pointer2)->name() << std::endl;
                    }
                }
            }

        }
    }


    //4-Consider the important node with special proto input defined in LayerProto as "repeated int32 related_info"
    //(1)-Find spread node & aggregate node
    std::cout << "consider 2 important nodes - the spread node and the aggregation node" << std::endl;
    SNode spread_node;// the last node whose unroll_decision is false; then turn to true
    SNode aggregate_node;// the first node whose unroll_decision is false; after several nodes which need to be unrolled
    for(int i = 0; i < graph_orig.nodes_size(); i++)
    {
        std::cout << "the nodes in the graph are respectively: " << graph_orig.node(i)->name() << std::endl;
        if(nodeid_proto[graph_orig.node(i)->orig()->id()].unroll_decision() == false && nodeid_proto[graph_orig.node(i + 1)->orig()->id()].unroll_decision() == true)
        {
            spread_node = graph_orig.node(i);
            std::cout << "the spread node is : " << spread_node->name() << std::endl;
            std::cout << "the origin of the spread node is : " << spread_node->orig()->name() << std::endl;
        }
        else if(nodeid_proto[graph_orig.node(i)->orig()->id()].unroll_decision() == true && nodeid_proto[graph_orig.node(i + 1)->orig()->id()].unroll_decision() == false)
        {
            aggregate_node = graph_orig.node(i + 1);
            std::cout << "the aggregate node is : " << aggregate_node->name() << std::endl;
            std::cout << "the origin of the aggregate node is : " << aggregate_node->orig()->name() << std::endl;
            break;
        }

    }

    //update the information in unrolled graph to determine which node is the spread node and which node is the aggregate node respectively
    for(int i = 0; i < graph_.nodes().size(); i++)
    {
        if(graph_.node(i)->orig() == spread_node)
	{
            spread_node = graph_.node(i);
	    std::cout << "spread node is: " << spread_node->name() << std::endl;
        }
	else if(graph_.node(i)->orig() == aggregate_node)
        {
            aggregate_node = graph_.node(i);
	    std::cout << "aggregate node is: " << aggregate_node->name() << std::endl;
        }
    }

    //(2)-Add edges for spread  node using the dst node information & Add edges for aggregate node using the src node information
    std::cout << "add edges for the spread node..." << std::endl;
for(const int& i: nodeid_proto[spread_node->orig()->id()].related_info())
        //use "nodeid_proto[spread_node->orig()->id()].related_info()" as an indicator of the timestamp; for all corresponding timestamps
    {
        if(i == window_size - 1) continue;

        for(int j = 0; j < nodes_timeinfo[i].size(); j++)//for one timestamp,check the dst nodes of the spread node
        {
		if(spread_node->orig()->CheckWhetherDstNode(nodes_timeinfo[i].at(j)->orig()) == true)
            {
		graph_.AddEdge(spread_node, nodes_timeinfo[i].at(j));
                std::cout << "adding edge from : " << spread_node->name() << " to : " << nodes_timeinfo[i].at(j)->name() << std::endl;
            }
        }
    }

    std::cout << "add edges for the aggregation node..." << std::endl;
for(const int& i: nodeid_proto[aggregate_node->orig()->id()].related_info())
        //use "nodeid_proto[aggregate_node->orig()->id()].related_info()" as an indicator of the timestamp; for all corresponding timestamps
    {
        if(i == window_size - 1) continue;

        for(int j = 0; j < nodes_timeinfo[i].size(); j++)//for one timestamp,check the src nodes of the aggregation node
        {
            if(aggregate_node->orig()->CheckWhetherSrcNode(nodes_timeinfo[i].at(j)->orig()) == true)
            {
		graph_.AddEdge(nodes_timeinfo[i].at(j), aggregate_node);
                std::cout << "adding edge from : " << nodes_timeinfo[i].at(j)->name() << " to : " << aggregate_node->name() << std::endl;
            }
        }
    }
    //for testing the unrolling part of the graph
    std::cout << "write the graph information to string and then write to a file..." << std::endl;
    std::cout << "# of nodes in graph_unroll : " << graph_.nodes().size() << std::endl;
    {
        //output the graph information to string and then write to a file
        string vis_folder=singa::Cluster::Get()->vis_folder();
        std::ofstream fout(vis_folder+"/nopartition.json", std::ofstream::out);
        fout<<graph_.ToString();
        fout.flush();
        fout.close();
    }

    //debugging
    std::cout << "Testing before topological ordering" << std::endl;
    for(int i = 0; i < graph_.nodes_size(); i++)
    {
        std::cout << graph_.node(i)->name() << std::endl;
    }


    // the third part to implement
   
    std::cout << std::endl;
    std::cout << "Construct Neural Network..." << std::endl;
    // topology sort

        
    graph_.Sort();
    std::cout << "Testing after topological ordering" << std::endl;
    for(int i = 0; i < graph_.nodes_size(); i++)
    {
        std::cout << graph_.node(i)->name() << std::endl;
    }    
    //std::cout << "pure graph without partition for recurrent neural network\n" << graph_.ToString();

    auto* factory = Singleton<Factory<Layer>>::Instance();

    // create Layers according to topology order
    map<string, LayerProto> protos_unroll;//the two new map data structures store the corresponding information between node and layer
    map<string, string> layer_origlayer;

    std::cout << "Create layers according to topological order..." << std::endl;
    for(SNode node: graph_.nodes())   
    {
	std::cout << "node name: " << node->name() << std::endl;
	std::cout << "orig node name: " << node->orig()->name() << std::endl;
        if(node->orig()->name() == node->name()) // for the nodes which are not unrolled
        {
	    std::cout << "This node is not unrolled..." << std::endl;
            LayerProto new_layer1;
            new_layer1 = nodeid_proto[node->id()];
            new_layer1.name() = node->name();//update the name info for the nodes which are not unrolled
            protos_unroll[new_layer1.name()] = new_layer1;//initialize these two map data structures -- here can also use node->name() as the "key"
            // the following steps are similar to FNN -- this part for "if" and for "else" can be combined
            shared_ptr<Layer> layer(factory->Create(protos_unroll[node->name()].type()));//use the proto to create a new layer
            layer->Init(protos_unroll[node->name()]);
            name2layer_[node->name()] = layer;
            layers_.push_back(layer);
        }
        else{ // for the nodes which are unrolled
	    std::cout << "This node is unrolled..." << std::endl;
            LayerProto new_layer2;
            new_layer2 = nodeid_proto[node->orig()->id()];
            new_layer2.name() = node->name();//update the name info for the nodes which are unrolled
            protos_unroll[new_layer2.name()] = new_layer2;//initialize these two map data structures -- here can also use node->name() as the "key"
            // the following steps are similar to FNN
            shared_ptr<Layer> layer(factory->Create(protos_unroll[node->name()].type()));
            layer->Init(protos_unroll[node->name()]);
            name2layer_[node->name()]=layer;
            layers_.push_back(layer);
        }

    }



    // connect Layers using the srclayer & dstlayer information
    std::cout << "Connect layers..." << std::endl;
    for(SNode node: graph_.nodes()) 
    {
        auto layer=name2layer_[node->name()];
        for(SNode dst: node->dstnodes())
            layer->AddDstLayer(name2layer_[dst->name()]);
        for(SNode src: node->srcnodes())
            layer->AddSrcLayer(name2layer_[src->name()]);
    }

    //initialize the mapping between layer and its corresponding orig layer
    std::cout << "Initialize the mapping between layers and their corresponding original layers..." << std::endl;
    for(SNode node: graph_.nodes())
    {
        if(node->orig()->name() == node->name()) // for the nodes which are not unrolled, set the origlayer as itself
        {
            layer_origlayer[node->name()] = node->name();
        }
        else// for the nodes which are unrolled, i.e., node->orig() != node
        {
            layer_origlayer[node->name()] = node->orig()->name();
        }
    }

    // setup layer properties, e.g., shapes
    std::cout << "Set up layer properties..." << std::endl;
    int paramid=0; // for one neuralnet, all the paramaters should have a distinct and ascending-order id

    for(auto& layer: layers_)
    {
        layer->Setup();// proto & srclayers
        if(layer->name() == layer_origlayer[layer->name()])// for the layers which are not obtained by unrolling
        {
            for(auto param: layer->GetParams()) //need to share parameters from its orig if any
            param->set_id(paramid++);
        }
        else // for the layers which are obtained by unrolling
        {
            const auto& params = layer->GetParams();
            const auto& origparams = name2layer_[layer_origlayer[layer->name()]]->GetParams();
            CHECK_EQ(params.size(), origparams.size());//used to check whether the setup function is correctly executed
            for(int i = 0; i < params.size(); i++)
            {
                params[i]->ShareData(origparams[i]);
                params[i]->set_id(paramid++);
            }
        }

    }
    LOG(INFO)<<"network graph witout partition\n"<<ToString();
}

// not consider this function right now - partition the neuralnet
void NeuralNet::PartitionNeuralNet()
{
    graph_=CreatePartitonedGraph(layers_, name2layer_);
    //DLOG(ERROR)<<"pure graph after partition\n"<<graph_.ToString();
    map<string, shared_ptr<Layer>> name2layer(name2layer_);
    map<string, vector<shared_ptr<Layer>>> share_param_layers;
    name2layer_.clear();
    layers_.clear();
    int gsize=group_size_;
    auto* factory=Singleton<Factory<Layer>>::Instance();
    // create Layers according to topology order
for(SNode node: graph_.nodes())
    {
        LayerProto proto;
        proto.set_name(node->name());
        proto.set_partitionid(node->val().partitionid);
        const string& origin=node->val().origin;
        if (origin=="kSlice")
        {
            proto.set_type(origin);
            SliceProto *slice=proto.mutable_slice_param();
            slice->set_slice_dimension(node->val().slice_dimension);
            slice->set_slice_num(node->dstnodes().size());
        }
        else if(origin== "kConcate")
        {
            proto.set_type(origin);
            ConcateProto *concate=proto.mutable_concate_param();
            concate->set_concate_dimension(node->val().concate_dimension);
            concate->set_concate_num(node->srcnodes().size());
        }
        else if(origin=="kSplit")
        {
            proto.set_type(origin);
            SplitProto *split=proto.mutable_split_param();
            split->set_num_splits(node->dstnodes().size());
        }
        else if(origin=="kBridgeSrc" || origin== "kBridgeDst")
        {
            proto.set_type(origin);
        }
        else
        {
            CHECK(name2layer.find(node->val().origin)!=name2layer_.end())
            <<"Unkown origin for node "<<node->val().origin;
        }
        shared_ptr<Layer> newlayer;
        if(proto.has_type())
        {
            // layers added due to partition
            shared_ptr<Layer> layer(factory->Create(proto.type()));
            layer->Init(proto);
            newlayer=layer;
        }
        else
        {
            // partitioned layers from origin neuralnet
            auto oldlayer=name2layer.at(node->val().origin);
            vector<int> shape=oldlayer->shape(nullptr);
            if(oldlayer->partition_type()==kNone)
            {
                newlayer=oldlayer;
            }
            else
            {
                int pdim=oldlayer->partition_dimension();
                shape[pdim]=shape[pdim]/gsize+
                            ((node->val().partitionid==gsize-1)?shape[pdim]%gsize:0);
                shared_ptr<Layer> layer(factory->Create(oldlayer->type()));
                layer->Init(*oldlayer, shape);
                layer->set_name(node->name());
                newlayer=layer;
                if(oldlayer->partition_type()==kDataPartition)
                    share_param_layers[node->val().origin].push_back(newlayer);
            }
            newlayer->set_partitionid(node->val().partitionid);
        }
        layers_.push_back(newlayer);
        name2layer_[node->name()]=newlayer;
    }

    // connect Layers.
for(SNode node: graph_.nodes())
    {
        auto layer=name2layer_[node->name()];
        layer->ClearDstLayers();
for(SNode dst: node->dstnodes())
            layer->AddDstLayer(name2layer_[dst->name()]);
        layer->ClearSrcLayers();
for(SNode src: node->srcnodes())
            layer->AddSrcLayer(name2layer_[src->name()]);
    }

    LOG(INFO)<<"Adjacency matrix\n"<<ToAdjacency();

    // set up layers after
    int paramid=0;
for(shared_ptr<Layer> layer: layers_)
    {
        const vector<int>& shape=layer->shape(nullptr);
        layer->SetupAfterPartition();
for(auto param: layer->GetParams())
            param->set_id(paramid++);
        const vector<int>& newshape=layer->shape(nullptr);
        if(shape.size())
            CHECK(std::equal(shape.begin(),shape.end(),newshape.begin()));
    }

    // share Params for layers generated from the same origin layer due to
    // data partition
for(auto & entry: share_param_layers)
    {
        auto layers= entry.second;
        auto owner=layers.begin();
        auto owner_params=(*owner)->GetParams();
        for(auto it=owner+1; it!=layers.end(); it++)
        {
            auto params=(*it)->GetParams();
            CHECK_EQ(params.size(), owner_params.size());
            for(size_t i=0; i<params.size(); i++)
                params.at(i)->ShareData(owner_params.at(i));
        }
    }
    LOG(INFO)<<"network graph after partition layers\n"<<ToString();
}

// not consider this function right now
Graph NeuralNet::CreatePartitonedGraph(const vector<shared_ptr<Layer>>& layers,
                                       const map<string, shared_ptr<Layer>>& name2layer)
{
    Graph graph;
    // partition origin nodes/layers
    map<string, vector<SNode>> layer2nodes; //from name of original layer to nodes
    int gsize=group_size_;
for(const auto& layer: layers)
    {
        vector<SNode> nodes;
        if(layer->partition_type()==kDataPartition||
                layer->partition_type()==kLayerPartition)
        {
            char suffix[4];
            for(int i=0; i<gsize; i++)
            {
                sprintf(suffix, "%02d", i);
                // differentiate partitions
                string nodename=layer->name()+"@"+string(suffix);
                auto node=graph.AddNode(nodename, LayerInfo {layer->name(), i,-1,-1});
                nodes.push_back(node);
            }
        }
        else if(layer->partition_type()==kNone)
        {
            auto node=graph.AddNode(layer->name(),
                                    LayerInfo {layer->name(), 0,-1,-1});
            nodes.push_back(node);
        }
        else
        {
            LOG(FATAL)<<"Unknown partition type "<<layer->partition_type();
        }
        layer2nodes[layer->name()]=nodes;
    }

    // connect nodes, nodes for ConcateLayer and SliceLayer are added.
for(shared_ptr<Layer> layer: layers)
    {
        string name=layer->name();
        PartitionType type=layer->partition_type();
        const vector<SNode>& nodes=layer2nodes.at(name);
        for(int srcid=0; srcid<layer->srclayers_size(); srcid++)
        {
            shared_ptr<Layer> srclayer=layer->srclayers()[srcid];
            string srcname=srclayer->name();
            const vector<SNode> srcnodes=layer2nodes.at(srcname);
            PartitionType srctype=srclayer->partition_type();
            ConnectionType connection=layer->connection_type(srcid);
            if(srctype==kNone)
            {
                CHECK_EQ(srcnodes.size(),1)
                <<"local layer "<<srcname<<" should not be partitioned";
                SNode srcnode=srcnodes[0];
                if(type==kDataPartition||(type==kLayerPartition&&connection==kOneToOne))
                {
                    LayerInfo info=srcnode->val();
                    info.slice_dimension=name2layer.at(name)->partition_dimension();
                    graph.InsertSliceNode(srcnode, nodes, info);
                }
                else if(type==kNone)
                {
                    CHECK_EQ(nodes.size(),1)
                    <<"local layer "<<name<<" should not be nodeed";
                    graph.AddEdge(srcnode, nodes[0]);
                }
                else     // type==kLayerPartition&&connection==kOneToAll
                {
                    graph.InsertSplitNode(srcnode, nodes);
                }
            }
            else if((type==kNone
                     &&(srctype==kDataPartition||srctype==kLayerPartition))
                    ||(type==kLayerPartition&&connection==kOneToAll&&
                       (srctype==kDataPartition||srctype==kLayerPartition)))
            {
                // copy/concate the whole srclayer for every dst partition
for(SNode node:nodes)
                {
                    LayerInfo info=node->val();
                    info.concate_dimension=name2layer.at(srcname)->partition_dimension();
                    CHECK_GE(info.concate_dimension,0);
                    graph.InsertConcateNode(srcnodes, node, info);
                }
            }
            else if((srctype==kLayerPartition&&type==kDataPartition)
                    || (srctype==kDataPartition&&type==kLayerPartition))
            {
                // the most complext scenario
                vector<SNode> slicenodes;
for(SNode srcnode: srcnodes)
                {
                    LayerInfo info=srcnode->val();
                    info.slice_dimension=name2layer.at(name)->partition_dimension();
                    slicenodes.push_back(graph.InsertSliceNode(srcnode, nodes,
                                         info, false));
                }
for(SNode node: nodes)
                {
                    LayerInfo info=node->val();
                    info.concate_dimension=name2layer.at(srcname)->partition_dimension();
                    CHECK_GE(info.concate_dimension,0);
                    graph.InsertConcateNode(slicenodes, node, info);
                }
            }
            else if((srctype==kDataPartition&&type==kDataPartition)||
                    (srctype==kLayerPartition&&type==kLayerPartition&&
                     layer->connection_type(srcid)==kOneToOne))
            {
                CHECK_EQ(srcnodes.size(), nodes.size());
                for(size_t i=0; i<srcnodes.size(); i++)
                {
                    graph.AddEdge(srcnodes[i], nodes[i]);
                }
            }
        }
    }
    // must do topology sort, because we have added new nodes.
    graph.Sort();
    //LOG(ERROR)<<graph.ToString();

    // add node for split layer
    bool data_node=true;
    vector<SNode> oldnodes=graph.nodes();
for(SNode node: oldnodes)
    {
        if(node->dstnodes_size()>1&&node->val().origin!="kSlice"
                &&node->val().origin!="kSplit"&&!data_node)
        {
            vector<SNode> dstnodes=node->dstnodes();
for(SNode dst: dstnodes)
                graph.RemoveEdge(node, dst);
            graph.InsertSplitNode(node, dstnodes);
        }
        data_node=false;
    }

    // add bridge
    oldnodes=graph.nodes();
for(SNode node: oldnodes)
    {
        vector<SNode> dstnodes=node->dstnodes();
        for(size_t i=0; i<dstnodes.size(); i++)
        {
            SNode dstnode=dstnodes.at(i);
            if(node->val().partitionid!=dstnode->val().partitionid)
            {
                graph.RemoveEdge(node, dstnode);
                graph.InsertBridgeNode(node, dstnode);
            }
        }
    }
    graph.Sort();
    return graph;
}

std::string NeuralNet::ToString()
{
    map<string, string> info;
for(auto layer: layers_)
    {
        info[layer->name()]=IntVecToString(layer->shape(nullptr));
        string type=layer->type();
    }
    return graph_.ToString(info);
}

std::string NeuralNet::ToAdjacency()
{
    string disp="";
for(auto& layer: layers_)
    {
        disp+=layer->name()+": ";
for(const auto& dst: layer->dstlayers())
            disp+=dst->name()+", ";
        disp+="\n";
    }
    return disp;
}


void NeuralNet::ToProto(NetProto *proto, bool copyData)
{
    proto->clear_layer();
}

string NeuralNet::DebugInfo()
{
    string ret;
    char display[4096];
for(auto& layer: layers_)
    {
        if(!layer->is_datalayer())
        {
            sprintf(display, "Forward layer  %10s data norm1 %13.9f\n",
                    layer->name().c_str(), layer->data(nullptr).asum_data());
            ret+=string(display);
        }
    }
    for (auto it = layers_.rbegin(); it != layers_.rend(); it++)
    {
        shared_ptr<Layer> layer=*it;
        if(!(layer->is_datalayer()||layer->is_losslayer()||layer->is_parserlayer()))
        {
            sprintf(display, "Backward layer %10s grad norm1 %13.9f\n",
                    layer->name().c_str(), layer->grad(nullptr).asum_data());
            ret+=string(display);
        }
    }
for(auto& layer: layers_)
    {
for(auto param: layer->GetParams())
        {
            sprintf(display, "Layer %10s, param id %2d, name %10s,\
          value norm1 %13.9f, grad norm1 %13.9f\n",
                    layer->name().c_str(), param->id(), param->name().c_str(),
                    param->data().asum_data(), param->grad().asum_data());
            ret+=string(display);
        }
    }
    return ret;
}
void NeuralNet::ShareParams(shared_ptr<NeuralNet> other, int flag)
{
for(auto& layer: layers_)
    {
        auto otherlayer=other->name2layer(layer->name());
        if(otherlayer!=nullptr)
        {
            const auto& otherparams=otherlayer->GetParams();
            const auto& params=layer->GetParams();
            CHECK_EQ(params.size(), otherparams.size());
            for(size_t i=0; i<params.size(); i++)
            {
                params[i]->ShareData(otherparams[i]);
            }
        }
    }
}

}  // namespace singa
