#ifndef INCLUDE_UTILS_GRAPH_H_
#define INCLUDE_UTILS_GRAPH_H_
#include <glog/logging.h>
#include <vector>
#include <string>
#include <map>
#include <stack>
#include <memory>

using std::vector;
using std::string;
using std::map;
using std::stack;
using std::pair;
using std::shared_ptr;
using std::make_shared;


typedef struct _LayerInfo
{
    // origin identifies the origin of this node, i.e., the corresponding layer
    string origin;
    int locationid;// locationidation id;
    int partitionid;
    int slice_dimension;
    int concate_dimension;
} LayerInfo;
typedef LayerInfo V;


class Node;
typedef shared_ptr<Node> SNode;

class Node
{
public:
    typedef shared_ptr<Node> SNode;
    Node(string name): name_(name) {}
    Node(string name, const V& v):
        name_(name), val_(v) {}

    void AddDstNode(SNode dstnode)
    {
        dstnodes_.push_back(dstnode);
    }
    void AddSrcNode(SNode srcnode)
    {
        srcnodes_.push_back(srcnode);
    }

    void RemoveDstNode(SNode dst)
    {
        auto iter=dstnodes_.begin();
        while((*iter)->name_!=dst->name_&&iter!=dstnodes_.end()) iter++;
        CHECK((*iter)->name_==dst->name_);
        dstnodes_.erase(iter);
    }
    void RemoveSrcNode(SNode src)
    {
        auto iter=srcnodes_.begin();
        while((*iter)->name_!=src->name_&&iter!=srcnodes_.end()) iter++;
        CHECK((*iter)->name_==src->name_);
        srcnodes_.erase(iter);
    }
    const string& name() const
    {
        return name_;
    }
    const V& val() const
    {
        return val_;
    }
    const SNode srcnodes(int k) const
    {
        return srcnodes_[k];
    }
    const SNode dstnodes(int k) const
    {
        return dstnodes_[k];
    }
    const vector<SNode>& srcnodes() const
    {
        return srcnodes_;
    }
    const vector<SNode>& dstnodes() const
    {
        return dstnodes_;
    }
    int  dstnodes_size() const
    {
        return dstnodes_.size();
    }
    int  srcnodes_size() const
    {
        return srcnodes_.size();
    }

    // for Recurrent Neural Network implementation
    const string& color() const
    {
        return color_;
    }

    const string& weight() const
    {
        return weight_;
    }

    const string& shape() const
    {
        return shape_;
    }

    const int& id() const
    {
        return id_;
    }

    const int& timestamp() const
    {
        return timestamp_;
    }

    const SNode& orig() const
    {
        return orig_;
    }

    void set_name(string k)
    {
       name_ = k;
    }

    void set_srcnodes(vector<SNode> k)
    {
	srcnodes_ = k;
    }

    void set_dstnodes(vector<SNode> k)
    {
	dstnodes_ = k;
    }

    void set_val(V k)
    {
	val_ = k;
    }

    void set_color(string k)
    {
	color_ = k;
    }

    void set_weight(string k)
    {
	weight_ = k;
    }

    void set_shape(string k)
    {
	shape_ = k;
    }

    void set_id(int k)
    {
       id_ = k;
    }

    void set_timestamp(int k)
    {
       timestamp_ = k;
    }

    void set_orig(SNode p)
    {
       orig_ = p;
    }

    bool CheckInputNode() const;
    bool CheckOutputNode() const;
    bool CheckWhetherSrcNode(SNode node) const;//check whether the input node is one of src nodes of this node
    bool CheckWhetherDstNode(SNode node) const;//check whether the input node is one of dst nodes of this node

private:
    string name_;
    vector<SNode> srcnodes_;
    vector<SNode> dstnodes_;

    V val_;
    // properties
    string color_, weight_, shape_;

    // for Recurrent Neural Network implementation
    int id_;//corresponding to the occurence order in vector "nodes_"
    int timestamp_;// used when unrolling the cyclic graph
    SNode orig_;
};


/**
 * For partition neuralnet and displaying the neuralnet structure
 */
class Graph
{
public:
    Graph() {}
    void Sort();
    const SNode& AddNode(string name, V origin)
    {
        nodes_.push_back(make_shared<Node>(name, origin));
        name2node_[name]=nodes_.back();
        // for Recurrent Neural Network implementation
        nodes_.back()->set_id(nodes_.size() - 1);
        nodes_.back()->set_timestamp(0); // default timestamp value 
        nodes_.back()->set_orig(nodes_.back()); // By default: use itself as the node's orig
        return nodes_.back();
    }
    const SNode& AddNode(string name)
    {
        nodes_.push_back(make_shared<Node>(name));
        name2node_[name]=nodes_.back();
        // for Recurrent Neural Network implementation
        nodes_.back()->set_id(nodes_.size() - 1);
        nodes_.back()->set_timestamp(0); // default timestamp value
        nodes_.back()->set_orig(nodes_.back()); // By default: use itself as the node's orig
        return nodes_.back();
    }

    //For RNN implementation - constructing graph when unrolling
    const SNode& AddNode(SNode node)
    {
        nodes_.push_back(node);
        return nodes_.back();
    }

    void AddEdge(SNode srcnode, SNode dstnode)
    {
        srcnode->AddDstNode(dstnode);
        dstnode->AddSrcNode(srcnode);
    }

    void AddEdge(const string& src, const string& dst)
    {
        CHECK(name2node_.find(src)!=name2node_.end())<<"can't find src node "<<src;
        CHECK(name2node_.find(dst)!=name2node_.end())<<"can't find dst node "<<dst;

        SNode srcnode=name2node_[src], dstnode=name2node_[dst];
        AddEdge(srcnode, dstnode);
    }

    void RemoveEdge(const string &src, const string& dst)
    {
        CHECK(name2node_.find(src)!=name2node_.end())<<"can't find src node "<<src;
        CHECK(name2node_.find(dst)!=name2node_.end())<<"can't find dst node "<<dst;

        SNode srcnode=name2node_[src], dstnode=name2node_[dst];
        RemoveEdge(srcnode, dstnode);
    }

    void RemoveEdge(SNode src, SNode dst)
    {
        src->RemoveDstNode(dst);
        dst->RemoveSrcNode(src);
    }

    const vector<SNode>& nodes() const
    {
        return nodes_;
    }

    const int nodes_size() const
    {
	return nodes_.size();
    }

    const SNode& node(string name) const
    {
        CHECK(name2node_.find(name)!= name2node_.end())
        <<"can't find dst node "<<name;
        return name2node_.at(name);
    }

    const SNode& node(int k) const
    {
        return nodes_[k];
    }


    const string ToString() const;
    const string ToString(const map<string, string>& info) const ;

    bool Check() const;

    SNode InsertSliceNode(SNode srcnode, const vector<SNode>& dstnodes,
                          const V& info, bool connect_dst=true);
    SNode InsertConcateNode(const vector<SNode>&srcnodes, SNode dstnode,
                            const V& info);
    SNode InsertSplitNode(SNode srcnode, const vector<SNode>& dstnodes);
    std::pair<SNode, SNode> InsertBridgeNode(SNode srcnode, SNode dstnode);
    void topology_sort_inner(SNode node, map<string, bool> *visited,
                             std::stack<string> *stack);

    // for Recurrent Neural Network implementation
    const vector<bool>& marked() const
    {
        return marked_;
    }

    const vector<int>& edge_to() const
    {
        return edge_to_;
    }

    const stack<SNode>& cycle() const
    {
        return cycle_;
    }

    const vector<pair<SNode, SNode> >& cycle_edges() const
    {
        return cycle_edges_;
    }

    const vector<bool>& on_stack() const
    {
        return on_stack_;
    }

    // Function1 - detect cycle in the graph and save the graph (now can only deal with the single-cycle situation)
    void DetectCycleAndSaveCycle();

    // Function2 - start from node v to detect whether there is a cycle
    void DFSForDetectCycleAndSaveCycle(SNode node_v);

    // for Function2
    bool hasCycle() const;
    //std::stack<SNode> cycle();

    // Function3 - change the cycle information to the edge (node pairs) representation
    void ChangeCycleToEdges();

    // Function4 - break one edge in the cycle and update the edge information
    void BreakEdge(std::pair<SNode, SNode> break_edge);//update the edge information in the whole graph according to the edge

    // Function5 - recover the breaking of one edge in the cycle and update the edge information
    void RecoverEdge(std::pair<SNode, SNode> recover_edge);// update the edge information in the whole graph according to the edge

    // Function6 - check whether the breaking is correct
    bool CheckCorrectBreaking(std::pair<SNode, SNode> break_edge);

private:
    vector<SNode> nodes_;
    map<string, SNode> name2node_;

    // for Recurrent Neural Network

    /*bool marked[];// to denote whether this node has been visited*/
    std::vector<bool> marked_;// to denote whether this node has been visited

    /*int edgeTo[];// to record the pre-visited node order*/
    std::vector<int> edge_to_;// to record the pre-visited node order

    stack<SNode> cycle_;

    std::vector<std::pair<SNode, SNode> > cycle_edges_;// another representation of all edges in the graph

    /*bool onStack[];// to denote whether this node is being visited*/
    std::vector<bool> on_stack_;// to denote whether this node is being visited
};
#endif // INCLUDE_UTILS_GRAPH_H_
