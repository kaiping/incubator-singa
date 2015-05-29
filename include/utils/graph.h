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
    bool CheckInputNode(SNode node);
    bool CheckOutputNode(SNode node);

private:
    string name_;
    vector<SNode> srcnodes_;
    vector<SNode> dstnodes_;

    V val_;
    // properties
    string color_, weight_, shape_;

    // for Recurrent Neural Network implementation
    int id;//corresponding to the occurence order in vector "nodes_"
    int timestamp;// used when unrolling the cyclic graph
    SNode orig;
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
        nodes_.back().id = nodes.size() - 1;
        return nodes_.back();
    }
    const SNode& AddNode(string name)
    {
        nodes_.push_back(make_shared<Node>(name));
        name2node_[name]=nodes_.back();
        // for Recurrent Neural Network implementation
        nodes_.back().id = nodes.size() - 1;
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
    };

    const SNode& node(string name) const
    {
        CHECK(name2node_.find(name)!= name2node_.end())
        <<"can't find dst node "<<name;
        return name2node_.at(name);
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

    // Function1 - detect cycle in the graph and save the graph (now can only deal with the single-cycle situation)
    void DetectCycleAndSaveCycle();

    // Function2 - start from node v to detect whether there is a cycle
    std::stack<SNode> DFSForDetectCycleAndSaveCycle(SNode node_v);

    // for Function2
    bool hasCycle();
    //std::stack<SNode> cycle();

    // Function3 - change the cycle information to the edge (node pairs) representation
    void ChangeCycleToEdges(std::stack<SNode> cycle_stack);

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
    bool marked[];// to denote whether this node has been visited
    int edgeTo[];// to record the pre-visited node order
    stack<SNode> cycle;
    std::vector<std::pair<SNode, SNode>> cycle_edges;// another representation of all edges in the graph
    bool onStack[];// to denote whether this node is being visited
};
#endif // INCLUDE_UTILS_GRAPH_H_
