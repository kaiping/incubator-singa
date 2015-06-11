#include <algorithm>
#include "utils/graph.h"

// for Recurrent Neural Network implementation
bool Node::CheckInputNode() const
{
    if(srcnodes_size() == 0) return true;
    else return false;
}

bool Node::CheckOutputNode() const
{
    if(dstnodes_size() == 0) return true;
    else return false;
}

bool Node::CheckWhetherSrcNode(SNode node) const//check whether the input node is one of src nodes of this node in class
{
    for(int i = 0; i < srcnodes_.size(); i++)
    {
        if(node == srcnodes_[i])
        {
            return true;
        }
    }
    return false;
}

bool Node::CheckWhetherDstNode(SNode node) const//check whether the input node is one of dst nodes of this node
{
    for(int i = 0; i < dstnodes_.size(); i++)
    {
        if(node == dstnodes_[i])
        {
            return true;
        }
    }
    return false;
}

const string Graph::ToString() const
{
    map<string, string> info;
    return ToString(info);
}

const string Graph::ToString(const map<string, string>& info) const
{
    map<string, int> nodeid;
    string disp="{\"directed\":1,\n";

    // add nodes
    disp+="\"nodes\":[\n";
    bool first=true;

    vector<string> colors= {"red", "blue", "black", "green"};
    // see for more shapes at http://www.graphviz.org/doc/info/shapes.html
    vector<string> shapes= {"box", "ellipse"};
    int id=0;
for(auto node: nodes_)
    {
        char str[1024];
        string name=node->name();
        string color=colors[(node->val().locationid)%colors.size()];
        string shape;
        string origin=node->val().origin;
        if(origin=="kSlice"||origin=="kConcate"||origin=="kSplit"
                ||origin=="kBridgeSrc"||origin=="kBridgeDst")
            shape=shapes[1];
        else
            shape=shapes[0];
        sprintf(str, "{\"id\":\"%s%s\", \"color\":\"%s\",\"shape\":\"%s\"}\n",
                name.c_str(), info.find(name)!=info.end()?info.at(name).c_str():"",
                color.c_str(), shape.c_str());
        if(!first)
            disp+=",";
        else
            first=false;
        disp+=string(str);
        nodeid[name]=id++;
    }
    disp+="]\n,";

    // add edges
    disp+="\"links\":[\n";
    first=true;
for(auto src: nodes_)
for(auto dst: src->dstnodes())
        {
            char str[1024];
            sprintf(str, "{\"source\":%d, \"target\":%d, \"color\":\"%s\"}\n",
                    nodeid[src->name()], nodeid[dst->name()], "black");
            if(!first)
                disp+=",";
            else
                first=false;
            disp+=string(str);
        }
    disp+="]\n";
    return disp+"}";
}

bool Graph::Check() const
{
    return true;
}


// visited all dst nodes and then push current node into the stack
void Graph::topology_sort_inner(SNode node,
                                map<string, bool> *visited,
                                std::stack<string> *stack)
{
    (*visited)[node->name()] = true;
    const vector<SNode>& dstnodes=node->dstnodes();
    for (auto it=dstnodes.rbegin(); it!=dstnodes.rend(); it++)
    {
        if ((*visited)[(*it)->name()])
            continue;
        topology_sort_inner((*it),visited, stack);
    }
    stack->push(node->name());
}

// sort to make `bottom' nodes be placed in the front positions
void Graph::Sort()
{
    // adjacent list from upper layers to lower layers
    std::map<string, bool> visited;
    // prepare adjacent list; input layers will be processed firstly,
    // hence no need to sort them (mark them as visited)
for (SNode node: nodes_)
    {
        visited[node->name()] = false;
    }
    // the `top' layer in the net will be placed at the bottom of the stack
    // and then be processed (i.e., forward) at last
    std::stack<string > stack;
for (SNode node: nodes_)
    {
        if (visited[node->name()] == false)
            topology_sort_inner(node, &visited, &stack);
    }
    nodes_.clear();

    while (!stack.empty())
    {
        nodes_.push_back(name2node_[stack.top()]);
        stack.pop();
    }
}



SNode Graph::InsertSliceNode(SNode srcnode, const vector<SNode>& dstnodes,
                             const V& info, bool connect_dst)
{
    V myinfo=info;
    myinfo.origin="kSlice";
    SNode node=AddNode("slice-"+srcnode->name(),myinfo);
    AddEdge(srcnode, node);
    if(connect_dst)
for(SNode dst: dstnodes)
            AddEdge(node, dst);
    return node;
}
SNode Graph::InsertConcateNode(const vector<SNode>&srcnodes, SNode dstnode,
                               const V& info)
{
    V myinfo=info;
    myinfo.origin="kConcate";
    SNode node=AddNode("concate-"+dstnode->name(),myinfo);
    AddEdge(node, dstnode);
for(SNode src: srcnodes)
        AddEdge(src, node);
    return node;
}
SNode Graph::InsertSplitNode(SNode srcnode, const vector<SNode>& dstnodes)
{
    V myinfo=srcnode->val();
    myinfo.origin="kSplit";
    SNode node=AddNode("split-"+srcnode->name(), myinfo);
    AddEdge(srcnode, node);
for(SNode dst: dstnodes)
        AddEdge(node, dst);
    return node;
}
std::pair<SNode, SNode> Graph::InsertBridgeNode(SNode srcnode, SNode dstnode)
{
    LayerInfo info=srcnode->val();
    info.origin="kBridgeSrc";
    SNode src=AddNode("s-"+srcnode->name()+"-"+dstnode->name(), info);
    info=dstnode->val();
    info.origin="kBridgeDst";
    SNode dst=AddNode("d-"+srcnode->name()+"-"+dstnode->name(), info);
    AddEdge(srcnode, src);
    AddEdge(src, dst);
    AddEdge(dst, dstnode);
    return pair<SNode, SNode> {src, dst};
}

// for Recurrent Neural Network Implementation
// Function1 - detect cycle in the graph and save the graph (now can only deal with the single-cycle situation)
void Graph::DetectCycleAndSaveCycle()//save the cycle in the graph
{
    /*onStack = new bool[nodes_.size()];//?need to be deleted later
    edgeTo = new SNode[nodes_.size()];
    marked = new bool[nodes_.size()];*/

    //Initialize marked_ and on_stack_ to be all "false", and edge_to_ to be all -1
    for(int i = 0; i < nodes_.size(); i++)
    {
        marked_.push_back(false);
        on_stack_.push_back(false);
        edge_to_.push_back(-1);
    }

    for(int v = 0; v < nodes_.size(); v++)
    {
        if(marked_[v] != true)//node v is not visited
        {
            DFSForDetectCycleAndSaveCycle(nodes_[v]);
        }
    }
}

// Function2 - start from node v to detect whether there is a cycle
void Graph::DFSForDetectCycleAndSaveCycle(SNode node_v)
{
    on_stack_[node_v.id] = true;
    marked_[node_v.id] = true;
    for(int i = 0; i < node_v.dstnodes_.size(); i++)
    {
        if(hasCycle()==true)
        {
            return;//?correct?
        }
        if(marked_[node_v.dstnodes_[i].id] != true)//this node is not visited
        {
            edge_to_[node_v.dstnodes_[i].id] = node_v.id;
            DFSForDetectCycleAndSaveCycle(node_v.dstnodes_[i]);
        }
        else if(on_stack_[node_v.dstnodes_[i].id] == true)
        {
            //std::stack<SNode> cycle;
            for(int j = node_v.id; j != node_v.dstnodes_[i].id; j = edgeTo[j])
            {
                cycle_.push(nodes_[j]);//push the node corresponding to the id "j" into the stack "cycle"
            }
            cycle_.push(node_v.dstnodes_[i]);
            cycle_.push(node_v);
        }
    }
    on_stack_[node_v.id] = false;

}

// for Function2
bool Graph::hasCycle() const
{
    return cycle_ != NULL;
}

void ChangeCycleToEdges()//save the cycle information into edges (pairs of nodes)
{
    while(cycle_.size() > 1)
    {
        std::pair<SNode, SNode> one_edge;
        one_edge.first = cycle_.top();
        cycle_.pop();
        one_edge.second = cycle_.top();
        cycle_edges_.push_back(one_edge);// the "first" field is the src node and the "second" field is the dest node
    }
}

// Function4 - break one edge in the cycle and update the edge information
void BreakEdge(std::pair<SNode, SNode> break_edge)//update the edge information in the whole graph according to the edge
{
    RemoveEdge(break_edge.first, break_edge.second);
}


// Function5 - recover the breaking of one edge in the cycle and update the edge information
void RecoverEdge(std::pair<SNode, SNode> recover_edge)// update the edge information in the whole graph according to the edge
{
    AddEdge(recover_edge.first, recover_edge.second);
}

// Function6 - check whether the breaking is correct
bool CheckCorrectBreaking(std::pair<SNode, SNode> break_edge)
{
    BreakEdge(break_edge);
    int cnt_indegree_zero = 0;// denote the number of nodes with in-degree value as 0
    int cnt_outdegree_zero = 0;// denote the number of nodes with out-degree value as 0
    for(int i = 0; i < nodes_.size(); i++)
    {
        if(nodes_[i].srcnodes_.size() == 0)
        {
            cnt_indegree_zero++;
        }
        else if(nodes_[i].dstnodes_.size() == 0)
        {
            cnt_outdegree_zero++;
        }
    }
    if((cnt_indegree_zero > 1) || (cnt_outdegree_zero > 1)) return false;
    else return true;
}
