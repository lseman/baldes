#pragma once
#include <vector>

class SSPNode {
public:
    SSPNode() = default;
    //    int get_id() const;

    //    int node_id = 0;
    //    double shortest_path;
    std::vector<int>    precursor_idx;
    std::vector<int>    precursor_edges_idx;
    std::vector<double> precursor_edges_weights;

    std::vector<int>    successor_idx;
    std::vector<int>    successor_edges_idx;
    std::vector<double> successor_edges_weights;

    double price = 0;

    //    bool visited = false;
    // bool in_tree = false;
    //    SSPNode *parent_node = nullptr; //parent node in shortest path tree

    void add_precursor(int pre_id, int pre_edge_id, double weight) {
        this->precursor_idx.push_back(pre_id);
        this->precursor_edges_idx.push_back(pre_edge_id);
        this->precursor_edges_weights.push_back(weight);
    }
    void add_successor(int succ_id, int succ_edge_id, double weight) {
        this->successor_idx.push_back(succ_id);
        this->successor_edges_idx.push_back(succ_edge_id);
        this->successor_edges_weights.push_back(weight);
    }
};
