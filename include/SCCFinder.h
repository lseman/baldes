/**
 * @file SCCFinder.h
 * @brief Header file for the SCC class, implementing Tarjan's algorithm to find Strongly Connected Components (SCCs).
 *
 * This file defines the SCC class, which is used to find Strongly Connected Components (SCCs) in a directed graph.
 * The graph is represented using an adjacency list, and the class provides several methods to add edges, convert from
 * other graph representations, find SCCs using Tarjan's algorithm, determine the topological order of SCCs, and export
 * the graph and its SCCs to a DOT file for visualization.
 *
 * The main components of this file include:
 * - The SCC class: Implements Tarjan's algorithm to detect SCCs in a directed graph.
 * - Methods for adding directed edges and converting from an unordered map representation.
 * - Methods for finding SCCs and determining their topological order.
 * - A method for exporting the graph and SCCs to a DOT file with visual distinctions for SCCs.
 *
 * The SCC class uses an internal stack, low-link values, and vertex indices to efficiently identify SCCs in a graph.
 * It also supports exporting the results in a format suitable for visualization using Graphviz.
 */

#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/**
 * @class SCC
 * @brief A class to find Strongly Connected Components (SCCs) in a directed graph using Tarjan's algorithm.
 *
 * This class provides methods to add edges to the graph, convert from an unordered map representation,
 * find SCCs using Tarjan's algorithm, determine the topological order of SCCs, and export the graph to a DOT file.
 *
 * @note The graph is represented using an adjacency list.
 */
class SCC {
public:
    SCC() : currentIndex(0) {}

    /**
     * @brief Adds a directed edge from vertex v to vertex w in the graph.
     *
     * This function ensures that the adjacency list and other related structures
     * are resized appropriately if the vertices v or w exceed the current size.
     * It then adds w to the adjacency list of v.
     *
     * @param v The starting vertex of the directed edge.
     * @param w The ending vertex of the directed edge.
     */
    void addEdge(int v, int w) {
        if (v >= adj.size() || w >= adj.size()) {
            int newSize = std::max(v, w) + 1;
            adj.resize(newSize);
            index.resize(newSize, -1);
            lowlink.resize(newSize, -1);
            onStack.resize(newSize, false);
        }
        adj[v].push_back(w);
    }

    /**
     * @brief Converts an unordered map representation of a graph into an internal graph representation.
     *
     * This function takes an unordered map where the keys are vertex identifiers and the values are 
     * vectors of adjacent vertices. It iterates through the map and adds edges to the internal graph 
     * representation using the addEdge function.
     *
     * @param map An unordered map where each key is a vertex and the corresponding value is a vector 
     *            of vertices that are adjacent to the key vertex.
     */
    void convertFromUnorderedMap(const std::unordered_map<int, std::vector<int>> &map) {
        for (const auto &pair : map) {
            int v = pair.first;
            for (int w : pair.second) { addEdge(v, w); }
        }
    }

    /**
     * @brief Finds and returns all Strongly Connected Components (SCCs) in the graph using Tarjan's algorithm.
     *
     * This function implements Tarjan's algorithm to find all SCCs in a directed graph. It iterates over all vertices
     * and applies the strongConnect function to each vertex that has not been visited yet (indicated by an index of -1).
     * The result is a vector of SCCs, where each SCC is represented as a vector of vertex indices.
     *
     * @return A vector of vectors, where each inner vector represents a strongly connected component in the graph.
     */
    std::vector<std::vector<int>> tarjanSCC() {
        std::vector<std::vector<int>> sccs;

        for (int v = 0; v < adj.size(); ++v) {
            if (index[v] == -1) { strongConnect(v, sccs); }
        }

        return sccs;
    }

    /**
     * @brief Computes the topological order of Strongly Connected Components (SCCs) in a directed graph.
     *
     * This function takes a list of SCCs and computes a topological ordering of these components.
     * The input is a vector of vectors, where each inner vector represents a SCC containing node indices.
     *
     * @param sccs A vector of vectors, where each inner vector represents a SCC.
     * @return A vector of integers representing the topological order of the SCCs.
     *
     * The function performs the following steps:
     * 1. Maps each node to its corresponding SCC.
     * 2. Constructs a directed graph where each node represents a SCC and edges represent dependencies between SCCs.
     * 3. Computes the in-degree of each SCC in the component graph.
     * 4. Uses Kahn's algorithm to find a topological order of the SCCs.
     */
    std::vector<int> topologicalOrderOfSCCs(const std::vector<std::vector<int>> &sccs) {
        std::unordered_map<int, int> nodeToScc;
        for (int i = 0; i < sccs.size(); ++i) {
            for (int node : sccs[i]) { nodeToScc[node] = i; }
        }

        std::vector<std::unordered_set<int>> componentGraph(sccs.size());
        for (int v = 0; v < adj.size(); ++v) {
            for (int w : adj[v]) {
                int sccV = nodeToScc[v];
                int sccW = nodeToScc[w];
                if (sccV != sccW) { componentGraph[sccV].insert(sccW); }
            }
        }

        std::vector<int> inDegree(sccs.size(), 0);
        for (const auto &neighbors : componentGraph) {
            for (int neighbor : neighbors) { ++inDegree[neighbor]; }
        }

        std::stack<int> zeroInDegreeNodes;
        for (int i = 0; i < inDegree.size(); ++i) {
            if (inDegree[i] == 0) { zeroInDegreeNodes.push(i); }
        }

        std::vector<int> topologicalOrder;
        while (!zeroInDegreeNodes.empty()) {
            int node = zeroInDegreeNodes.top();
            zeroInDegreeNodes.pop();
            topologicalOrder.push_back(node);

            for (int neighbor : componentGraph[node]) {
                --inDegree[neighbor];
                if (inDegree[neighbor] == 0) { zeroInDegreeNodes.push(neighbor); }
            }
        }

        return topologicalOrder;
    }

    /**
     * @brief Exports the strongly connected components (SCCs) of a graph to a DOT file.
     *
     * This function generates a DOT file representing the graph with nodes colored
     * according to their SCC. Each SCC is assigned a different color from a predefined
     * set of colors.
     *
     * @param filename The name of the file to which the DOT representation will be written.
     * @param sccs A vector of vectors, where each inner vector represents a strongly connected component
     *             and contains the nodes belonging to that SCC.
     */
    void exportToDot(const std::string &filename, const std::vector<std::vector<int>> &sccs) {
        std::ofstream dotFile(filename);
        if (!dotFile.is_open()) {
            std::cerr << "Error: Could not open file for writing\n";
            return;
        }

        // Start DOT file format
        dotFile << "digraph G {\n";

        // Assign each SCC a different color
        std::vector<std::string> colors = {"red",    "green", "blue",    "yellow", "orange",
                                           "purple", "cyan",  "magenta", "lime",   "pink"};

        // Create a mapping from node to SCC index
        std::unordered_map<int, int> nodeToScc;
        for (int i = 0; i < sccs.size(); ++i) {
            for (int node : sccs[i]) { nodeToScc[node] = i; }
        }

        // Output nodes with their SCC colors
        for (int v = 0; v < adj.size(); ++v) {
            std::string color = colors[nodeToScc[v] % colors.size()];
            dotFile << "    " << v << " [style=filled, fillcolor=\"" << color << "\"];\n";
        }

        // Output edges
        for (int v = 0; v < adj.size(); ++v) {
            for (int w : adj[v]) { dotFile << "    " << v << " -> " << w << ";\n"; }
        }

        // End DOT file format
        dotFile << "}\n";
        dotFile.close();

        std::cout << "Graph exported to " << filename << "\n";
    }

private:
    std::vector<std::vector<int>> adj;
    std::vector<int>              index;
    std::vector<int>              lowlink;
    std::vector<bool>             onStack;
    std::stack<int>               stack;
    int                           currentIndex;

    /**
     * @brief Finds and records the strongly connected components (SCCs) in a directed graph using Tarjan's algorithm.
     *
     * This function is a recursive implementation of Tarjan's algorithm for finding SCCs in a directed graph.
     * It updates the indices and lowlink values of the vertices, and uses a stack to keep track of the vertices
     * in the current SCC. When an SCC is found, it is added to the list of SCCs.
     *
     * @param v The current vertex being processed.
     * @param sccs A reference to a vector of vectors, where each inner vector represents an SCC found in the graph.
     */
    void strongConnect(int v, std::vector<std::vector<int>> &sccs) {
        index[v]   = currentIndex;
        lowlink[v] = currentIndex;
        currentIndex++;
        stack.push(v);
        onStack[v] = true;

        for (int w : adj[v]) {
            if (index[w] == -1) {
                strongConnect(w, sccs);
                lowlink[v] = std::min(lowlink[v], lowlink[w]);
            } else if (onStack[w]) {
                lowlink[v] = std::min(lowlink[v], index[w]);
            }
        }

        if (lowlink[v] == index[v]) {
            std::vector<int> scc;
            int              w;
            do {
                w = stack.top();
                stack.pop();
                onStack[w] = false;
                scc.push_back(w);
            } while (w != v);
            sccs.push_back(scc);
        }
    }
};
