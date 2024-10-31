/**
 * @file Reader.h
 * @brief Header file for reading and processing problem instance data, specifically for Vehicle Routing Problem with
 * Time Windows (VRPTW).
 *
 * This file defines the `InstanceData` structure, which holds various data related to a problem instance, such as
 * distances, travel costs, demands, time windows, and other relevant information. It also includes several inline
 * functions for manipulating and processing instance data, such as reducing time windows, deleting arcs, and reading an
 * instance from a file.
 *
 *
 * This file is essential for handling the input and initialization of problem instances, and it ensures that the data
 * is formatted and processed correctly for further use in solving the VRPTW.
 *
 */

#pragma once

#include "Definitions.h"
#include "config.h"
#include <cmath>
#include <cstring>
#include <fmt/core.h>
#include <fstream>
#include <iostream>
#include <sstream> // This is where std::istringstream is defined
#include <vector>

/**
 * @struct InstanceData
 * @brief A structure to hold various data related to an instance of a problem.
 *
 * This structure contains multiple vectors and variables to store distances,
 * travel costs, demands, time windows, service times, and other relevant data
 * for solving a problem instance.
 *
 */
struct InstanceData {
    std::vector<std::vector<double>>                    distance;
    std::vector<std::vector<double>>                    travel_cost;
    std::vector<int>                                    demand;
    std::vector<double>                                 window_open;
    std::vector<double>                                 window_close;
    std::vector<std::vector<std::pair<double, double>>> time_windows; // New field for multiple time windows
    std::vector<int>                                    n_tw;
    std::vector<double>                                 service_time;
    std::vector<double>                                 demand_additional;
    int                                                 q          = 0;
    int                                                 nC         = 0;
    int                                                 nV         = 0;
    int                                                 nN         = 0;
    int                                                 nR         = 0;
    double                                              T_max      = 0.0;
    double                                              T_avg      = 0.0;
    double                                              demand_sum = 0.0;
    int                                                 nV_min     = 0;
    int                                                 nV_max     = 0;
    int                                                 nD_min     = 0;
    int                                                 nD_max     = 0;
    std::vector<int>                                    deleted_arcs;
    int                                                 deleted_arcs_n     = 0;
    int                                                 deleted_arcs_n_max = 0;

    std::vector<double> x_coord;
    std::vector<double> y_coord;
    ProblemType         problem_type;

    int getNbVertices() const { return nN - 1; }

    /**
     * @brief Retrieves the distance value between two points.
     *
     * This function returns the distance value between the points
     * specified by the indices `i` and `j`.
     *
     * @param i The index of the first point.
     * @param j The index of the second point.
     * @return The distance value between the points `i` and `j`.
     */
    double getcij(int i, int j) const { return distance[i][j]; }

    // define getDistanceMatrix
    /**
     * @brief Retrieves the distance matrix.
     *
     * This function returns a 2D vector containing the distance matrix.
     * Each element in the outer vector represents a row in the matrix,
     * and each element in the inner vector represents a column in the matrix.
     *
     * @return std::vector<std::vector<double>> The distance matrix.
     */
    std::vector<std::vector<double>> getDistanceMatrix() const { return distance; }

    /**
     * @brief Retrieves the demand value for a given index.
     *
     * @param i The index for which the demand value is requested.
     * @return double The demand value at the specified index.
     */
    double getDemand(int i) const { return demand[i]; }

    /**
     * @brief Retrieves the capacity.
     *
     * This function returns the value of the capacity, which is stored in the member variable `q`.
     *
     * @return int The capacity value.
     */
    int getCapacity() const { return q; }
};

/**
 * @brief Reduces the time windows for the vertices in the VRPTW instance.
 *
 * This function iteratively reduces the time windows for the vertices in the VRPTW (Vehicle Routing Problem with Time
 * Windows) instance. The reduction is performed based on the distances, service times, and existing time windows of the
 * vertices. The function stops iterating if no further reduction is possible or after a maximum number of iterations.
 *
 */
inline int VRPTW_reduce_time_windows(InstanceData &instance) {
    int    it      = 0;
    bool   changed = true;
    double aux_1;

    while (changed && it < 20) {
        // fmt::print("Iteration {}\n", it);
        changed = false;

        for (int k = 1; k < instance.nN - 1; ++k) {
            aux_1 = instance.window_close[k];
            for (int i = 0; i < instance.nN; ++i) {
                if (i != k && aux_1 > instance.window_open[i] + instance.distance[i][k] + instance.service_time[i]) {
                    aux_1 = instance.window_open[i] + instance.distance[i][k] + instance.service_time[i];
                }
            }
            if (aux_1 > instance.window_open[k]) {
                // fmt::print("Reducing time window for vertex {}\n", k);
                instance.window_open[k] = aux_1;
                changed                 = true;
            }
        }

        for (int k = 1; k < instance.nN - 1; ++k) {
            aux_1 = instance.window_close[k];
            for (int i = 0; i < instance.nN; ++i) {
                if (i != k && aux_1 > instance.window_open[i] - instance.distance[k][i] - instance.service_time[i]) {
                    aux_1 = instance.window_open[i] - instance.distance[k][i] - instance.service_time[i];
                }
            }
            if (aux_1 > instance.window_open[k]) {
                // fmt::print("Reducing time window for vertex {}\n", k);
                instance.window_open[k] = aux_1;
                changed                 = true;
            }
        }

        for (int k = 1; k < instance.nN - 1; ++k) {
            aux_1 = instance.window_open[k];
            for (int i = 0; i < instance.nN; ++i) {
                if (i != k && aux_1 < instance.window_close[i] + instance.distance[i][k] + instance.service_time[i]) {
                    aux_1 = instance.window_close[i] + instance.distance[i][k] + instance.service_time[i];
                }
            }
            if (aux_1 < instance.window_close[k]) {
                // fmt::print("Reducing time window for vertex {}\n", k);
                instance.window_close[k] = aux_1;
                changed                  = true;
            }
        }

        for (int k = 1; k < instance.nN - 1; ++k) {
            aux_1 = instance.window_open[k];
            for (int i = 0; i < instance.nN; ++i) {
                if (i != k && aux_1 < instance.window_close[i] - instance.distance[k][i] - instance.service_time[i]) {
                    aux_1 = instance.window_close[i] - instance.distance[k][i] - instance.service_time[i];
                }
            }
            if (aux_1 < instance.window_close[k]) {
                // fmt::print("Reducing time window for vertex {}\n", k);
                instance.window_close[k] = aux_1;
                changed                  = true;
            }
        }

        ++it;
    }
    return 1;
}

/**
 * Calculates the greatest common divisor (GCD) of two integers using the Euclidean algorithm.
 *
 */
inline int VRPTW_mcd(int m, int n) {
    int aux;

    if (m < n) {
        aux = n;
        n   = m;
        m   = aux;
    }

    if (n <= 1) return 1;

    while (n != 0) {
        aux = m;
        m   = n;
        n   = aux % n;
    }

    return m;
}

/**
 * Reads an instance file and populates the provided `InstanceData` object with the data.
 *
 */
inline int VRPTW_read_instance(const std::string &file_name, InstanceData &instance, bool mtw = false) {
    std::ifstream myfile(file_name);
    if (!myfile.is_open()) { return 0; }

    std::string current_line;
    if (!mtw)
        for (int i = 0; i < 4; ++i) { std::getline(myfile, current_line); }

    if (!mtw) myfile >> instance.nV >> instance.q;
    instance.nN = N_SIZE;

    std::vector<double> xcoord(instance.nN);
    std::vector<double> ycoord(instance.nN);
    instance.demand.resize(instance.nN);
    instance.demand_additional.resize(instance.nN);
    instance.window_open.resize(instance.nN);
    instance.window_close.resize(instance.nN);
    instance.service_time.resize(instance.nN);
    instance.n_tw.resize(instance.nN);

    // Initialize time windows vector only if mtw is enabled
    if (mtw) { instance.time_windows.resize(instance.nN); }

    if (!mtw)
        for (int i = 0; i < 4; ++i) { std::getline(myfile, current_line); }

    if (mtw)
        for (int i = 0; i < 1; ++i) { std::getline(myfile, current_line); }

    int i = 0;

    if (!mtw) {
        while (myfile >> i >> xcoord[i] >> ycoord[i] >> instance.demand[i] >> instance.window_open[i] >>
               instance.window_close[i] >> instance.service_time[i]) {
            instance.n_tw[i] = 0;
            // Check if we need to resize
            if (i >= instance.nN) {
                instance.nN *= 2;
                xcoord.resize(instance.nN);
                ycoord.resize(instance.nN);
                instance.demand.resize(instance.nN);
                instance.window_open.resize(instance.nN);
                instance.window_close.resize(instance.nN);
                instance.service_time.resize(instance.nN);
                instance.n_tw.resize(instance.nN);
            }
        }
    } else {
        instance.nV = 25;
        instance.q  = 100;
        std::string tws;
        fmt::print("{}\n", instance.nN);

        // Ensure the file is open
        if (!myfile.is_open()) {
            fmt::print("Error: File could not be opened.\n");
            std::throw_with_nested(std::runtime_error("Error: File could not be opened."));
        }

        // Read the lines manually first to ensure the format is correct
        std::string line;
        while (std::getline(myfile, line)) {
            std::istringstream iss(line);

            // Skip the extra whitespace manually for each field
            if (!(iss >> i)) {
                fmt::print("Error parsing CUST NO.\n");
                continue;
            }

            // Extract the remaining fields manually, skipping whitespace
            double x, y;
            int    demand, n_tw;

            iss >> x >> y >> demand >> n_tw >> tws; // Read coordinates, demand, and number of time windows
            if (iss.fail()) {
                fmt::print("Error parsing fields.\n");
                continue;
            }

            fmt::print("i: {}, x: {}, y: {}, demand: {}, n_tw: {}\n", i, x, y, demand, n_tw);

            xcoord[i]          = x;
            ycoord[i]          = y;
            instance.demand[i] = demand;
            instance.n_tw[i]   = n_tw;
            if (i != 0)
                instance.service_time[i] = 1; // Assuming service time is 10 for all nodes
            else
                instance.service_time[i] = 0;
            // Parsing multiple time windows (e.g., "45,105;113,174;188,250")
            std::replace(tws.begin(), tws.end(), ';', ' '); // Replace semicolons with spaces
            std::replace(tws.begin(), tws.end(), ',', ' '); // Replace commas with spaces

            std::istringstream tw_stream(tws);
            int                start, end;

            // Parse each start, end pair from the stream
            while (tw_stream >> start >> end) {
                fmt::print("start: {}, end: {}\n", start, end);
                instance.time_windows[i].emplace_back(start, end);
            }

            instance.window_open[i]  = instance.time_windows[i][0].first;
            instance.window_close[i] = instance.time_windows[i][0].second;

            fmt::print("window_open: {}, window_close: {}\n", instance.window_open[i], instance.window_close[i]);
        }
    }

    // Example output to verify the data
    instance.nN                            = i + 2;
    instance.nC                            = i;
    xcoord[instance.nN - 1]                = xcoord[0];
    ycoord[instance.nN - 1]                = ycoord[0];
    instance.demand[instance.nN - 1]       = instance.demand[0];
    instance.window_open[instance.nN - 1]  = instance.window_open[0];
    instance.window_close[instance.nN - 1] = instance.window_close[0];
    instance.service_time[instance.nN - 1] = instance.service_time[0];
    if (mtw) { instance.time_windows[instance.nN - 1] = instance.time_windows[0]; }

    instance.x_coord.resize(instance.nN);
    instance.y_coord.resize(instance.nN);
    instance.demand.resize(instance.nN);
    instance.window_open.resize(instance.nN);
    instance.window_close.resize(instance.nN);
    instance.service_time.resize(instance.nN);
    if (mtw) { instance.time_windows.resize(instance.nN); }

    instance.distance.resize(instance.nN, std::vector<double>(instance.nN));

    for (int i = 0; i < instance.nN; ++i) {
        instance.x_coord[i] = xcoord[i];
        instance.y_coord[i] = ycoord[i];
        for (int j = 0; j < instance.nN; ++j) {
            if (!mtw) {
                int  x                  = xcoord[i] - xcoord[j];
                int  y                  = ycoord[i] - ycoord[j];
                auto aux                = (int)(10 * sqrt(x * x + y * y));
                instance.distance[i][j] = 1.0 * aux;

            } else {
                double x                = xcoord[i] - xcoord[j];
                double y                = ycoord[i] - ycoord[j];
                auto   aux              = (double)(10 * sqrt(x * x + y * y));
                instance.distance[i][j] = 1.0 * aux;
            }
        }
        instance.window_open[i] *= 10;
        instance.window_close[i] *= 10;
        instance.service_time[i] *= 10;
    }

    // Presolve for the time windows
    VRPTW_reduce_time_windows(instance);

    instance.travel_cost = instance.distance;

    instance.T_max      = 0.0;
    instance.T_avg      = 0.0;
    instance.demand_sum = 0.0;

    for (int i = 1; i < instance.nN - 1; ++i) {
        instance.demand_sum += instance.demand[i];
        instance.T_avg += instance.window_close[i] - instance.window_open[i];
        if (instance.T_max <
            instance.window_close[i] + instance.service_time[i] + instance.distance[i][instance.nN - 1]) {
            instance.T_max =
                instance.window_close[i] + instance.service_time[i] + instance.distance[i][instance.nN - 1];
        }
    }
    instance.T_max = instance.window_close[0];

    if (mtw) { instance.T_max = instance.time_windows[0][0].second; }

    instance.T_avg /= static_cast<double>(instance.nC);

#ifdef MCD
    auto aux_int = VRPTW_mcd(instance.q, instance.demand[1]);
    for (i = 2; (i < instance.nN - 1) && (aux_int > 1); i++) aux_int = VRPTW_mcd(aux_int, instance.demand[i]);

    if (aux_int > 1) {
        instance.q /= aux_int;
        for (i = 1; i < instance.nN - 1; i++) instance.demand[i] /= aux_int;
    }
#endif

    instance.problem_type = ProblemType::vrptw;
    return 1;
}

inline int CVRP_read_instance(const std::string &file_name, InstanceData &instance) {
    std::ifstream myfile(file_name);
    if (!myfile.is_open()) {
        std::cerr << "Failed to open file: " << file_name << std::endl;
        return 0;
    }

    std::string line;
    int         dimension          = 0;
    bool        node_coord_section = false, demand_section = false, depot_section = false;

    while (std::getline(myfile, line)) {
        std::istringstream iss(line);
        std::string        key;
        iss >> key;

        if (key == "NAME" || key == "COMMENT" || key == "TYPE" || key == "EDGE_WEIGHT_TYPE") {
            // Ignore these fields
            continue;
        } else if (key == "DIMENSION") {
            std::string colon;
            iss >> colon >> dimension;
            instance.nN = dimension + 1;
            instance.x_coord.resize(instance.nN + 1); // 1-based indexing, so +1
            instance.y_coord.resize(instance.nN + 1);
            instance.demand.resize(instance.nN + 1);
            instance.service_time.resize(instance.nN + 1);
            instance.window_open.resize(instance.nN + 1);
            instance.window_close.resize(instance.nN + 1);
            instance.n_tw.resize(instance.nN + 1);
            instance.distance.resize(instance.nN + 1, std::vector<double>(instance.nN + 1));
        } else if (key == "CAPACITY") {
            std::string colon;
            iss >> colon >> instance.q; // Vehicle capacity
        } else if (key == "NODE_COORD_SECTION") {
            node_coord_section = true;
            continue;
        } else if (key == "DEMAND_SECTION") {
            node_coord_section = false;
            demand_section     = true;
            continue;
        } else if (key == "DEPOT_SECTION") {
            demand_section = false;
            depot_section  = true;
            continue;
        } else if (key == "EOF") {
            break; // End of file
        }

        if (node_coord_section) {
            int    node;
            double x, y;
            // get the node, x and y given the format 1    764  255
            std::istringstream coord_line(line);   // Reinitialize stream for each line
            if (!(coord_line >> node >> x >> y)) { // Attempt to parse node and coordinates
                std::cerr << "Error reading NODE_COORD_SECTION at line: " << line << std::endl;
                continue;
            }
            node -= 1; // Adjust to 0-based indexing
            if (node >= 0 && node < dimension) {
                instance.x_coord[node] = x;
                instance.y_coord[node] = y;
            } else {
                std::cerr << "Node index out of range: " << node << std::endl;
            }
        } else if (demand_section) {
            int                node, demand;
            std::istringstream demand_line(line); // Reinitialize stream for each line
            if (!(demand_line >> node >> demand)) {
                std::cerr << "Error reading DEMAND_SECTION at line: " << line << std::endl;
                continue;
            }
            node -= 1; // Adjust to 0-based indexing

            if (node >= 0 && node < dimension) {
                instance.demand[node] = demand;
            } else {
                std::cerr << "Node index out of range: " << node << std::endl;
            }
        } else if (depot_section) {
            int depot;
            iss >> depot;
            if (depot == -1) break; // End of DEPOT_SECTION
        }
    }

    // Calculate Euclidean distances between nodes
    int max_travel_time = 0;

    // Calculate maximum travel time for setting window close time
    for (int i = 0; i < instance.nN; ++i) {
        for (int j = 0; j < instance.nN; ++j) {
            int dx                  = instance.x_coord[i] - instance.x_coord[j];
            int dy                  = instance.y_coord[i] - instance.y_coord[j];
            int aux                 = static_cast<int>(10 * std::sqrt(dx * dx + dy * dy));
            instance.distance[i][j] = aux / 10.0;

            if (aux > max_travel_time) { max_travel_time = aux; }
        }
    }

    // Assign the travel cost
    instance.travel_cost = instance.distance;

    // Set service time and window times
    for (int i = 0; i < instance.nN; ++i) {
        instance.service_time[i] = 0;
        instance.window_open[i]  = 0;
        instance.window_close[i] = max_travel_time + max_travel_time/2.0;
    }

    // define n_tw as 0
    for (int i = 0; i < instance.nN; i++) { instance.n_tw[i] = 0; }

    instance.x_coord[instance.nN - 1]      = instance.x_coord[0];
    instance.y_coord[instance.nN - 1]      = instance.y_coord[0];
    instance.demand[instance.nN - 1]       = instance.demand[0];
    instance.service_time[instance.nN - 1] = instance.service_time[0];
    instance.window_open[instance.nN - 1]  = instance.window_open[0];
    instance.window_close[instance.nN - 1] = instance.window_close[0];
    instance.n_tw[instance.nN - 1]         = instance.n_tw[0];

    instance.nV = 30;

    for (int i = 1; i < instance.nN - 1; ++i) { instance.demand_sum += instance.demand[i]; }
    instance.T_max        = instance.window_close[0];
    instance.problem_type = ProblemType::cvrp;

    return 1; // Success
}