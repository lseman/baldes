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
 * The main components of this file include:
 * - The `InstanceData` structure: Contains all the relevant data for a VRPTW problem instance, such as distance
 * matrices, demands, time windows, service times, and other problem-specific information.
 * - `VRPTW_delete_arc`: An inline function for managing deleted arcs in the graph.
 * - `VRPTW_reduce_time_windows`: An inline function to iteratively reduce the time windows for vertices based on
 * constraints.
 * - `VRPTW_mcd`: An inline function that calculates the greatest common divisor (GCD) of two integers, used for
 * capacity and demand adjustments.
 * - `VRPTW_read_instance`: An inline function to read a VRPTW instance file and populate the `InstanceData` structure
 * with relevant data.
 *
 * This file is essential for handling the input and initialization of problem instances, and it ensures that the data
 * is formatted and processed correctly for further use in solving the VRPTW.
 */

#pragma once

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
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
    std::vector<std::vector<double>> distance;
    std::vector<std::vector<double>> travel_cost;
    std::vector<int>                 demand;
    std::vector<double>              window_open;
    std::vector<double>              window_close;
    std::vector<double>              service_time;
    std::vector<double>              demand_additional;
    std::vector<std::vector<double>> distance_additional;
    int                              q          = 0;
    int                              nC         = 0;
    int                              nV         = 0;
    int                              nN         = 0;
    int                              nR         = 0;
    double                           T_max      = 0.0;
    double                           T_avg      = 0.0;
    double                           demand_sum = 0.0;
    int                              nV_min     = 0;
    int                              nV_max     = 0;
    int                              nD_min     = 0;
    int                              nD_max     = 0;
    std::vector<int>                 deleted_arcs;
    int                              deleted_arcs_n     = 0;
    int                              deleted_arcs_n_max = 0;

    std::vector<double> x_coord;
    std::vector<double> y_coord;

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
inline int VRPTW_read_instance(const std::string &file_name, InstanceData &instance) {
    std::ifstream myfile(file_name);
    if (!myfile.is_open()) { return 0; }

    std::string current_line;
    for (int i = 0; i < 4; ++i) { std::getline(myfile, current_line); }

    myfile >> instance.nV >> instance.q;
    std::cout << "nV: " << instance.nV << " q: " << instance.q << std::endl;
    instance.nN = 102;

    std::vector<int> xcoord(instance.nN);
    std::vector<int> ycoord(instance.nN);
    instance.demand.resize(instance.nN);
    instance.demand_additional.resize(instance.nN);
    instance.window_open.resize(instance.nN);
    instance.window_close.resize(instance.nN);
    instance.service_time.resize(instance.nN);

    for (int i = 0; i < 4; ++i) { std::getline(myfile, current_line); }

    int i = 0;
    ;
    while (myfile >> i >> xcoord[i] >> ycoord[i] >> instance.demand[i] >> instance.window_open[i] >>
           instance.window_close[i] >> instance.service_time[i]) {
        //++i;
        if (i >= instance.nN) {
            instance.nN *= 2;
            xcoord.resize(instance.nN);
            ycoord.resize(instance.nN);
            instance.demand.resize(instance.nN);
            instance.window_open.resize(instance.nN);
            instance.window_close.resize(instance.nN);
            instance.service_time.resize(instance.nN);
        }
    }

    // Example output to verify the data
    instance.nN = i + 2;
    instance.nC = i;
    //instance.nV = instance.nC;

    xcoord[instance.nN - 1]                = xcoord[0];
    ycoord[instance.nN - 1]                = ycoord[0];
    instance.demand[instance.nN - 1]       = instance.demand[0];
    instance.window_open[instance.nN - 1]  = instance.window_open[0];
    instance.window_close[instance.nN - 1] = instance.window_close[0];
    instance.service_time[instance.nN - 1] = instance.service_time[0];

    xcoord.resize(instance.nN);
    ycoord.resize(instance.nN);
    instance.demand.resize(instance.nN);
    instance.window_open.resize(instance.nN);
    instance.window_close.resize(instance.nN);
    instance.service_time.resize(instance.nN);

    instance.x_coord.resize(instance.nN);
    instance.y_coord.resize(instance.nN);

    myfile.close();

    instance.distance.resize(instance.nN, std::vector<double>(instance.nN));

    for (int i = 0; i < instance.nN; ++i) {
        instance.x_coord[i] = xcoord[i];
        instance.y_coord[i] = ycoord[i];
        for (int j = 0; j < instance.nN; ++j) {
            int  x                  = xcoord[i] - xcoord[j];
            int  y                  = ycoord[i] - ycoord[j];
            auto aux                = (int)(10 * sqrt(x * x + y * y));
            instance.distance[i][j] = 1.0 * aux;
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
    std::cout << "T_max: " << instance.T_max << std::endl;
    instance.T_avg /= static_cast<double>(instance.nC);

#ifdef MCD
    auto aux_int = VRPTW_mcd(instance.q, instance.demand[1]);
    for (i = 2; (i < instance.nN - 1) && (aux_int > 1); i++) aux_int = VRPTW_mcd(aux_int, instance.demand[i]);

    if (aux_int > 1) {
        instance.q /= aux_int;
        for (i = 1; i < instance.nN - 1; i++) instance.demand[i] /= aux_int;
    }
#endif

    return 1;
}
