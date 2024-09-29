/**
 * @file Label.h
 * @brief This file contains the definition of the Label struct and LabelComparator class.
 */

#pragma once
#include "Common.h"

/**
 * @struct Label
 * @brief Represents a label used in a solver.
 *
 * This struct contains various properties and methods related to a label used in a solver.
 * It stores information such as the set of Fj, id, is_extended flag, vertex, cost, real_cost, SRC_cost,
 * resources, predecessor, is_dominated flag, jobs_covered, jobs_ordered, job_id, cut_storage,
 * parent, children, status, visited, and SRCmap.
 *
 * The struct provides constructors to initialize the label with or without a job_id.
 * It also provides methods to set the covered jobs, add a job to the covered jobs, check if a job is
 * already covered, and initialize the label with new values.
 *
 * The struct overloads the equality and greater than operators for comparison.
 */
struct Label {
    // int                   id;
    bool                       is_extended = false;
    int                        vertex;
    double                     cost         = 0.0;
    double                     real_cost    = 0.0;
    std::array<double, R_SIZE> resources    = {};
    std::vector<int>           jobs_covered = {}; // Add jobs_covered to Label
    int                        job_id       = -1; // Add job_id to Label
    Label                     *parent       = nullptr;
#ifdef SRC3
    std::array<std::uint16_t, MAX_SRC_CUTS> SRCmap = {};
#endif
#ifdef SRC
    std::vector<double> SRCmap;
#endif
    // uint64_t             visited_bitmap; // Bitmap for visited jobs
    std::array<uint64_t, num_words> visited_bitmap = {0};
#ifdef UNREACHABLE_DOMINANCE
    std::array<uint64_t, num_words> unreachable_bitmap = {0};
#endif

    // Constructor with job_id
    Label(int v, double c, const std::vector<double> &res, int pred, int job_id)
        : vertex(v), cost(c), resources({res[0]}), job_id(job_id) {}

    // Constructor without job_id
    Label(int v, double c, const std::vector<double> &res, int pred)
        : vertex(v), cost(c), resources({res[0]}), job_id(-1) {}

    // Default constructor
    Label() : vertex(-1), cost(0), resources({0.0}), job_id(-1) {}

    void set_extended(bool extended) { is_extended = extended; }

    /**
     * @brief Checks if a job has been visited.
     *
     * This function determines whether a job, identified by its job_id, has been visited.
     * It uses a bitmask (visited_bitmap) where each bit represents the visit status of a job.
     *
     */
    bool visits(int job_id) const { return visited_bitmap[job_id / 64] & (1ULL << (job_id % 64)); }

    /**
     * @brief Resets the state of the object to its initial values.
     *
     */
    inline void reset() {
        this->vertex    = -1;
        this->cost      = 0.0;
        this->resources = {};
        // this->job_id      = -1;
        this->real_cost   = 0.0;
        this->parent      = nullptr;
        this->is_extended = false;
        // this->jobs_covered.clear();

        std::memset(visited_bitmap.data(), 0, visited_bitmap.size() * sizeof(uint64_t));
#ifdef UNREACHABLE_DOMINANCE
        std::memset(unreachable_bitmap.data(), 0, unreachable_bitmap.size() * sizeof(uint64_t));
#endif
#ifdef SRC3
        std::memset(SRCmap.data(), 0, SRCmap.size() * sizeof(std::uint16_t));
#endif
#ifdef SRC
        SRCmap.clear();
#endif
    }

    void addJob(int job) { jobs_covered.push_back(job); }

    /**
     * @brief Initializes the object with the given parameters.
     *
     */
    inline void initialize(int vertex, double cost, const std::vector<double> &resources, int job_id) {
        this->vertex = vertex;
        this->cost   = cost;

        // Assuming `resources` is a vector or array-like structure with the same size as the input
        std::copy(resources.begin(), resources.end(), this->resources.begin());

        this->job_id = job_id;
    }

    bool operator>(const Label &other) const { return cost > other.cost; }

    bool operator<(const Label &other) const { return cost < other.cost; }
};

/**
 * @class LabelComparator
 * @brief Comparator class for comparing two Label objects based on their cost.
 *
 * This class provides an overloaded operator() that allows for comparison
 * between two Label pointers. The comparison is based on the cost attribute
 * of the Label objects, with the comparison being in descending order.
 */
class LabelComparator {
public:
    bool operator()(Label *a, Label *b) { return a->cost > b->cost; }
};
