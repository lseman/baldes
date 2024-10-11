#pragma once
#include "../third_party/hgs_vrptw/Individual.h"
#include "../third_party/hgs_vrptw/Params.h"
#include <vector>
#include <xgboost/c_api.h> // XGBoost C++ API

struct TrainingData {
    std::vector<float> features;
    float              fitness;
};

// Buffer to store training data in real-time
std::vector<TrainingData> trainingBuffer;
const int                 bufferMaxSize = 1000; // Maximum size of the buffer before retraining the model

class SmartHGS {
public:
    BoosterHandle booster;

    SmartHGS() { initializeXGBoostModel(); }

    // Collect training data for XGBoost
    inline void collectTrainingData(Individual *offspring, int nbIter, int nbIterNonProd, Params *params) {
        std::vector<float> features = {
            static_cast<float>(nbIter),                             // Current iteration number
            static_cast<float>(nbIterNonProd),                      // Number of non-improving iterations
            static_cast<float>(offspring->isFeasible),              // Feasibility status
            static_cast<float>(offspring->myCostSol.penalizedCost), // Fitness score
            static_cast<float>(params->penaltyCapacity),            // Penalty for capacity
            static_cast<float>(params->penaltyTimeWarp)             // Penalty for time warp
        };

        // Add to the rolling buffer
        trainingBuffer.push_back({features, static_cast<float>(offspring->myCostSol.penalizedCost)});

        // If the buffer exceeds the limit, remove older data
        if (trainingBuffer.size() > bufferMaxSize) { trainingBuffer.erase(trainingBuffer.begin()); }
    }

    // Initialize the XGBoost model
    void initializeXGBoostModel() { XGBoosterCreate(nullptr, 0, &booster); }

    // Retrain the XGBoost model using collected data
    void retrainXGBoostModel() {
        // Prepare training data
        std::vector<float> train_data;
        std::vector<float> labels;

        for (const auto &entry : trainingBuffer) {
            train_data.insert(train_data.end(), entry.features.begin(), entry.features.end());
            labels.push_back(entry.fitness);
        }

        // Create DMatrix for training
        DMatrixHandle dmatrix;
        XGDMatrixCreateFromMat(train_data.data(), trainingBuffer.size(), trainingBuffer[0].features.size(), NAN,
                               &dmatrix);

        XGDMatrixSetFloatInfo(dmatrix, "label", labels.data(), labels.size());

        // Set parameters and retrain model
        XGBoosterSetParam(booster, "objective", "reg:squarederror");
        XGBoosterSetParam(booster, "eta", "0.1");
        XGBoosterSetParam(booster, "max_depth", "6");

        XGBoosterUpdateOneIter(booster, 0, dmatrix); // Perform one boosting round

        // Clean up
        XGDMatrixFree(dmatrix);
    }

    // Run the genetic algorithm, collecting data and using XGBoost
    void runGeneticAlgorithm(Params *params, int maxIterNonProd, int timeLimit) {
        int nbIter        = 0;
        int nbIterNonProd = 0;

        // Run the genetic algorithm
        while (nbIterNonProd <= maxIterNonProd && !params->isTimeLimitExceeded()) {
            // Example selection, crossover, and local search process (replace with your real logic)
            Individual *offspring = new Individual(); // Create a new individual (to be implemented)
            collectTrainingData(offspring, nbIter, nbIterNonProd, params);

            // Retrain the XGBoost model periodically
            if (nbIter % 1000 == 0) {
                retrainXGBoostModel(); // Retrain the model every 1000 iterations
            }

            // Continue with the rest of the genetic algorithm logic...
            nbIter++;
        }
    }

    // Predict the fitness of an individual using the XGBoost model
    double predictFitness(const std::vector<double> &features) {
        DMatrixHandle dmatrix;
        // convert features to float
        std::vector<float> features_float(features.begin(), features.end());
        XGDMatrixCreateFromMat(features_float.data(), 1, features_float.size(), NAN, &dmatrix);

        bst_ulong    out_len;
        const float *out_result;

        // Updated call to XGBoosterPredict with all necessary arguments
        XGBoosterPredict(booster, dmatrix, 0, 0, 0, &out_len, &out_result);

        XGDMatrixFree(dmatrix);
        return static_cast<double>(out_result[0]); // Return predicted fitness
    }

    ~SmartHGS() {
        XGBoosterFree(booster); // Free the booster resources
    }
};
