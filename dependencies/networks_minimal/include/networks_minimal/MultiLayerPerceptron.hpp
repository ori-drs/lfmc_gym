// Created by Siddhant Gangapurwala

#ifndef MULTI_LAYER_PERCEPTRON_HPP
#define MULTI_LAYER_PERCEPTRON_HPP

#include <vector>
#include <fstream>

#include "Activation.hpp"

class MultiLayerPerceptron {
public:
    MultiLayerPerceptron() = delete;

    MultiLayerPerceptron(const std::vector<unsigned int> &networkLayers,
                         const std::reference_wrapper<Activation> &networkActivation,
                         bool outputActivation = false) {
        networkLayers_ = networkLayers;
        outputActivation_ = outputActivation;
        networkActivations_.assign(networkLayers_.size() - (outputActivation_ ? 1 : 2), networkActivation);

        if (outputActivation_) networkActivations_.push_back(networkActivation);

        initializeNetworkVariables();
    }

    MultiLayerPerceptron(const std::vector<unsigned int> &networkLayers,
                         const std::reference_wrapper<Activation> &networkActivation,
                         const std::string &networkParametersPath, bool outputActivation = false)
            : MultiLayerPerceptron(networkLayers, networkActivation, outputActivation) {
        loadNetworkParametersFromFile(networkParametersPath);
    }

    MultiLayerPerceptron(const std::vector<unsigned int> &networkLayers,
                         const std::vector<std::reference_wrapper<Activation>> &networkActivations) {
        networkLayers_ = networkLayers;
        networkActivations_ = networkActivations;

        if (networkActivations_.size() == networkLayers_.size() - 2) outputActivation_ = false;
        else if (networkActivations_.size() == networkLayers_.size() - 1) outputActivation_ = true;
        else throw std::runtime_error(std::string("Check the size of the network layers and activation functions"));

        initializeNetworkVariables();
    }

    MultiLayerPerceptron(const std::vector<unsigned int> &networkLayers,
                         const std::vector<std::reference_wrapper<Activation>> &networkActivations,
                         const std::string &networkParametersPath) : MultiLayerPerceptron::MultiLayerPerceptron(
            networkLayers, networkActivations) {
        loadNetworkParametersFromFile(networkParametersPath);
    }

    void loadNetworkParametersFromFile(const std::string &networkParametersPath) {
        /// https://stackoverflow.com/a/22988866

        if (networkParametersPath_ == networkParametersPath) return;

        std::ifstream dataFile;
        dataFile.open(networkParametersPath);
        std::string line;
        std::vector<double> values;
        unsigned int rows = 0;

        while (std::getline(dataFile, line)) {
            std::stringstream lineStream(line);
            std::string cell;

            while (std::getline(lineStream, cell, ',')) {
                values.push_back(std::stod(cell));
            }

            ++rows;
        }

        networkParameters_ = Eigen::Map<const Eigen::Matrix<typename Eigen::MatrixXd::Scalar,
                Eigen::MatrixXd::RowsAtCompileTime, Eigen::MatrixXd::ColsAtCompileTime,
                Eigen::RowMajor>>(values.data(), rows, values.size() / rows);

        resetNetworkParameters();
        networkParametersPath_ = networkParametersPath;
    }

    const Eigen::MatrixXd &forward(const Eigen::MatrixXd &networkInput) {
        latentOutput_[0] = networkInput;

        for (int layer = 0; layer < networkLayers_.size() - 2; layer++) {
            latentOutput_[layer + 1] = networkWeights_[layer] * latentOutput_[layer] + networkBiases_[layer];
            latentOutput_[layer + 1] = networkActivations_[layer].get().forward(latentOutput_[layer + 1]);
        }

        latentOutput_.back() = networkWeights_.back() * latentOutput_[latentOutput_.size() - 2] + networkBiases_.back();

        if (outputActivation_)
            latentOutput_.back() = networkActivations_[networkActivations_.size() - 1].get().forward(
                    latentOutput_.back());

        return latentOutput_.back();
    }

    const Eigen::MatrixXd &gradient(const Eigen::MatrixXd &networkInput) {
        latentOutput_[0] = networkInput;

        for (int layer = 0; layer < networkLayers_.size() - 2; ++layer) {
            latentLinearOutput_[layer] = networkWeights_[layer] * latentOutput_[layer] + networkBiases_[layer];
            latentOutput_[layer + 1] = networkActivations_[layer].get().forward(latentLinearOutput_[layer]);
        }

        latentLinearOutput_.back() =
                networkWeights_.back() * latentOutput_[latentOutput_.size() - 2] + networkBiases_.back();

        if (outputActivation_) {
            networkDerivative_ =
                    networkActivations_[networkActivations_.size() - 1].get().gradient(
                            latentLinearOutput_.back()).asDiagonal() *
                    networkWeights_.back();
        } else {
            networkDerivative_ = networkWeights_.back();
        }

        for (int layer = static_cast<int>(networkActivations_.size() - (outputActivation_ ? 2 : 1));
             layer >= 0; --layer) {
            networkDerivative_ = networkDerivative_ *
                                 networkActivations_[layer].get().gradient(latentLinearOutput_[layer]).asDiagonal() *
                                 networkWeights_[layer];
        }

        return networkDerivative_;
    }

    const Eigen::MatrixXd &latentLayerOutput(const int &layer) {
        return latentOutput_[layer];
    }

private:
    void initializeNetworkVariables() {
        networkWeights_.resize(networkLayers_.size() - 1);
        networkBiases_.resize(networkLayers_.size() - 1);
        latentLinearOutput_.resize(networkLayers_.size() - 1);
        latentOutput_.resize(networkLayers_.size());

        for (int layer = 0; layer < networkLayers_.size() - 1; ++layer) {
            networkWeights_[layer].resize(networkLayers_[layer + 1], networkLayers_[layer]);
            networkBiases_[layer].resize(networkLayers_[layer + 1], 1);
            latentLinearOutput_[layer].resize(networkLayers_[layer], 1);
            latentOutput_[layer].resize(networkLayers_[layer], 1);
        }

        latentOutput_.back().resize(networkLayers_.back(), 1);
    }

    void resetNetworkParameters() {
        unsigned int networkParametersCount = 0;

        for (int layer = 0; layer < networkLayers_.size() - 1; ++layer) {
            networkParametersCount += networkLayers_[layer] * networkLayers_[layer + 1] + networkLayers_[layer + 1];
        }

        if (networkParametersCount != networkParameters_.rows() * networkParameters_.cols()) {
            throw std::runtime_error(std::string("The number of network parameters loaded are not as expected"));
        }

        unsigned int networkParametersOffset = 0;

        for (int layer = 0; layer < networkLayers_.size() - 1; ++layer) {
            networkWeights_[layer] = Eigen::Map<Eigen::MatrixXd>(
                    networkParameters_.row(0).segment(
                            networkParametersOffset, networkLayers_[layer] * networkLayers_[layer + 1]).data(),
                    networkWeights_[layer].rows(),
                    networkWeights_[layer].cols());
            networkParametersOffset += networkLayers_[layer] * networkLayers_[layer + 1];

            networkBiases_[layer] = Eigen::Map<Eigen::MatrixXd>(
                    networkParameters_.row(0).segment(networkParametersOffset, networkLayers_[layer + 1]).data(),
                    networkBiases_[layer].rows(), networkBiases_[layer].cols());
            networkParametersOffset += networkLayers_[layer + 1];
        }
    }

private:
    // Network Parameters
    std::string networkParametersPath_;
    Eigen::MatrixXd networkParameters_;

    std::vector<Eigen::MatrixXd> networkWeights_;
    std::vector<Eigen::MatrixXd> networkBiases_;

    // Network Layers and Activations
    std::vector<unsigned int> networkLayers_;
    std::vector<std::reference_wrapper<Activation>> networkActivations_;

    // Forward Pass Variable
    std::vector<Eigen::MatrixXd> latentOutput_;

    // Gradient Variables
    std::vector<Eigen::MatrixXd> latentLinearOutput_;
    Eigen::MatrixXd networkDerivative_;

    // Flags
    bool outputActivation_;

};

#endif // MULTI_LAYER_PERCEPTRON_HPP
