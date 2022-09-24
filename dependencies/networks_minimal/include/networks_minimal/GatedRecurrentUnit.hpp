// Created by Siddhant Gangapurwala

#ifndef GATED_RECURRENT_UNIT_HPP
#define GATED_RECURRENT_UNIT_HPP

#include <vector>
#include <fstream>

#include "Activation.hpp"

class GatedRecurrentUnit {
public:
    GatedRecurrentUnit() = delete;

    GatedRecurrentUnit(const unsigned int &inputDimension, const unsigned int &hiddenStateDimension) {
        inputDimension_ = inputDimension;
        hiddenStateDimension_ = hiddenStateDimension;
        tripledHiddenStateDimension_ = 3 * hiddenStateDimension_;

        // Initialize hidden state to zeros
        networkHiddenState_.setZero(hiddenStateDimension_, 1);

        // Initialize network forward pass containers
        inputGatesLatent_.setZero(tripledHiddenStateDimension_, 1);
        hiddenGatesLatent_.setZero(tripledHiddenStateDimension_, 1);

        rInputGatesLatent_.setZero(hiddenStateDimension_, 1);
        zInputGatesLatent_.setZero(hiddenStateDimension_, 1);
        nInputGatesLatent_.setZero(hiddenStateDimension_, 1);

        rHiddenGatesLatent_.setZero(hiddenStateDimension_, 1);
        zHiddenGatesLatent_.setZero(hiddenStateDimension_, 1);
        nHiddenGatesLatent_.setZero(hiddenStateDimension_, 1);

        rLatent_.setZero(hiddenStateDimension_, 1);
        zLatent_.setZero(hiddenStateDimension_, 1);
        nLatent_.setZero(hiddenStateDimension_, 1);

        // Resize network weights and bias vectors
        networkWeights_.resize(2);
        networkBiases_.resize(2);

        networkWeights_[0].resize(tripledHiddenStateDimension_, inputDimension_);
        networkBiases_[0].resize(tripledHiddenStateDimension_, 1);

        networkWeights_[1].resize(tripledHiddenStateDimension_, hiddenStateDimension_);
        networkBiases_[1].resize(tripledHiddenStateDimension_, 1);
    }

    GatedRecurrentUnit(const unsigned int &inputDimension, const unsigned int &hiddenStateDimension,
                       const std::string &networkParametersPath)
            : GatedRecurrentUnit::GatedRecurrentUnit(inputDimension, hiddenStateDimension) {
        loadNetworkParametersFromFile(networkParametersPath);
    }

    void loadNetworkParametersFromFile(const std::string &networkParametersPath) {
        /// https://stackoverflow.com/a/22988866

        if (networkParametersPath_ == networkParametersPath) return;

        std::ifstream dataFile;
        dataFile.open(networkParametersPath);
        std::string line;
        std::vector<double> values;
        unsigned int _rows = 0;

        while (std::getline(dataFile, line)) {
            std::stringstream lineStream(line);
            std::string cell;

            while (std::getline(lineStream, cell, ',')) {
                values.push_back(std::stod(cell));
            }

            ++_rows;
        }

        networkParameters_ = Eigen::Map<const Eigen::Matrix<typename Eigen::MatrixXd::Scalar,
                Eigen::MatrixXd::RowsAtCompileTime, Eigen::MatrixXd::ColsAtCompileTime,
                Eigen::RowMajor>>(values.data(), _rows, static_cast<long>(values.size()) / _rows);

        resetNetworkParameters();
        networkParametersPath_ = networkParametersPath;
    }

    const Eigen::MatrixXd &forward(const Eigen::MatrixXd &networkInput) {
        return forward(networkInput, networkHiddenState_);
    }

    const Eigen::MatrixXd &forward(const Eigen::MatrixXd &networkInput, const Eigen::MatrixXd &networkHiddenState) {
        inputGatesLatent_ = networkWeights_[0] * networkInput + networkBiases_[0];
        hiddenGatesLatent_ = networkWeights_[1] * networkHiddenState + networkBiases_[1];

        rInputGatesLatent_.col(0) = inputGatesLatent_.col(0).head(hiddenStateDimension_);
        zInputGatesLatent_.col(0) = inputGatesLatent_.col(0).segment(hiddenStateDimension_, hiddenStateDimension_);
        nInputGatesLatent_.col(0) = inputGatesLatent_.col(0).tail(hiddenStateDimension_);

        rHiddenGatesLatent_.col(0) = hiddenGatesLatent_.col(0).head(hiddenStateDimension_);
        zHiddenGatesLatent_.col(0) = hiddenGatesLatent_.col(0).segment(hiddenStateDimension_, hiddenStateDimension_);
        nHiddenGatesLatent_.col(0) = hiddenGatesLatent_.col(0).tail(hiddenStateDimension_);

        rLatent_ = activation.sigmoid.forward(rInputGatesLatent_ + rHiddenGatesLatent_);
        zLatent_ = activation.sigmoid.forward(zInputGatesLatent_ + zHiddenGatesLatent_);
        nLatent_ = activation.tanh.forward(nInputGatesLatent_ + rLatent_.cwiseProduct(nHiddenGatesLatent_));

        networkHiddenState_ = nLatent_ + (zLatent_.cwiseProduct(networkHiddenState - nLatent_));
        return networkHiddenState_;
    }

    void resetHiddenState() {
        networkHiddenState_.setZero();
    }

    void resetHiddenState(const Eigen::MatrixXd &networkHiddenState) {
        networkHiddenState_ = networkHiddenState;
    }

    const Eigen::MatrixXd &getNetworkHiddenState() {
        return networkHiddenState_;
    }

private:
    void resetNetworkParameters() {
        unsigned int networkParametersCount =
                (inputDimension_ * tripledHiddenStateDimension_) + tripledHiddenStateDimension_;
        networkParametersCount += (hiddenStateDimension_ * tripledHiddenStateDimension_) + tripledHiddenStateDimension_;

        if (networkParametersCount != networkParameters_.rows() * networkParameters_.cols()) {
            throw std::runtime_error(
                    std::string("The number of network parameters loaded are not as expected. Required ") +
                    std::to_string(networkParametersCount) + std::string(" but received ") +
                    std::to_string(networkParameters_.rows() * networkParameters_.cols()));
        }

        unsigned int networkParametersOffset = 0;

        networkWeights_[0] = Eigen::Map<Eigen::MatrixXd>(
                networkParameters_.row(0).segment(networkParametersOffset,
                                                  inputDimension_ * tripledHiddenStateDimension_).data(),
                networkWeights_[0].rows(), networkWeights_[0].cols());
        networkParametersOffset += inputDimension_ * tripledHiddenStateDimension_;

        networkBiases_[0] = Eigen::Map<Eigen::MatrixXd>(
                networkParameters_.row(0).segment(networkParametersOffset, tripledHiddenStateDimension_).data(),
                networkBiases_[0].rows(), networkBiases_[0].cols());
        networkParametersOffset += tripledHiddenStateDimension_;

        networkWeights_[1] = Eigen::Map<Eigen::MatrixXd>(
                networkParameters_.row(0).segment(networkParametersOffset,
                                                  hiddenStateDimension_ * tripledHiddenStateDimension_).data(),
                networkWeights_[1].rows(), networkWeights_[1].cols());
        networkParametersOffset += hiddenStateDimension_ * tripledHiddenStateDimension_;

        networkBiases_[1] = Eigen::Map<Eigen::MatrixXd>(
                networkParameters_.row(0).segment(networkParametersOffset, tripledHiddenStateDimension_).data(),
                networkBiases_[1].rows(), networkBiases_[1].cols());
    }


private:
    // Network Input and Hidden State Dimensions
    unsigned int inputDimension_;
    unsigned int hiddenStateDimension_, tripledHiddenStateDimension_;

    // Network Parameters
    std::string networkParametersPath_;
    Eigen::MatrixXd networkParameters_;

    std::vector<Eigen::MatrixXd> networkWeights_;
    std::vector<Eigen::MatrixXd> networkBiases_;

    Eigen::MatrixXd networkHiddenState_;

    // Forward pass containers
    Eigen::MatrixXd inputGatesLatent_, hiddenGatesLatent_;
    Eigen::MatrixXd rInputGatesLatent_, zInputGatesLatent_, nInputGatesLatent_;
    Eigen::MatrixXd rHiddenGatesLatent_, zHiddenGatesLatent_, nHiddenGatesLatent_;
    Eigen::MatrixXd rLatent_, zLatent_, nLatent_;
};

#endif // GATED_RECURRENT_UNIT_HPP
