// Created by Siddhant Gangapurwala.

#ifndef ACTUATION_DYNAMICS_INFERENCE_ACTUATION_HPP
#define ACTUATION_DYNAMICS_INFERENCE_ACTUATION_HPP

#include "networks_minimal/MultiLayerPerceptron.hpp"
#include "networks_minimal/GatedRecurrentUnit.hpp"


class Actuation {
public:
    Actuation() = delete;

    explicit Actuation(const std::string &parametersDirectory, const int &actuatorCount = 12) :
            recurrentBlock_(2, 8, parametersDirectory + "/gru.txt"),
            denseBlock_({10, 8, 8, 1}, activation.leakyReLu, parametersDirectory + "/mlp.txt") {

        actuatorCount_ = actuatorCount;

        networkInput_.setZero(2, 1);
        networkInputScaling_.setOnes(2, 1);
        denseInput_.setZero(10, 1);

        networkOutputScaling_ = 1.;
        effectiveActuation_.setZero(actuatorCount_);

        for (auto h = 0; h < actuatorCount_; ++h) {
            hiddenStates_.emplace_back(Eigen::MatrixXd::Zero(8, 1));
        }
    }

    Actuation(const std::string &parametersDirectory, const Eigen::MatrixXd &networkInputScaling,
              const double &networkOutputScaling, const int &actuatorCount = 12)  :
            Actuation(parametersDirectory, actuatorCount) {
        networkInputScaling_ = networkInputScaling;
        networkOutputScaling_ = networkOutputScaling;
    }

    const Eigen::VectorXd &getActuationTorques(
            const Eigen::VectorXd &jointPositionErrors, const Eigen::VectorXd &jointVelocities) {

        if (jointPositionErrors.size() != actuatorCount_)
            throw std::runtime_error(std::string("Expected size of jointPositionErrors argument to be ") +
                                     std::to_string(actuatorCount_) + std::string(" but received ") +
                                     std::to_string(jointPositionErrors.size()));

        if (jointVelocities.size() != actuatorCount_)
            throw std::runtime_error(std::string("Expected size of jointVelocities argument to be ") +
                                     std::to_string(actuatorCount_) + std::string(" but received ") +
                                     std::to_string(jointVelocities.size()));

        for (auto j = 0; j < actuatorCount_; ++j) {
            networkInput_.col(0) << jointPositionErrors[j], jointVelocities[j];
            networkInput_ = networkInput_.cwiseProduct(networkInputScaling_);

            hiddenStates_[j] = recurrentBlock_.forward(networkInput_, hiddenStates_[j]);
            denseInput_.col(0) << networkInput_.col(0), hiddenStates_[j].col(0);
            effectiveActuation_[j] = denseBlock_.forward(denseInput_)(0, 0) * networkOutputScaling_;
        }

        return effectiveActuation_;
    }

    void reset() {
        for (auto &hiddenState: hiddenStates_) {
            hiddenState.setZero();
        }
    }

private:
    GatedRecurrentUnit recurrentBlock_;
    MultiLayerPerceptron denseBlock_;

    Eigen::MatrixXd networkInput_, networkInputScaling_;
    Eigen::MatrixXd denseInput_;

    double networkOutputScaling_;

    std::vector<Eigen::MatrixXd> hiddenStates_;

    Eigen::VectorXd effectiveActuation_;

    int actuatorCount_;
};

#endif //ACTUATION_DYNAMICS_INFERENCE_ACTUATION_HPP
