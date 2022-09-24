// Created by Siddhant Gangapurwala

#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include <Eigen/Dense>


class Activation {
public:
    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd &input) {
        return input;
    }

    virtual Eigen::MatrixXd gradient(const Eigen::MatrixXd &input) {
        return Eigen::MatrixXd::Ones(input.rows(), input.cols());
    }
};

class ReLU : public Activation {
public:
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input) override {
        return input.cwiseMax(0.);
    }

    Eigen::MatrixXd gradient(const Eigen::MatrixXd &input) override {
        return (input.array() > 0.).cast<double>();
    }
};

class TanH : public Activation {
public:
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input) override  {
        return input.array().tanh();
    }

    Eigen::MatrixXd gradient(const Eigen::MatrixXd &input) override {
        return 1. - input.array().tanh().pow(2);
    }
};

class SoftSign : public Activation {
public:
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input) override {
        return input.array() / (input.array().abs() + 1.);
    }

    Eigen::MatrixXd gradient(const Eigen::MatrixXd &input) override {
        return 1. / (input.array().abs() + 1.).pow(2);
    }
};

class Sigmoid : public Activation {
public:
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input) override {
        return 1. / ((input.array() * -1.).exp() + 1.);
    }

    Eigen::MatrixXd gradient(const Eigen::MatrixXd &input) override {
        Eigen::MatrixXd sigmoid = Sigmoid::forward(input);
        return sigmoid.array() * (1 - sigmoid.array());
    }
};

class LeakyReLU : public Activation {
public:
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input) override {
        return input.cwiseMax(0.) + (0.01 * input.cwiseMin(0.));
    }

    Eigen::MatrixXd gradient(const Eigen::MatrixXd &input) override {
        Eigen::MatrixXd leakyReLUGradient = input;

        for (auto r = 0; r < leakyReLUGradient.rows(); ++r) {
            for (auto c = 0; c < leakyReLUGradient.cols(); ++c) {
                if (leakyReLUGradient(r, c) > 0.) leakyReLUGradient(r, c) = 1.0;
                else if (leakyReLUGradient(r, c) < 0.) leakyReLUGradient(r, c) = 0.01;
            }
        }

        return leakyReLUGradient;
    }
};

struct ActivationHandler {
    ReLU relu;
    TanH tanh;
    SoftSign softsign;
    Sigmoid sigmoid;
    LeakyReLU leakyReLu;
} activation;

#endif // ACTIVATION_HPP
