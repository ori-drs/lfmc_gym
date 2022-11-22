#ifndef _LFMC_GYM_AVC_COMMAND_HPP
#define _LFMC_GYM_AVC_COMMAND_HPP


enum VelocityCommandMode {
    Zero,
    Heading,
    Lateral,
    Yaw,
    Direction,
    Constant
};

class VelocityCommand {
public:
    explicit VelocityCommand(const Yaml::Node &cfg, const double &controlTimeStep)
            : uniformRealDistribution_(-1., 1.) {
        controlTimeStep_ = controlTimeStep;

        velocityCommandLimits_[0] = cfg["limit_heading_velocity"].template As<double>();
        velocityCommandLimits_[1] = cfg["limit_lateral_velocity"].template As<double>();
        velocityCommandLimits_[2] = cfg["limit_yaw_rate"].template As<double>();

        velocityCommandLimitsFinal_[0] = cfg["limit_heading_velocity_final"].template As<double>();
        velocityCommandLimitsFinal_[1] = cfg["limit_lateral_velocity_final"].template As<double>();
        velocityCommandLimitsFinal_[2] = cfg["limit_yaw_rate_final"].template As<double>();

        velocityLimitIncrementStep_ = std::max(1., cfg["limit_increment_reset_steps"].template As<double>());

        velocityCommandLimitsIncrement_[0] =
                (velocityCommandLimitsFinal_[0] - velocityCommandLimits_[0]) / velocityLimitIncrementStep_;
        velocityCommandLimitsIncrement_[1] =
                (velocityCommandLimitsFinal_[1] - velocityCommandLimits_[1]) / velocityLimitIncrementStep_;
        velocityCommandLimitsIncrement_[2] =
                (velocityCommandLimitsFinal_[2] - velocityCommandLimits_[2]) / velocityLimitIncrementStep_;

        limitVelocityMagnitude_ = cfg["limit_velocity_magnitude"].template As<double>();

        commandProbabilities_[VelocityCommandMode::Zero] = cfg["probability_zero_command"].template As<double>();
        commandProbabilities_[VelocityCommandMode::Heading] = cfg["probability_heading_command"].template As<double>();
        commandProbabilities_[VelocityCommandMode::Lateral] = cfg["probability_lateral_command"].template As<double>();
        commandProbabilities_[VelocityCommandMode::Yaw] = cfg["probability_yaw_command"].template As<double>();
        commandProbabilities_[VelocityCommandMode::Direction] = cfg["probability_direction_command"].template As<double>();
        commandProbabilities_[VelocityCommandMode::Constant] = cfg["probability_constant_command"].template As<double>();

        validateProbabilities();

        commandProbabilitiesKeys_.reserve(commandProbabilities_.size());
        commandProbabilitiesValues_.reserve(commandProbabilities_.size());

        for (const auto &p: commandProbabilities_) {
            commandProbabilitiesKeys_.push_back(p.first);
            commandProbabilitiesValues_.push_back(p.second);
        }

        commandModeSamplingDistribution_ = std::discrete_distribution(
                cbegin(commandProbabilitiesValues_), cend(commandProbabilitiesValues_));

        commandSamplingStepsRange_[0] = static_cast<int>(floor(
                cfg["command_sampling_time_min"].template As<double>() / controlTimeStep_));
        commandSamplingStepsRange_[1] = static_cast<int>(ceil(
                cfg["command_sampling_time_max"].template As<double>() / controlTimeStep_));

        velocityCommand_.setZero();
        directionGoal_.setZero();
    }

    void step(const Eigen::Matrix3d &robotRotation) {
        if (stepsUntilNextSample_-- <= 0) {
            auto commandMode = commandMode_;
            commandMode_ = sampleCommandMode();

            // With a probability of 0.125, invert the previous velocity command
            if (commandMode == commandMode_ && commandMode_ != VelocityCommandMode::Direction &&
                uniformRealDistribution_(gen_) < -0.75) {
                velocityCommand_ *= -1.;

                stepsUntilNextSample_ = std::uniform_int_distribution<int>(
                        commandSamplingStepsRange_[0], commandSamplingStepsRange_[1])(gen_);
            } else {
                sampleVelocityCommand();
            }
        }

        if (commandMode_ == VelocityCommandMode::Direction) {
            updateVelocityForGoal(robotRotation);
        }
    }

    void reset(const Eigen::Matrix3d &robotRotation) {
        stepsUntilNextSample_ = 0;
        step(robotRotation);
    }

    void incrementVelocityCommandLimits() {
        for (auto v = 0; v < velocityCommandLimits_.size(); ++v) {
            if (velocityCommandLimits_[v] <= velocityCommandLimitsFinal_[v] - velocityCommandLimitsIncrement_[v]) {
                velocityCommandLimits_[v] += velocityCommandLimitsIncrement_[v];
            }
        }
    }

    const Eigen::Vector3d &getVelocityCommand() {
        return velocityCommand_;
    }

    void setSeed(const int &seed) {
        gen_.seed(seed);
    }

    bool zeroCommandMode() {
        return commandMode_ == VelocityCommandMode::Zero;
    }

private:
    void sampleVelocityCommand() {
        velocityCommand_ = Eigen::Vector3d::NullaryExpr([&]() { return uniformRealDistribution_(gen_); });
        velocityCommand_ = velocityCommand_.cwiseProduct(velocityCommandLimits_);

        stepsUntilNextSample_ = std::uniform_int_distribution<int>(
                commandSamplingStepsRange_[0], commandSamplingStepsRange_[1])(gen_);

        if (commandMode_ == VelocityCommandMode::Zero) {
            velocityCommand_.setZero();
        } else if (commandMode_ == VelocityCommandMode::Heading) {
            velocityCommand_.tail(2).setZero();
            validateVelocityCommand();
        } else if (commandMode_ == VelocityCommandMode::Lateral) {
            velocityCommand_[0] = 0.;
            velocityCommand_[2] = 0.;
            validateVelocityCommand();
        } else if (commandMode_ == VelocityCommandMode::Yaw) {
            velocityCommand_.head(2).setZero();
            validateVelocityCommand();
        } else if (commandMode_ == VelocityCommandMode::Constant) {
            validateVelocityCommand();
        } else {
            updateDirectionGoal();
        }
    }

    VelocityCommandMode sampleCommandMode() {
        return static_cast<VelocityCommandMode>(commandModeSamplingDistribution_(gen_));
    }

    void validateVelocityCommand() {
        double velocityCommandNorm = velocityCommand_.norm();

        if (velocityCommandNorm < limitVelocityMagnitude_) {
            if (velocityCommandNorm < 1e-2) {
                velocityCommand_.setZero();

                if (commandMode_ == VelocityCommandMode::Heading) {
                    velocityCommand_[0] = getRandomSignDouble() * limitVelocityMagnitude_;
                } else if (commandMode_ == VelocityCommandMode::Lateral) {
                    velocityCommand_[1] = getRandomSignDouble() * limitVelocityMagnitude_;
                } else if (commandMode_ == VelocityCommandMode::Yaw) {
                    velocityCommand_[2] = getRandomSignDouble() * limitVelocityMagnitude_;
                } else if (commandMode_ == VelocityCommandMode::Constant) {
                    velocityCommand_ = Eigen::Vector3d::NullaryExpr([&]() { return uniformRealDistribution_(gen_); });
                    validateVelocityCommand();
                }
            } else {
                velocityCommand_ = velocityCommand_ * (limitVelocityMagnitude_ / velocityCommandNorm);
            }
        }
    }

    void updateDirectionGoal() {
        directionGoal_ = Eigen::Vector2d::NullaryExpr([&]() { return uniformRealDistribution_(gen_); });
    }

    void updateVelocityForGoal(const Eigen::Matrix3d &robotRotation) {
        desiredHeadingAngle_ = std::atan2(directionGoal_[1], directionGoal_[0]);
        robotHeadingAngle_ = std::atan2(robotRotation.col(0)[1], robotRotation.col(0)[0]);
        headingAngleOffset_ = std::atan2(std::sin(desiredHeadingAngle_ - robotHeadingAngle_),
                                         std::cos(desiredHeadingAngle_ - robotHeadingAngle_));
        velocityCommand_[0] = std::cos(headingAngleOffset_);
        velocityCommand_[1] = std::sin(headingAngleOffset_);
    }

    void validateProbabilities() {
        double sum = 0.;

        for (auto &p: commandProbabilities_) {
            sum += p.second;
        }

        if (sum < 1e-4) {
            for (auto &p: commandProbabilities_) {
                p.second = 0.;
            }

            commandProbabilities_[VelocityCommandMode::Zero] = 0.2;
            commandProbabilities_[VelocityCommandMode::Constant] = 0.8;
        } else {
            for (auto &p: commandProbabilities_) {
                p.second /= sum;
            }
        }
    }

    int getRandomSignInt() {
        return randomSignDistribution_(gen_) == 0 ? -1 : 1;
    }

    double getRandomSignDouble() {
        return randomSignDistribution_(gen_) == 0 ? -1. : 1.;
    }

private:
    Eigen::Vector3d velocityCommandLimits_, velocityCommandLimitsFinal_, velocityCommandLimitsIncrement_;
    double limitVelocityMagnitude_;

    std::map<VelocityCommandMode, double> commandProbabilities_;

    std::vector<VelocityCommandMode> commandProbabilitiesKeys_;
    std::vector<double> commandProbabilitiesValues_;

    std::array<int, 2> commandSamplingStepsRange_{};
    int stepsUntilNextSample_{};
    double controlTimeStep_, velocityLimitIncrementStep_;

    std::discrete_distribution<int> commandModeSamplingDistribution_, randomSignDistribution_{0.5, 0.5};
    std::uniform_real_distribution<double> uniformRealDistribution_;
    std::mt19937 gen_;

    VelocityCommandMode commandMode_;
    Eigen::Vector3d velocityCommand_;
    Eigen::Vector2d directionGoal_;

    double desiredHeadingAngle_ = 0., robotHeadingAngle_ = 0., headingAngleOffset_ = 0.;
};

#endif //_LFMC_GYM_AVC_COMMAND_HPP
