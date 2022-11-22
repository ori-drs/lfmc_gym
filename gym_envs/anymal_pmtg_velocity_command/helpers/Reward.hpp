#ifndef _LFMC_GYM_AVC_REWARD_HPP
#define _LFMC_GYM_AVC_REWARD_HPP


class RewardHandler {
private:
    float r_baseOrientation_ = 0.f;
    float r_baseLinearTrackingVelocity_ = 0.f, r_baseAngularVelocityTracking_ = 0.f;
    float r_jointTorque_ = 0.f, r_jointVelocity_ = 0.f;
    float r_actionSmoothness_ = 0.f, r_feetPhase_ = 0.f;
    float r_feetClearance_ = 0.f, r_feetSlip_ = 0.f;
    float r_jointPosition_ = 0.f;
    float r_pronking_ = 0.f, r_trotting_ = 0.f;
    float r_baseHeight_ = 0.f;
    float r_symmetryZ_ = 0.f;
    float r_feetDeviation_ = 0.f;
    float r_jointJerk_ = 0.f;
    float r_verticalLinearVelocity_ = 0.f;
    float r_horizontalAngularVelocity_ = 0.f;

    // Containers
    Eigen::VectorXd desiredJointPosition_, action_;
    Eigen::VectorXd jointVelocity_, prevJointVelocity_;
    Eigen::VectorXd desiredBaseOrientation_;
    std::array<bool, 4> feetContactState_{};
    std::array<float, 4> feetStanceDuration_{};
    std::array<std::string, 4> feetFrames_;
    raisim::Vec<3> framePosition_{}, frameVelocity_{};

    raisim::Vec<3> footPosition1_{}, footPosition2_{};

    int feetInAir_ = 0;
    float simStepsCount_ = 0.f;
    float controlStepTime_, simulationStepTime_;

public:
    explicit RewardHandler(const float &controlStepTime, const float &simulationStepTime) {
        controlStepTime_ = controlStepTime;
        simulationStepTime_ = simulationStepTime;

        desiredBaseOrientation_.setZero(3);
        desiredBaseOrientation_[2] = 1.0;

        feetFrames_[0] = "LF_shank_fixed_LF_FOOT";
        feetFrames_[1] = "RF_shank_fixed_RF_FOOT";
        feetFrames_[2] = "LH_shank_fixed_LH_FOOT";
        feetFrames_[3] = "RH_shank_fixed_RH_FOOT";

        action_.setZero(16);

        desiredJointPosition_.setZero(12);
        desiredJointPosition_ << -0.138589, 0.480936, -0.761428,
                0.138589, 0.480936, -0.761428,
                -0.138589, -0.480936, 0.761428,
                0.138589, -0.480936,  0.761428;

        jointVelocity_.setZero(12);
        prevJointVelocity_.setZero(12);
    }

    void simStep(raisim::ArticulatedSystem *&robot, ObservationHandler &observationHandler,
                 VelocityCommand &velocityCommandHandler, const float &jointTorqueSquaredSum) {
        // Base height
        r_baseHeight_ += squaredTanHyperbolic(robot->getBasePosition()[2] - 
                (observationHandler.getNominalGeneralizedCoordinates()[2] + 0.025));

        // Base orientation
        r_baseOrientation_ += squaredTanHyperbolic(
                observationHandler.getBaseRotation().e().row(2).transpose() - desiredBaseOrientation_, 10.);

        // Base linear velocity
        r_baseLinearTrackingVelocity_ +=
                1.f - static_cast<float>(tanh(4. * (observationHandler.getBaseLinearVelocity().col(0).head(2) -
                                                    observationHandler.getDesiredBaseLinearVelocity().col(0).head(
                                                            2)).squaredNorm()));

        // Base angular velocity
        r_baseAngularVelocityTracking_ +=
                1.f - static_cast<float>(tanh(2 * std::pow(observationHandler.getBaseAngularVelocity()[2] -
                                                           observationHandler.getDesiredBaseAngularVelocity()[2], 2.)));

        // Joint torque
        r_jointTorque_ += 0.001f * jointTorqueSquaredSum;

        // Joint velocity
        r_jointVelocity_ += 0.01f * static_cast<float>(
                observationHandler.getGeneralizedVelocity().tail(12).squaredNorm());

        // Joint position
        r_jointPosition_ += static_cast<float>((
                observationHandler.getDesiredJointPosition() -
                observationHandler.getNominalJointConfiguration().tail(12)
        ).cwiseMax(-M_PI_4).cwiseMin(M_PI_4).squaredNorm());

        /// Symmetry - Z (heights of diagonal feet should match)
        robot->getFramePosition(feetFrames_[0], footPosition1_);
        robot->getFramePosition(feetFrames_[3], footPosition2_);
        r_symmetryZ_ += static_cast<float>(std::pow((footPosition1_ - footPosition2_)[2], 2.));

        robot->getFramePosition(feetFrames_[1], footPosition1_);
        robot->getFramePosition(feetFrames_[2], footPosition2_);
        r_symmetryZ_ += static_cast<float>(std::pow((footPosition1_ - footPosition2_)[2], 2.));

        /// Feet positions should not deviate significantly from the hip
        robot->getFramePosition(feetFrames_[0], footPosition1_);
        r_feetDeviation_ += static_cast<float>(
                ((observationHandler.getBaseRotationVerticalComponent().e().transpose() *
                  (footPosition1_.e() - robot->getBasePosition().e())) -
                 observationHandler.getNominalFeetPositions().segment(0, 3)).head(2).squaredNorm());

        robot->getFramePosition(feetFrames_[1], footPosition1_);
        r_feetDeviation_ += static_cast<float>(
                ((observationHandler.getBaseRotationVerticalComponent().e().transpose() *
                  (footPosition1_.e() - robot->getBasePosition().e())) -
                 observationHandler.getNominalFeetPositions().segment(3, 3)).head(2).squaredNorm());

        robot->getFramePosition(feetFrames_[2], footPosition1_);
        r_feetDeviation_ += static_cast<float>(
                ((observationHandler.getBaseRotationVerticalComponent().e().transpose() *
                  (footPosition1_.e() - robot->getBasePosition().e())) -
                 observationHandler.getNominalFeetPositions().segment(6, 3)).head(2).squaredNorm());

        robot->getFramePosition(feetFrames_[3], footPosition1_);
        r_feetDeviation_ += static_cast<float>(
                ((observationHandler.getBaseRotationVerticalComponent().e().transpose() *
                  (footPosition1_.e() - robot->getBasePosition().e())) -
                 observationHandler.getNominalFeetPositions().segment(9, 3)).head(2).squaredNorm());

        // Get feet contact state
        feetContactState_.fill(false);
        feetInAir_ = 4;

        for (auto &contact: robot->getContacts()) {
            if (contact.getlocalBodyIndex() == robot->getBodyIdx("LF_SHANK")) {
                feetContactState_[0] = true;
                feetInAir_ -= 1;
            } else if (contact.getlocalBodyIndex() == robot->getBodyIdx("RF_SHANK")) {
                feetContactState_[1] = true;
                feetInAir_ -= 1;
            } else if (contact.getlocalBodyIndex() == robot->getBodyIdx("LH_SHANK")) {
                feetContactState_[2] = true;
                feetInAir_ -= 1;
            } else if (contact.getlocalBodyIndex() == robot->getBodyIdx("RH_SHANK")) {
                feetContactState_[3] = true;
                feetInAir_ -= 1;
            }
        }

        if (velocityCommandHandler.zeroCommandMode()) {
            r_pronking_ += feetInAir_ >= 1 ? 1.f : 0.f;
        } else {
            r_pronking_ += feetInAir_ > 2 ? 1.f : 0.f;
        }

        // Get the RL agent to ensure the feet are in stance phase during zero velocity command
        if (velocityCommandHandler.zeroCommandMode()) {
            for (auto f = 0; f < observationHandler.getFeetPhase().size(); ++f) {
                r_feetPhase_ += observationHandler.getFeetPhase()[f] > 0 ? 1. : 0.;
            }
        }

        if (!velocityCommandHandler.zeroCommandMode()) {
            if ((feetContactState_[0] == feetContactState_[3]) && (feetContactState_[1] == feetContactState_[2]) &&
                (feetContactState_[0] != feetContactState_[1])) {
                r_trotting_ += 1.f;
            } else r_trotting_ += 0.f;
        } else {
            r_trotting_ += feetInAir_ == 0 ? 1.25f : 0.f;
        }

        for (auto f = 0; f < feetContactState_.size(); ++f) {
            robot->getFramePosition(feetFrames_[f], framePosition_);
            robot->getFrameVelocity(feetFrames_[f], frameVelocity_);

            if (!feetContactState_[f]) {
                feetStanceDuration_[f] = std::min(0.f, feetStanceDuration_[f]) - simulationStepTime_;
            } else {
                r_feetSlip_ += static_cast<float>(frameVelocity_.e().head(2).squaredNorm());
                feetStanceDuration_[f] = std::max(0.f, feetStanceDuration_[f]) + simulationStepTime_;
            }

            if (!velocityCommandHandler.zeroCommandMode() && feetStanceDuration_[f] > -0.6f &&
                feetStanceDuration_[f] < 0.f) {
                r_feetClearance_ += std::min(-feetStanceDuration_[f], 0.3f);
            }
        }

        // Vertical velocity
        r_verticalLinearVelocity_ += static_cast<float>(std::pow(observationHandler.getBaseLinearVelocity()[2], 2));

        // Horizontal Angular velocity
        r_horizontalAngularVelocity_ += static_cast<float>(
                observationHandler.getBaseAngularVelocity().head(2).squaredNorm());

        /// Increment Sim Steps Counter
        simStepsCount_ += 1.f;
    }

    void computeRewards(ObservationHandler &observationHandler) {
        if (simStepsCount_ < 1.f) {
            clearBuffers();
            simStepsCount_ = 0.f;

            return;
        }

        // Joint jerk
        r_jointJerk_ = static_cast<float>(
                (observationHandler.getGeneralizedVelocity().tail(12) - 2. * jointVelocity_ +
                 prevJointVelocity_).squaredNorm());

        // Smoothness
        r_actionSmoothness_ = static_cast<float>(
                (action_ - observationHandler.getAction()).squaredNorm());

        /// Store previous states
        action_ = observationHandler.getAction();
        desiredJointPosition_ = observationHandler.getDesiredJointPosition();

        prevJointVelocity_ = jointVelocity_;
        jointVelocity_ = observationHandler.getGeneralizedVelocity().tail(12);

        r_baseHeight_ /= simStepsCount_;
        r_baseOrientation_ /= simStepsCount_;
        r_baseLinearTrackingVelocity_ /= simStepsCount_;
        r_baseAngularVelocityTracking_ /= simStepsCount_;
        r_jointTorque_ /= simStepsCount_;
        r_jointVelocity_ /= simStepsCount_;
        r_jointPosition_ /= simStepsCount_;
        r_symmetryZ_ /= simStepsCount_;
        r_feetDeviation_ /= simStepsCount_;
        r_pronking_ /= simStepsCount_;
        r_trotting_ /= simStepsCount_;
        r_feetSlip_ /= simStepsCount_;
        r_feetClearance_ /= simStepsCount_;
        r_verticalLinearVelocity_ /= simStepsCount_;
        r_horizontalAngularVelocity_ /= simStepsCount_;
        r_feetPhase_ /= simStepsCount_;

        simStepsCount_ = 0.f;
    }

    void clearBuffers() {
        r_baseHeight_ = 0.f;
        r_baseOrientation_ = 0.f;
        r_baseLinearTrackingVelocity_ = 0.f;
        r_baseAngularVelocityTracking_ = 0.f;
        r_jointTorque_ = 0.f;
        r_jointJerk_ = 0.f;
        r_jointVelocity_ = 0.f;
        r_jointPosition_ = 0.f;
        r_symmetryZ_ = 0.f;
        r_feetDeviation_ = 0.f;
        r_pronking_ = 0.f;
        r_trotting_ = 0.f;
        r_feetSlip_ = 0.f;
        r_feetClearance_ = 0.f;
        r_actionSmoothness_ = 0.f;
        r_verticalLinearVelocity_ = 0.f;
        r_horizontalAngularVelocity_ = 0.f;
        r_feetPhase_ = 0.f;
    }

    void reset() {
        action_.setZero();
        desiredJointPosition_ << -0.089, 0.712, -1.03,
                0.089, 0.712, -1.03,
                -0.089, -0.712, 1.03,
                0.089, -0.712, 1.03;
        feetStanceDuration_.fill(0.f);

        prevJointVelocity_.setZero();
        jointVelocity_.setZero();

        simStepsCount_ = 0.f;
        clearBuffers();
    }

    void reset(const Eigen::VectorXd &jointPosition) {
        reset();
        desiredJointPosition_ = jointPosition;
    }

    float &getBaseOrientationReward() {
        return r_baseOrientation_;
    }

    float &getFeetPhaseReward() {
        return r_feetPhase_;
    }

    float &getBaseLinearVelocityTrackingReward() {
        return r_baseLinearTrackingVelocity_;
    }

    float &getBaseAngularVelocityTrackingReward() {
        return r_baseAngularVelocityTracking_;
    }

    float &getJointTorqueReward() {
        return r_jointTorque_;
    }

    float &getJointVelocityReward() {
        return r_jointVelocity_;
    }

    float &getJointPositionReward() {
        return r_jointPosition_;
    }

    float &getActionSmoothnessReward() {
        return r_actionSmoothness_;
    }

    float &getFeetClearanceReward() {
        return r_feetClearance_;
    }

    float &getFeetSlipReward() {
        return r_feetSlip_;
    }

    float &getPronkingReward() {
        return r_pronking_;
    }

    float &getBaseHeightReward() {
        return r_baseHeight_;
    }

    float &getSymmetryZReward() {
        return r_symmetryZ_;
    }

    float &getFeetDeviationReward() {
        return r_feetDeviation_;
    }

    float &getTrottingReward() {
        return r_trotting_;
    }

    float &getJointJerkReward() {
        return r_jointJerk_;
    }

    float &getVerticalLinearVelocityReward() {
        return r_verticalLinearVelocity_;
    }

    float &getHorizontalAngularVelocityReward() {
        return r_horizontalAngularVelocity_;
    }

    static double logisticKernel(double x) {
        return 1. / (std::exp(x) + 2. + std::exp(-x));
    }

    static double logisticKernel(const Eigen::VectorXd &x) {
        double s = 0;

        for (auto i = 0; i < x.size(); ++i) {
            s += logisticKernel(x[i]);
        }

        return s;
    }

    static double exponentialSquaredError(double x, double y, double scaling = -1.) {
        return exp(scaling * std::pow((x - y), 2.));
    }

    static double exponentialSquaredError(const Eigen::VectorXd &x, const Eigen::VectorXd &y, double scaling = -1.) {
        Eigen::VectorXd e = x - y;
        return exp(scaling * e.cwiseProduct(e).sum());
    }

    static float squaredTanHyperbolic(const double &x, double scaling = 1.) {
        return static_cast<float>(tanh(scaling * x * x));
    }

    static float squaredTanHyperbolic(const Eigen::VectorXd &x, double scaling = 1.) {
        return static_cast<float>(tanh(scaling * x.squaredNorm()));
    }

    static float inverseSquaredTanHyperbolic(const double &x, double scaling = 1.) {
        return 1.f - squaredTanHyperbolic(x, scaling);
    }

    static float inverseSquaredTanHyperbolic(const Eigen::VectorXd &x, double scaling = 1.) {
        return 1.f - squaredTanHyperbolic(x, scaling);
    }
};

#endif //_LFMC_GYM_AVC_REWARD_HPP
