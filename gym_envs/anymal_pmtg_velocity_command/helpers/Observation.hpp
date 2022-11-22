#ifndef _LFMC_GYM_AVC_OBSERVATION_HPP
#define _LFMC_GYM_AVC_OBSERVATION_HPP


class ObservationHandler {
private:
    Eigen::VectorXd observation_;
    Eigen::VectorXd jointPositionErrorHistory_, jointVelocityHistory_;

    Eigen::VectorXd jointNominalConfig_, feetNominalPositions_;
    Eigen::VectorXd generalizedCoordinate_, generalizedVelocity_, generalizedForce_;
    Eigen::VectorXd baseLinearVelocity_, baseAngularVelocity_;
    Eigen::VectorXd desiredBaseLinearVelocity_, desiredBaseAngularVelocity_;
    Eigen::VectorXd desiredJointPosition_, action_, feetPhase_;
    raisim::Mat<3, 3> baseRotation_{}, baseRotationVerticalComponent_{};
    raisim::Vec<3> baseQuaternionVerticalComponent_{};

    Eigen::VectorXd varJoint_;

    int observationDim_ = 72;
    int historyLength_ = 4, nJoints_ = 12;
    double nominalBaseHeight_ = 0.55;

    Eigen::VectorXd nominalFeetPhase_;
    Eigen::VectorXd nominalGeneralizedCoordinates_;

public:
    ObservationHandler() {
        observation_.setZero(observationDim_);

        jointVelocityHistory_.setZero(historyLength_ * nJoints_);
        jointPositionErrorHistory_.setZero(historyLength_ * nJoints_);

        jointNominalConfig_.setZero(nJoints_);
        jointNominalConfig_ << -0.138589, 0.480936, -0.761428,
                0.138589, 0.480936, -0.761428,
                -0.138589, -0.480936, 0.761428,
                0.138589, -0.480936, 0.761428;

        feetNominalPositions_.setZero(12);
        feetNominalPositions_ << 0.3 + 0.1, 0.2, -0.55,
                0.3 + 0.1, -0.2, -0.55,
                -0.3 - 0.1, 0.2, -0.55,
                -0.3 - 0.1, -0.2, -0.55;

        generalizedCoordinate_.setZero(19);
        generalizedVelocity_.setZero(18);
        generalizedForce_.setZero(18);

        baseLinearVelocity_.setZero(3);
        baseAngularVelocity_.setZero(3);
        desiredBaseLinearVelocity_.setZero(3);
        desiredBaseAngularVelocity_.setZero(3);

        baseRotation_.setIdentity();
        varJoint_.setZero(nJoints_);

        nominalGeneralizedCoordinates_.setZero(19);
        nominalGeneralizedCoordinates_[2] = nominalBaseHeight_;
        nominalGeneralizedCoordinates_[3] = 1.;
        nominalGeneralizedCoordinates_.tail(12) = jointNominalConfig_;

        nominalFeetPhase_.setZero(4);
        nominalFeetPhase_ << -M_PI, 0., 0., -M_PI;

        action_.setZero(16);
        feetPhase_.setZero(4);
    }

    void setSeed(const int &seed) {
        srand(seed);
    }

    void updateObservation(raisim::ArticulatedSystem *&robot, const Eigen::VectorXd &action,
                           const Eigen::VectorXd &feetPhase, const Eigen::VectorXd &desiredJointPosition,
                           bool addCommandNoise = false) {
        robot->getState(generalizedCoordinate_, generalizedVelocity_);
        generalizedForce_ = robot->getGeneralizedForce().e();
        baseRotation_ = robot->getBaseOrientation();

        // Get vertical component of rotation
        baseQuaternionVerticalComponent_ = generalizedCoordinate_.segment(3, 4);
        baseQuaternionVerticalComponent_.e().segment(1, 2).setZero();
        baseQuaternionVerticalComponent_.e().normalize();

        quaternionToRotationMatrix(baseQuaternionVerticalComponent_, baseRotationVerticalComponent_);

        feetPhase_ = feetPhase;
        action_ = action;
        desiredJointPosition_ = desiredJointPosition;

        updateJointHistory(desiredJointPosition);

        baseLinearVelocity_ = baseRotation_.e().transpose() * generalizedVelocity_.segment(0, 3);
        baseAngularVelocity_ = baseRotation_.e().transpose() * generalizedVelocity_.segment(3, 3);

        /// Set State Vector
        observation_.segment(0, 3) = baseRotation_.e().row(2).transpose();   // Indices 0 - 2: Gravity Axes
        observation_.segment(3, nJoints_) = generalizedCoordinate_.tail(nJoints_);    // Indices 3 - 14: Joint Positions
        observation_.segment(15, 3) = baseAngularVelocity_;    // Index 15 - 17: Angular Velocity
        observation_.segment(18, nJoints_) = generalizedVelocity_.tail(nJoints_);   // Index 18 - 29: Joint Velocities
        observation_.segment(30, 3) = baseLinearVelocity_;  // Index 30 - 32: Base Linear Velocity
        observation_.segment(33, 2) = desiredBaseLinearVelocity_.head(2);  // Index 33 - 34: Velocity Command Linear
        observation_.segment(35, 1) = desiredBaseAngularVelocity_.tail(1);  // Index 35: Velocity Command Angular
        observation_.segment(36, 12) =
                desiredJointPosition - generalizedCoordinate_.tail(12);  // Index 36 - 48: Joint position error
        observation_.segment(48, 16) = action;  // Index 48 - 64: Previous action
        observation_.segment(64, 4) = vectorSin(feetPhase);  // Index 64 - 68: Previous action (sin of phases)
        observation_.segment(68, 4) = vectorCos(feetPhase);  // Index 68 - 72: Previous action (cos of phases)

        if (addCommandNoise) {
            observation_.segment(33, 2) = desiredBaseLinearVelocity_.head(2) + 0.025 * Eigen::VectorXd::Random(2);
            observation_.segment(35, 1) = desiredBaseAngularVelocity_.tail(1) + 0.025 * Eigen::VectorXd::Random(1);
        }
    }

    void updateVelocityCommand(const Eigen::VectorXd &velocityCommand) {
        desiredBaseLinearVelocity_.setZero();
        desiredBaseLinearVelocity_.head(2) = velocityCommand.head(2);

        desiredBaseAngularVelocity_.setZero();
        desiredBaseAngularVelocity_[2] = velocityCommand[2];

        observation_.segment(33, 2) = desiredBaseLinearVelocity_.head(2);  // Index 33 - 34: Velocity Command Linear
        observation_.segment(35, 1) = desiredBaseAngularVelocity_.tail(1);  // Index 35: Velocity Command Angular
    }

    Eigen::VectorXd getVelocityCommand() {
        return observation_.segment(33, 3);
    }

    void updateJointHistory(const Eigen::VectorXd &desiredJointPosition) {
        varJoint_ = jointPositionErrorHistory_;
        jointPositionErrorHistory_.head((historyLength_ - 1) * nJoints_) = varJoint_.tail(
                (historyLength_ - 1) * nJoints_);
        jointPositionErrorHistory_.tail(nJoints_) = desiredJointPosition - generalizedCoordinate_.tail(nJoints_);

        varJoint_ = jointVelocityHistory_;
        jointVelocityHistory_.head((historyLength_ - 1) * nJoints_) = varJoint_.tail((historyLength_ - 1) * nJoints_);
        jointVelocityHistory_.tail(nJoints_) = generalizedVelocity_.tail(nJoints_);
    }

    void reset(raisim::ArticulatedSystem *&raisim, const Eigen::VectorXd &velocityCommand) {
        jointPositionErrorHistory_.setZero();
        jointVelocityHistory_.setZero();

        updateVelocityCommand(velocityCommand);
        updateObservation(raisim, Eigen::VectorXd::Zero(16), nominalFeetPhase_, jointNominalConfig_);
    }

    raisim::Mat<3, 3> &getBaseRotationVerticalComponent() {
        return baseRotationVerticalComponent_;
    }

    Eigen::VectorXd &getObservation() {
        return observation_;
    }

    Eigen::VectorXd &getAction() {
        return action_;
    }

    Eigen::VectorXd &getFeetPhase() {
        return feetPhase_;
    }

    Eigen::VectorXd &getDesiredJointPosition() {
        return desiredJointPosition_;
    }

    Eigen::VectorXd &getNominalJointConfiguration() {
        return jointNominalConfig_;
    }

    Eigen::VectorXd &getNominalFeetPositions() {
        return feetNominalPositions_;
    }

    Eigen::VectorXd &getJointPositionErrorHistory() {
        return jointPositionErrorHistory_;
    }

    Eigen::VectorXd &getJointVelocityHistory() {
        return jointVelocityHistory_;
    }

    [[nodiscard]] int getObservationDim() const {
        return observationDim_;
    }

    Eigen::VectorXd &getBaseLinearVelocity() {
        return baseLinearVelocity_;
    }

    Eigen::VectorXd &getBaseAngularVelocity() {
        return baseAngularVelocity_;
    }

    Eigen::VectorXd &getGeneralizedCoordinate() {
        return generalizedCoordinate_;
    }

    Eigen::VectorXd &getGeneralizedVelocity() {
        return generalizedVelocity_;
    }

    Eigen::VectorXd &getGeneralizedForce() {
        return generalizedForce_;
    }

    raisim::Mat<3, 3> &getBaseRotation() {
        return baseRotation_;
    }

    Eigen::VectorXd &getNominalGeneralizedCoordinates() {
        return nominalGeneralizedCoordinates_;
    }

    Eigen::VectorXd &getDesiredBaseLinearVelocity() {
        return desiredBaseLinearVelocity_;
    }

    Eigen::VectorXd &getDesiredBaseAngularVelocity() {
        return desiredBaseAngularVelocity_;
    }

    static void quaternionToRotationMatrix(const raisim::Vec<4> &q, raisim::Mat<3, 3> &r) {
        r(0, 0) = 2. * (q(0) * q(0) + q(1) * q(1)) - 1.;
        r(0, 1) = 2. * (q(1) * q(2) - q(0) * q(3));
        r(0, 2) = 2. * (q(1) * q(3) + q(0) * q(2));

        r(1, 0) = 2. * (q(1) * q(2) + q(0) * q(3));
        r(1, 1) = 2. * (q(0) * q(0) + q(2) * q(2)) - 1.;
        r(1, 2) = 2. * (q(2) * q(3) - q(0) * q(1));

        r(2, 0) = 2. * (q(1) * q(3) - q(0) * q(2));
        r(2, 1) = 2. * (q(2) * q(3) + q(0) * q(1));
        r(2, 2) = 2. * (q(0) * q(0) + q(3) * q(3)) - 1.;
    }

    static void quaternionToRotationMatrix(const Eigen::Matrix<double, 4, 1> &q, Eigen::Matrix<double, 3, 3> &r) {
        r(0, 0) = 2. * (q(0) * q(0) + q(1) * q(1)) - 1.;
        r(0, 1) = 2. * (q(1) * q(2) - q(0) * q(3));
        r(0, 2) = 2. * (q(1) * q(3) + q(0) * q(2));

        r(1, 0) = 2. * (q(1) * q(2) + q(0) * q(3));
        r(1, 1) = 2. * (q(0) * q(0) + q(2) * q(2)) - 1.;
        r(1, 2) = 2. * (q(2) * q(3) - q(0) * q(1));

        r(2, 0) = 2. * (q(1) * q(3) - q(0) * q(2));
        r(2, 1) = 2. * (q(2) * q(3) + q(0) * q(1));
        r(2, 2) = 2. * (q(0) * q(0) + q(3) * q(3)) - 1.;
    }

    static Eigen::VectorXd vectorSin(const Eigen::VectorXd &v) {
        Eigen::VectorXd t(v);

        for (auto i = 0; i < v.size(); ++i) {
            t[i] = std::sin(v[i]);
        }

        return t;
    }

    static Eigen::VectorXd vectorCos(const Eigen::VectorXd &v) {
        Eigen::VectorXd t(v);

        for (auto i = 0; i < v.size(); ++i) {
            t[i] = std::cos(v[i]);
        }

        return t;
    }
};

#endif //_LFMC_GYM_AVC_OBSERVATION_HPP
