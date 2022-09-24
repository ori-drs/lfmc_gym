#ifndef _OKAPI_GYM_VISUALIZATION_HPP
#define _OKAPI_GYM_VISUALIZATION_HPP

#include "Utility.hpp"


class VisualizationHandler {
private:
    bool visualizable_;

    raisim::Vec<3> baseRPY_{};
    raisim::Mat<3, 3> baseRotationHeadingComponent_{};
    raisim::Vec<4> baseQuaternionHeadingComponent_{};
    raisim::Vec<3> headingVisualOffset_{}, lateralVisualOffset_{};
    raisim::Vec<3> basePosition_{};

public:
    explicit VisualizationHandler(const bool &visualizable) {
        visualizable_ = visualizable;
        baseRotationHeadingComponent_.setIdentity();
    }

    void setServer(const std::unique_ptr<raisim::RaisimServer> &server) const {
        if (!visualizable_) return;
        server->addVisualCylinder("heading_velocity", 0.0025, 0.0025, 1., 0., 0., 1.);
        server->addVisualCylinder("lateral_velocity", 0.0025, 0.0025, 0., 1., 0., 1.);
        server->addVisualCylinder("yaw_rate", 0.0025, 0.0025, 0., 0., 1., 1.);
    }
    
    void updateVelocityVisual(raisim::ArticulatedSystem *&robot,
                              ObservationHandler &observationHandler,
                              const std::unique_ptr<raisim::RaisimServer> &server) {
        if (!visualizable_) return;

        baseRPY_.e() = okapi::rotationMatrixToRPY(robot->getBaseOrientation().e());

        /// Heading velocity visual
        baseRPY_[0] = 0.;
        baseRPY_[1] = M_PI_2;
        baseRotationHeadingComponent_.e() = okapi::rpyToRotationMatrix(baseRPY_.e());
        raisim::rotMatToQuat(baseRotationHeadingComponent_, baseQuaternionHeadingComponent_);
        server->getVisualObject("heading_velocity")->setOrientation(baseQuaternionHeadingComponent_.e());

        headingVisualOffset_ = robot->getBaseOrientation() *
                               raisim::Vec<3>{observationHandler.getDesiredBaseLinearVelocity()[0] * 0.25, 0., 0.};
        robot->getBasePosition(basePosition_);
        basePosition_[0] += headingVisualOffset_[0];
        basePosition_[1] += headingVisualOffset_[1];
        basePosition_[2] += 0.35;
        server->getVisualObject("heading_velocity")->setPosition(basePosition_.e());
        server->getVisualObject("heading_velocity")->setCylinderSize(
                0.02, softSign(std::abs(observationHandler.getDesiredBaseLinearVelocity()[0])) * 1.5);

        /// Lateral velocity visual
        baseRPY_[0] = M_PI_2;
        baseRPY_[1] = 0.;
        baseRotationHeadingComponent_.e() = okapi::rpyToRotationMatrix(baseRPY_.e());
        raisim::rotMatToQuat(baseRotationHeadingComponent_, baseQuaternionHeadingComponent_);
        server->getVisualObject("lateral_velocity")->setOrientation(baseQuaternionHeadingComponent_.e());

        lateralVisualOffset_ = robot->getBaseOrientation() *
                               raisim::Vec<3>{0., observationHandler.getDesiredBaseLinearVelocity()[1] * 0.25, 0.};
        robot->getBasePosition(basePosition_);
        basePosition_[0] += lateralVisualOffset_[0];
        basePosition_[1] += lateralVisualOffset_[1];
        basePosition_[2] += 0.35;
        server->getVisualObject("lateral_velocity")->setPosition(basePosition_.e());
        server->getVisualObject("lateral_velocity")->setCylinderSize(
                0.02, softSign(std::abs(observationHandler.getDesiredBaseLinearVelocity()[1])) * 1.5);

        /// Yaw rate visual
        robot->getBasePosition(basePosition_);
        basePosition_[2] += 0.35;
        server->getVisualObject("yaw_rate")->setPosition(basePosition_.e());
        server->getVisualObject("yaw_rate")->setCylinderSize(
                0.2 * softSign(std::abs(observationHandler.getDesiredBaseAngularVelocity()[2])), 0.05);

        if (observationHandler.getDesiredBaseAngularVelocity()[2] < 0) {
            server->getVisualObject("yaw_rate")->setColor(0., 1., 1., 1.);
        } else {
            server->getVisualObject("yaw_rate")->setColor(1., 0., 1., 1.);
        }
    }

    static double softSign(const double &input) {
        return input / (std::abs(input) + 1.);
    }

    static Eigen::MatrixXd softSign(const Eigen::MatrixXd &input) {
        return input.array() / (input.array().abs() + 1.);
    }
};


#endif //_OKAPI_GYM_VISUALIZATION_HPP
