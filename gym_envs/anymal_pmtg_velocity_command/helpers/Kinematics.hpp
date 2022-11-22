//
// Created by siddhant on 31/10/22.
//

//
// Based on the code by Mathieu Geisert (https://fr.linkedin.com/in/mathieu-geisert-b290a38a)
// And by Joonho Lee (https://github.com/leggedrobotics/learning_quadrupedal_locomotion_over_challenging_terrain_supplementary)
//

#ifndef _LFMC_GYM_KINEMATICS_HPP
#define _LFMC_GYM_KINEMATICS_HPP


class InverseKinematics {
private:
    double distThighToShankSquared_;
    double distShankToFootSquared_;

    double offsetKneeJointPosition_;

    double reachMinAligned_, reachMaxAligned_;
    double reachMin_, reachMax_;

    Eigen::Matrix<double, 4, 3> posBaseToHipInBase_;
    Eigen::Matrix<double, 4, 3> posBaseToHipCenterInBase_;

    Eigen::Matrix<double, 4, 3> posHipToThighInHip_;
    Eigen::Matrix<double, 4, 3> posThighToShankInThigh_;
    Eigen::Matrix<double, 4, 3> posShankToFootInShank_;

    Eigen::Matrix<double, 4, 1> offsetHAAToFootY_;
    Eigen::Matrix<double, 4, 1> offsetHFEToFootY_;

    // Utility variables
    Eigen::Matrix<double, 3, 1> targetJointPositionsLimb_;
    Eigen::Matrix<double, 12, 1> targetJointPositions_;

    Eigen::Matrix<double, 3, 1> posHipCenterToFootInBase_;

public:
    InverseKinematics() {
        /// Position from Base (Origin) to Hip represented in Base Frame
        posBaseToHipInBase_.row(0) << 0.3, 0.104, 0.0;  // LF
        posBaseToHipInBase_.row(1) << 0.3, -0.104, 0.0;  // RF
        posBaseToHipInBase_.row(2) << -0.3, 0.104, 0.0;  // LH
        posBaseToHipInBase_.row(3) << -0.3, -0.104, 0.0;  // RH

        /// Position from Hip (Origin) to (center of) Thigh represented in Hip Frame
        posHipToThighInHip_.row(0) << 0.06, 0.08381, 0.;    // LF
        posHipToThighInHip_.row(1) << 0.06, -0.08381, 0.;    // RF
        posHipToThighInHip_.row(2) << -0.06, 0.08381, 0.;    // LH
        posHipToThighInHip_.row(3) << -0.06, -0.08381, 0.;    // RH

        /// Position from Base (Origin) to (center of) Hip represented in Hip Frame
        posBaseToHipCenterInBase_ = posBaseToHipInBase_;

        // Introduce an Offset along x-axis to adjust to the center of hip
        for (auto h = 0; h < 4; ++h) {
            posBaseToHipCenterInBase_.row(h)[0] += posHipToThighInHip_.row(h)[0];
        }

        /// Position from Thigh to Shank represented in Thigh Frame
        posThighToShankInThigh_.row(0) << 0.0, 0.1003, -0.285;    // LF
        posThighToShankInThigh_.row(1) << 0.0, -0.1003, -0.285;    // RF
        posThighToShankInThigh_.row(2) << 0.0, 0.1003, -0.285;    // LH
        posThighToShankInThigh_.row(3) << 0.0, -0.1003, -0.285;    // RH

        /// Position from Shank to Foot represented in Shank Frame
        posShankToFootInShank_.row(0) << 0.08795, -0.01305, -0.31547;    // LF
        posShankToFootInShank_.row(1) << 0.08795, 0.01305, -0.31547;    // RF
        posShankToFootInShank_.row(2) << -0.08795, -0.01305, -0.31547;    // LH
        posShankToFootInShank_.row(3) << -0.08795, 0.01305, -0.31547;    // RH

        /// Deviations along the y-axis from HFE and HAA to each of the Feet
        for (auto f = 0; f < 4; ++f) {
            offsetHFEToFootY_.row(f)[0] = posThighToShankInThigh_.row(f)[1] + posShankToFootInShank_.row(f)[1];
            offsetHAAToFootY_.row(f)[0] = offsetHFEToFootY_.row(f)[0] + posHipToThighInHip_.row(f)[1];
        }

        /// Get the lengths of each links (assume no deviation along y-axis)
        // Distance from Thigh to Shank
        distThighToShankSquared_ =
                std::pow(posThighToShankInThigh_.row(0)[0], 2) + std::pow(posThighToShankInThigh_.row(0)[2], 2);

        // Distance from Shank to Foot
        distShankToFootSquared_ =
                std::pow(posShankToFootInShank_.row(0)[0], 2) + std::pow(posShankToFootInShank_.row(0)[2], 2);

        /// Get the workspace - minimum and maximum reach
        reachMinAligned_ = std::abs(sqrt(distThighToShankSquared_) - std::sqrt(distShankToFootSquared_)) + 0.1;
        reachMaxAligned_ = std::sqrt(distThighToShankSquared_) + std::sqrt(distShankToFootSquared_) - 0.05;

        reachMin_ = std::sqrt(offsetHAAToFootY_[0] * offsetHAAToFootY_[0] + reachMinAligned_ * reachMinAligned_);
        reachMax_ = std::sqrt(offsetHAAToFootY_[0] * offsetHAAToFootY_[0] + reachMaxAligned_ * reachMaxAligned_);

        /// Knee joint position offset
        offsetKneeJointPosition_ =
                std::abs(std::atan(posShankToFootInShank_.row(0)[0] / posShankToFootInShank_.row(0)[2]));
    }

    ~InverseKinematics() = default;

    const Eigen::Matrix<double, 12, 1> &mapFeetPositionsToJointPositions(
            const Eigen::Matrix<double, 12, 1> &feetPositions) {
        for (auto f = 0; f < 4; ++f) {
            sagittalInverseKinematics(targetJointPositionsLimb_, feetPositions.segment(3 * f, 3), f);
            targetJointPositions_.segment(3 * f, 3) = targetJointPositionsLimb_;
        }

        return targetJointPositions_;
    }

private:
    inline bool sagittalInverseKinematics(
            Eigen::Matrix<double, 3, 1> &targetJointPositionsLimb,
            const Eigen::Matrix<double, 3, 1> &desiredPosBaseToFootInBaseFrame,
            long limb) {
        /// Position from (center of) Hip to Foot represented in Base Frame
        posHipCenterToFootInBase_ = desiredPosBaseToFootInBaseFrame -
                                    Eigen::Matrix<double, 3, 1>(posBaseToHipCenterInBase_.row(limb));

        /// Ensure target is within the workspace
        double reach = posHipCenterToFootInBase_.norm();

        if (reach > reachMax_) {
            posHipCenterToFootInBase_ /= reach;
            posHipCenterToFootInBase_ *= reachMax_;
        } else if (reach < reachMin_) {
            posHipCenterToFootInBase_ /= reach;
            posHipCenterToFootInBase_ *= reachMin_;
        }

        double distHAAToFootY = offsetHAAToFootY_[limb];
        double distHAAToFootYSquared = distHAAToFootY * distHAAToFootY;

        double posYZSquared = posHipCenterToFootInBase_.tail(2).squaredNorm();

        if (posYZSquared < distHAAToFootYSquared) {
            posHipCenterToFootInBase_.tail(2) /= std::sqrt(posYZSquared);
            posHipCenterToFootInBase_.tail(2) *= (std::abs(distHAAToFootY) + 0.01);

            if (posHipCenterToFootInBase_[0] > reachMaxAligned_) {
                posHipCenterToFootInBase_[0] /= std::abs(posHipCenterToFootInBase_[0]);
                posHipCenterToFootInBase_[0] *= reachMaxAligned_;
            }

            posYZSquared = posHipCenterToFootInBase_.tail(2).squaredNorm();
        }

        /// Compute HAA position
        double rSquared = posYZSquared - distHAAToFootYSquared;
        double r = std::sqrt(rSquared);
        double delta = std::atan2(posHipCenterToFootInBase_.y(), -posHipCenterToFootInBase_.z());

        double beta = std::atan2(r, distHAAToFootY);
        targetJointPositionsLimb[0] = beta + delta - M_PI_2;

        /// Simplification for ANYmal C - Compute KFE position
        double lSquared = (rSquared + posHipCenterToFootInBase_[0] * posHipCenterToFootInBase_[0]);
        double phi1 = std::acos((distThighToShankSquared_ + lSquared - distShankToFootSquared_) * 0.5 /
                                (sqrt(distThighToShankSquared_ * lSquared)));
        double phi2 = std::acos((distShankToFootSquared_ + lSquared - distThighToShankSquared_) * 0.5 /
                                (sqrt(distShankToFootSquared_ * lSquared)));

        targetJointPositionsLimb[2] = phi1 + phi2 - offsetKneeJointPosition_;

        if (limb < 2) {
            targetJointPositionsLimb[2] *= -1.0;
        }

        /// Compute HFE position
        double thetaPrime = atan2(posHipCenterToFootInBase_[0], r);

        if (limb > 1) {
            targetJointPositionsLimb[1] = -phi1 - thetaPrime;
        } else {
            targetJointPositionsLimb[1] = phi1 - thetaPrime;
        }
        return true;
    }
};

#endif //_LFMC_GYM_KINEMATICS_HPP
