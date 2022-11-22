//
// Created by siddhant on 20/07/22.
//

#ifndef _LFMC_GYM_UTILITY_HPP
#define _LFMC_GYM_UTILITY_HPP

namespace lfmc {
    std::string randomString(size_t length) {
        /// https://stackoverflow.com/a/12468109

        auto randomChar = []() -> char {
            const char characterSet[] =
                    "0123456789"
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "abcdefghijklmnopqrstuvwxyz";
            const size_t max_index = (sizeof(characterSet) - 1);
            return characterSet[std::rand() % max_index];
        };

        std::string str(length, 0);
        std::generate_n(str.begin(), length, randomChar);

        return str;
    }

    inline Eigen::Vector3d rotationMatrixToRPY(const Eigen::Matrix3d &R) {
        Eigen::Vector3d RPY(3);
        RPY(0) = -atan2(-R(2, 1), R(2, 2));

        // asin(x) returns nan if x > 1 || x < -1, this is potentially possible due
        // to floating point precision issues, so we check if this occurs and bound
        // the result to +/- (pi/2) based on the sign of x, *IFF* the input value
        // was finite.  If it was not, do not make it finite, that would hide an
        // error elsewhere.

        RPY(1) = -asin(R(2, 0));

        if (std::isfinite(R(2, 0)) && !std::isfinite(RPY(1))) {
            RPY(1) = copysign(M_PI / 2.0, R(2, 0));
        }

        RPY(2) = -atan2(-R(1, 0), R(0, 0));

        return RPY;
    }

    inline Eigen::Matrix3d rpyToRotationMatrix(const Eigen::Vector3d &RPY) {
        double cx = cos(RPY(0));
        double sx = sin(RPY(0));
        double cy = cos(RPY(1));
        double sy = sin(RPY(1));
        double cz = cos(RPY(2));
        double sz = sin(RPY(2));

        Eigen::Matrix3d R(3, 3);
        R(0, 0) = cz * cy;
        R(0, 1) = -sz * cx + cz * sy * sx;
        R(0, 2) = sz * sx + cz * sy * cx;
        R(1, 0) = sz * cy;
        R(1, 1) = cz * cx + sz * sy * sx;
        R(1, 2) = -cz * sx + sz * sy * cx;
        R(2, 0) = -sy;
        R(2, 1) = cy * sx;
        R(2, 2) = cy * cx;

        return R;
    }

    inline Eigen::Matrix3d rotationMatrixToGravityComponent(const Eigen::Matrix3d &R) {
        Eigen::VectorXd rpy = rotationMatrixToRPY(R);
        rpy.head(2).setZero();
        return rpyToRotationMatrix(rpy);
    }

    inline Eigen::Matrix3d directionVectorToRotationMatrix(const Eigen::Vector3d &directionVector) {
        Eigen::Matrix3d R(3, 3);

        Eigen::Vector3d directionVector3d;
        directionVector3d << directionVector;
        directionVector3d.normalize();

        Eigen::Vector3d zAxis;
        zAxis << 0.0, 0.0, 1.0;

        Eigen::Vector3d xAxis = zAxis.cross(directionVector3d);
        xAxis.normalize();

        Eigen::Vector3d yAxis = directionVector3d.cross(xAxis);
        yAxis.normalize();

        R.row(0) << xAxis;
        R.row(1) << yAxis;
        R.row(2) << directionVector3d;

        return R;
    }

    inline double angleWrapFast(const double &angle) {
        // 2 * M_PI = 6.28318530718
        // 1 / (2 * M_PI) = 0.15915494309
         return angle - 6.28318530718 * std::floor((angle + M_PI) * 0.15915494309);
    }

    inline double angleWrapAccurate(const double &angle) {
        return std::atan2(std::sin(angle), std::cos(angle));
    }
}

#endif //_LFMC_GYM_UTILITY_HPP
