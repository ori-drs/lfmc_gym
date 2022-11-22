//
// Created by Siddhant on 31/10/22.
//

//
// Based on the code by Mathieu Geisert (https://fr.linkedin.com/in/mathieu-geisert-b290a38a)
// And by Joonho Lee (https://github.com/leggedrobotics/learning_quadrupedal_locomotion_over_challenging_terrain_supplementary)
//

#ifndef _LFMC_GYM_MOTION_HPP
#define _LFMC_GYM_MOTION_HPP


class MotionGenerator {
private:
    double baseFrequency_, controlStepTime_;
    Eigen::Matrix<double, 4, 1> feetClearance_, feetPhase_, feetOscillatorFrequency_;

    // Utility variables
    Eigen::VectorXd nominalFeetPosition_;
    Eigen::Matrix<double, 12, 1> feetPositionTarget_;
    double desiredHeight_ = 0., phaseResidual_ = 0.;
    double currentTrajectoryStep_ = 0., currentTrajectoryStepSquared_ = 0., currentTrajectoryStepCubed_ = 0.;

public:
    MotionGenerator(const double &baseFrequency, const double &feetClearance,
                    const double &controlStepTime, const Eigen::VectorXd &nominalFeetPosition) {
        baseFrequency_ = baseFrequency;
        controlStepTime_ = controlStepTime;

        feetClearance_.setConstant(feetClearance);

        nominalFeetPosition_ = nominalFeetPosition;

        reset();
    }

    ~MotionGenerator() = default;

    void reset() {
        // Right Front and Left Hind
        feetPhase_[1] = feetPhase_[2] = 0.;

        // Left Front and Right Hind
        feetPhase_[0] = feetPhase_[3] = -M_PI;

        // Oscillator frequency
        feetOscillatorFrequency_.setConstant(baseFrequency_);

        // Utility variables
        feetPositionTarget_ = nominalFeetPosition_;

        desiredHeight_ = 0.;
        currentTrajectoryStep_ = 0.;
    }

    const Eigen::Matrix<double, 12, 1> &advance(const Eigen::Matrix<double, 4, 1> &deltaOscillatorFrequency,
                                                const Eigen::Matrix<double, 3, 1> &gravityAxis,
                                                bool zeroCommand = false) {
        feetPositionTarget_ = nominalFeetPosition_;

        for (auto j = 0; j < 4; ++j) {
            // Policy outputs deviation to the oscillator frequency
            feetOscillatorFrequency_[j] = deltaOscillatorFrequency[j] + baseFrequency_;

            // phase_{t} = phase_{t-1} + 2 * pi * oscillator_frequency * delta_6
            phaseResidual_ = 6.28318530718 * feetOscillatorFrequency_[j] * controlStepTime_;
            feetPhase_[j] += phaseResidual_; // 2 * M_PI = 6.28318530718

            // Wrap angles between [-M_PI, M_PI)
            feetPhase_[j] = lfmc::angleWrapAccurate(feetPhase_[j]);

            // Force stopping for zero velocity command
            if (zeroCommand) {
                if (feetPhase_[j] <= phaseResidual_) {
                    feetPhase_[j] = feetPhase_[j] < (-M_PI_4 + phaseResidual_) ? -M_PI_2 : 0.;
                }
            }

            desiredHeight_ = 0.0;

            // If phase_{t} < 0, consider the foot to be in stance, else swing.
            if (feetPhase_[j] > 0.0) {
                currentTrajectoryStep_ = feetPhase_[j] / M_PI_2;

                if (currentTrajectoryStep_ < 1.0) {
                    currentTrajectoryStepSquared_ = currentTrajectoryStep_ * currentTrajectoryStep_;
                    currentTrajectoryStepCubed_ = currentTrajectoryStepSquared_ * currentTrajectoryStep_;

                    // Lifting foot up
                    // delta_height = (-2 * t^3) + (3 * t^2)
                    desiredHeight_ = (-2 * currentTrajectoryStepCubed_ + 3 * currentTrajectoryStepSquared_);
                } else {
                    currentTrajectoryStep_ = currentTrajectoryStep_ - 1.;

                    currentTrajectoryStepSquared_ = currentTrajectoryStep_ * currentTrajectoryStep_;
                    currentTrajectoryStepCubed_ = currentTrajectoryStepSquared_ * currentTrajectoryStep_;

                    // Putting foot down
                    // delta_height = (2 * t^3) - (3 * t^2) + 1
                    desiredHeight_ = (2 * currentTrajectoryStepCubed_ - 3 * currentTrajectoryStepSquared_ + 1.0);
                }

                // Scale desired foot height using the nominal foot clearance
                desiredHeight_ *= feetClearance_[j];
            }

            // Given the desired foot height and gravity axis, get the desired foot position target
            feetPositionTarget_.col(0)[j * 3 + 2] = 0.;
            feetPositionTarget_.segment(j * 3, 3) += gravityAxis * (nominalFeetPosition_[3 * j + 2] + desiredHeight_);
        }

        return feetPositionTarget_;
    }

    const Eigen::Matrix<double, 4, 1> &getFeetPhase() {
        return feetPhase_;
    }

};

#endif //_LFMC_GYM_MOTION_HPP
