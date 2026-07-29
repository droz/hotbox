#pragma once
// Minimal drop-in of the Arduino PID_v1 library for native compilation.
// Implements the same discrete-time position PID as the real library.

constexpr int DIRECT  = 0;
constexpr int REVERSE = 1;
constexpr int AUTOMATIC = 1;
constexpr int MANUAL    = 0;

class PID {
public:
    PID(double* input, double* output, double* setpoint,
        double kp, double ki, double kd, int direction)
        : input_(input), output_(output), setpoint_(setpoint),
          kp_(kp), ki_(ki), kd_(kd), direction_(direction) {}

    void SetMode(int mode) { mode_ = mode; }

    void SetOutputLimits(double min_out, double max_out) {
        out_min_ = min_out;
        out_max_ = max_out;
        // Clamp integrator to output range to match PID_v1 behaviour.
        if (i_term_ > out_max_) i_term_ = out_max_;
        if (i_term_ < out_min_) i_term_ = out_min_;
    }

    // Returns true when output was updated.
    bool Compute() {
        if (mode_ != AUTOMATIC) return false;
        double error = *setpoint_ - *input_;
        if (direction_ == REVERSE) error = -error;

        i_term_ += ki_ * error;
        if (i_term_ > out_max_) i_term_ = out_max_;
        if (i_term_ < out_min_) i_term_ = out_min_;

        double d_input = *input_ - last_input_;
        *output_ = kp_ * error + i_term_ - kd_ * d_input;
        if (*output_ > out_max_) *output_ = out_max_;
        if (*output_ < out_min_) *output_ = out_min_;

        last_input_ = *input_;
        return true;
    }

private:
    double* input_;
    double* output_;
    double* setpoint_;
    double kp_, ki_, kd_;
    int direction_;
    int mode_ = MANUAL;
    double out_min_ = -255.0, out_max_ = 255.0;
    double i_term_ = 0.0;
    double last_input_ = 0.0;
};
