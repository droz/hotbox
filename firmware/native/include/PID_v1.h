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
          disp_kp_(kp), disp_ki_(ki), disp_kd_(kd), direction_(direction) {
        SetTunings(kp, ki, kd);
    }

    void SetMode(int mode) { mode_ = mode; }

    void SetTunings(double kp, double ki, double kd) {
        if (kp < 0.0 || ki < 0.0 || kd < 0.0) return;
        disp_kp_ = kp;
        disp_ki_ = ki;
        disp_kd_ = kd;
        const double sample_time_s = sample_time_ms_ / 1000.0;
        kp_ = kp;
        ki_ = ki * sample_time_s;
        kd_ = kd / sample_time_s;
        if (direction_ == REVERSE) {
            kp_ = -kp_;
            ki_ = -ki_;
            kd_ = -kd_;
        }
    }

    void SetSampleTime(int sample_time_ms) {
        if (sample_time_ms <= 0) return;
        const double ratio = static_cast<double>(sample_time_ms) / static_cast<double>(sample_time_ms_);
        ki_ *= ratio;
        kd_ /= ratio;
        sample_time_ms_ = sample_time_ms;
    }

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
    double kp_ = 0.0, ki_ = 0.0, kd_ = 0.0;
    double disp_kp_ = 0.0, disp_ki_ = 0.0, disp_kd_ = 0.0;
    int direction_;
    int mode_ = MANUAL;
    int sample_time_ms_ = 100;
    double out_min_ = -255.0, out_max_ = 255.0;
    double i_term_ = 0.0;
    double last_input_ = 0.0;
};
