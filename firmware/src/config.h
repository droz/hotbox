#pragma once

#include <Arduino.h>

namespace hotbox {

constexpr int kCanTxPin = D10;
constexpr int kCanRxPin = D9;

constexpr int kVertMotorP = A0;
constexpr int kVertMotorM = A1;
constexpr int kVertEncA = D2;
constexpr int kVertEncB = D3;
constexpr int kVertHall = D4;

constexpr int kHorizMotorP = A2;
constexpr int kHorizMotorM = A3;
constexpr int kHorizEncA = D5;
constexpr int kHorizEncB = D6;
constexpr int kHorizHall = D7;

#ifndef HOTBOX_NODE_ID
#define HOTBOX_NODE_ID 0
#endif

constexpr float kControlPeriodS = 0.02f;
constexpr float kMaxVelocityDegS = 30.0f;
constexpr float kMaxAccelDegS2 = 120.0f;
constexpr float kHomingVelocityDegS = 5.0f;
constexpr float kTicksPerDegree = 8.0f;

}  // namespace hotbox
