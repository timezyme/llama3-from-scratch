// Abstract base class for GPU operators (e.g. attention, FFN, norm).
// This is a stub — every method throws at runtime to enforce that
// derived classes provide real implementations for weight transfer
// and kernel dispatch.

#pragma once

#include "config.h"
#include "prelude.h"

class AbstractOperator {
  public:
    ~AbstractOperator() {
        throw runtime_error("Destructor not implemented, beware of "
                            "weights!, fallback to derived class");
    }

    AbstractOperator() {
        throw runtime_error(
            "Constructor not implemented, fallback to derived class");
    }

    // Transfer weights from host to device memory.
    bool to_gpu() const {
        throw runtime_error(
            "to_gpu() not implemented, fallback to derived class");
    }

    // Execute the operator's forward pass (launch kernel).
    bool call() {
        throw runtime_error(
            "call() not implemented, fallback to derived class");
    }

    // Check whether weights are currently on the GPU.
    bool is_in_gpu() {
        throw runtime_error(
            "is_in_gpu() not implemented, fallback to derived class");
    }

  private:
    bool in_gpu = false; // tracks host/device residency
};

