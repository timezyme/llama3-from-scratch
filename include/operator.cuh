// Abstract base class for GPU operators (e.g. attention, FFN, norm).
// Every method throws at runtime to force derived classes to provide
// real implementations. Not currently used by the inference pipeline
// (operators are called directly via free functions in kernels.cuh),
// but kept as a scaffold for future operator encapsulation.

#pragma once

#include "config.h"
#include "prelude.h"

class AbstractOperator {
  public:
    ~AbstractOperator() {
        throw runtime_error("AbstractOperator::~AbstractOperator not overridden");
    }

    AbstractOperator() {
        throw runtime_error("AbstractOperator must be subclassed");
    }

    // Transfer weights from host to device memory.
    bool to_gpu() const {
        throw runtime_error("AbstractOperator::to_gpu() not overridden");
    }

    // Execute the operator's forward pass (launch kernel).
    bool call() {
        throw runtime_error("AbstractOperator::call() not overridden");
    }

    // Check whether weights are currently on the GPU.
    bool is_in_gpu() {
        throw runtime_error("AbstractOperator::is_in_gpu() not overridden");
    }

  private:
    bool in_gpu = false;
};

