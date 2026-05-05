// Operator base class scaffold. Not used by the current inference
// pipeline — every kernel is invoked through a free function in
// kernels.cuh, which is simpler and faster to read for a fresh
// reviewer than virtual dispatch. Kept here as the starting point for
// future operator encapsulation (e.g. when adding pluggable
// quantization or sampling backends). All methods throw so any
// accidental use surfaces immediately.

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

