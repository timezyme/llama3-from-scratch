// Converters for reduced-precision floating-point formats (BF16, FP16) to FP32.
// Used by the weight loader to decode model dumps stored in half-precision.

#pragma once
#include <cstdint>
#include <cstring>

// BF16 (bfloat16) -> FP32.
// BF16 shares FP32's 8-bit exponent, so conversion is just a 16-bit left shift
// to place the bits into the upper half of a 32-bit float.
static inline float bf16_to_float(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    float out;
    std::memcpy(&out, &u, sizeof(out)); // type-pun via memcpy (alias-safe)
    return out;
}

// FP16 (IEEE 754 binary16) -> FP32.
// Handles all cases: normals, subnormals, zero, infinity, and NaN.
// Rebases the 5-bit exponent (bias 15) to the FP32 8-bit exponent (bias 127).
static inline float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16; // sign bit -> bit 31
    uint32_t expu = (uint32_t)(h & 0x7C00u) >> 10;  // 5-bit exponent
    uint32_t mant = (uint32_t)(h & 0x03FFu);         // 10-bit mantissa

    uint32_t f;
    if (expu == 0) {
        if (mant == 0) {
            f = sign; // +/- zero
        } else {
            // Subnormal FP16: normalize by shifting mantissa until the
            // implicit leading 1 appears, adjusting the exponent accordingly.
            int32_t exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FFu; // strip the now-implicit leading 1

            uint32_t exp_f = (uint32_t)(exp + (127 - 15)); // rebias exponent
            uint32_t mant_f = mant << 13;                   // 10-bit -> 23-bit mantissa
            f = sign | (exp_f << 23) | mant_f;
        }
    } else if (expu == 0x1F) {
        // Infinity or NaN: set FP32 exponent to all-ones, preserve mantissa.
        f = sign | 0x7F800000u | (mant << 13);

    } else {
        // Normal value: rebias exponent and widen mantissa.
        uint32_t exp_f = expu + (127 - 15);
        uint32_t mant_f = mant << 13;
        f = sign | (exp_f << 23) | mant_f;
    }

    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}
