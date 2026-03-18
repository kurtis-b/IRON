// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {
// Define a utility function to convert an f32 buffer to bf16 using the AIE API
// in a way that preserves the number of elements (which means the total size
// of the input in bits gets reduced by half)
void convert_copy_f32_to_bf16(float *restrict input_vector, bfloat16 *restrict output_vector, const int32_t vector_size)
{
    event0();
    aie::accum<accfloat, 16> acc;
    for (unsigned i = 0; i < vector_size; i += 16) {
        aie::vector<float, 16> input = aie::load_v<16>(input_vector + i);
        acc.from_vector(input, 0);
        aie::store_v(output_vector + i, acc.to_vector<bfloat16>());
    }
    event1();
    return;
}

void convert_copy_bf16_to_f32(bfloat16 *restrict input_vector, float *restrict output_vector, const int32_t vector_size)
{
    event0();
    aie::accum<accfloat, 16> acc;
    for (unsigned i = 0; i < vector_size; i += 16) {
        aie::vector<bfloat16, 16> input = aie::load_v<16>(input_vector + i);
        acc.from_vector(input, 0);
        aie::store_v(output_vector + i, acc.to_vector<float>());
    }
    event1();
    return;
}

void pack_stats_f32_to_bf16_packet(float *restrict sum_vector,
                                   float *restrict sumsq_vector,
                                   bfloat16 *restrict output_packet,
                                   const int32_t vector_size)
{
    event0();
    aie::accum<accfloat, 16> acc;
    for (unsigned i = 0; i < vector_size; i += 16) {
        aie::vector<float, 16> sum = aie::load_v<16>(sum_vector + i);
        acc.from_vector(sum, 0);
        aie::store_v(output_packet + i, acc.to_vector<bfloat16>());
    }
    for (unsigned i = 0; i < vector_size; i += 16) {
        aie::vector<float, 16> sumsq = aie::load_v<16>(sumsq_vector + i);
        acc.from_vector(sumsq, 0);
        aie::store_v(output_packet + vector_size + i, acc.to_vector<bfloat16>());
    }
    event1();
    return;
}

void unpack_stats_bf16_packet_to_f32(bfloat16 *restrict input_packet,
                                     float *restrict sum_vector,
                                     float *restrict sumsq_vector,
                                     const int32_t vector_size)
{
    event0();
    aie::accum<accfloat, 16> acc;
    for (unsigned i = 0; i < vector_size; i += 16) {
        aie::vector<bfloat16, 16> sum = aie::load_v<16>(input_packet + i);
        acc.from_vector(sum, 0);
        aie::store_v(sum_vector + i, acc.to_vector<float>());
    }
    for (unsigned i = 0; i < vector_size; i += 16) {
        aie::vector<bfloat16, 16> sumsq = aie::load_v<16>(input_packet + vector_size + i);
        acc.from_vector(sumsq, 0);
        aie::store_v(sumsq_vector + i, acc.to_vector<float>());
    }
    event1();
    return;
}

} // extern "C"
