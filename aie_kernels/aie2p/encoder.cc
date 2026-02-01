// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "../aie_kernel_utils.h"
#include "../generic/add.cc"
#include "gelu.cc"
#include "zero.cc"

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Blocked MatMul kernel (vectorized) utilizing the aie::mmul class.
 * The matrices are assumed to be pre-tiled with the following shapes
 * for the aie:mmul class: A => rxs, B => sxt, C => rxt.
 *
 * The matrix dimensions of the kernel are defined by rowA, colA and colB.
 * In this particular kernel we expand the aie::mmul two times in each
 * input matrices A (in 'm' dimension, or rowA) and B (in 'n' dimension, or
 * ColB), leading to a 1x1 expansion in output matrix C (see C00, C01, C02, C03
 * below). This expansion helps with accumulator registers usage, which leads in
 * attaining high kernel efficiency (SIMD utilization).
 *
 * Data within each tile (rxs, sxt and rxt) are assumed to be in row-major
 * order. Also, the entire tiles themselves are stored in row-major order, as
 * shown in the example below for matrix A:
 *
 *      <-s->
 *    _  ________________________
 * 	  r |  1 |  2 |  3 | ...
 * 	  _ |____|____|____|
 * 	    |  x | x+1| x+2| ...
 * 	    |____|____|____|
 * 	    |.
 * 	    |.
 * 	    |.
 *
 * A simplified example of this kernel can be found in the AIE-API
 * documentation: https://xilinx.github.io/aie_api/group__group__mmul.html
 */
template <typename T_in,
          typename T_out,
          unsigned rowA,
          unsigned colA,
          unsigned colB,
          unsigned r,
          unsigned s,
          unsigned t>
void matmul_vectorized_1x1_mmul(const T_in *__restrict pHalfA1,
                                const T_in *__restrict pHalfA2,
                                const T_in *__restrict pB,
                                T_out *__restrict pC)
{
    // Don't change functionality with debug mode since the weight matrix is an identity matrix
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;
    const int szA = MMUL::size_A / 2; // The tiles have half of the rows expected for mmul api

    event0();

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (unsigned z = 0; z < rowA; z += 1) {

        T_out *__restrict pC1 = pC + (z * colB) * MMUL::size_C;

        for (unsigned j = 0; j < colB; j += 1)
#ifdef OPT_PERF_ENABLED
            AIE_LOOP_FLATTEN
#endif
            {

                const T_in *__restrict pHA1 = pHalfA1 + (z * colA) * szA;
                const T_in *__restrict pHA2 = pHalfA2 + (z * colA) * szA;
                const T_in *__restrict pB1 = pB + (j)*MMUL::size_B;
                aie::vector<T_in, szA> HA10;
                aie::vector<T_in, szA> HA20;
                aie::vector<T_in, MMUL::size_B> B0;

                // Load partial results from C buffer for accumulation in-place. The
                // zero.cc function handles the zeroing of data when a new
                // accumulation is needed (after the 'K' reduction dimension)
                aie::vector<T_out, MMUL::size_C> acc_C00;
                acc_C00 = aie::load_v<MMUL::size_C>(pC1);

                MMUL C00(acc_C00);

                for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
                    AIE_LOOP_FLATTEN
#endif
                    {
                        HA10 = aie::load_v<szA>(pHA1);
                        pHA1 += szA;
                        HA20 = aie::load_v<szA>(pHA2);
                        pHA2 += szA;
                        auto A0 = ::aie::concat(HA10, HA20); // Expects HA10 to be the first rows and HA20 the next rows
                        B0 = aie::load_v<MMUL::size_B>(pB1);
                        pB1 += MMUL::size_B * colB;

                        C00.mac(A0, B0);
                    }
                aie::store_v(pC1, C00.template to_vector<T_out>());
                pC1 += MMUL::size_C;
            }
    }

    event1();
}

/* Blocked MatMul kernel (vectorized) utilizing the aie::mmul class.
 * The matrices are assumed to be pre-tiled with the following shapes
 * for the aie:mmul class: A => rxs, B => sxt, C => rxt.
 *
 * The matrix dimensions of the kernel are defined by rowA, colA and colB.
 * In this particular kernel we expand the aie::mmul two times in each
 * input matrices A (in 'm' dimension, or rowA) and B (in 'n' dimension, or
 * ColB), leading to a 1x1 expansion in output matrix C (see C00, C01, C02, C03
 * below). This expansion helps with accumulator registers usage, which leads in
 * attaining high kernel efficiency (SIMD utilization).
 *
 * Data within each tile (rxs, sxt and rxt) are assumed to be in row-major
 * order. Also, the entire tiles themselves are stored in row-major order, as
 * shown in the example below for matrix A:
 *
 *      <-s->
 *    _  ________________________
 * 	  r |  1 |  2 |  3 | ...
 * 	  _ |____|____|____|
 * 	    |  x | x+1| x+2| ...
 * 	    |____|____|____|
 * 	    |.
 * 	    |.
 * 	    |.
 *
 * A simplified example of this kernel can be found in the AIE-API
 * documentation: https://xilinx.github.io/aie_api/group__group__mmul.html
 */
template <typename T_in,
          typename T_out,
          unsigned rowA,
          unsigned colA,
          unsigned colB,
          unsigned r,
          unsigned s,
          unsigned t>
void matmul_with_acc_vectorized_1x1_mmul(const T_in *__restrict pA,
                                         const T_in *__restrict pB,
                                         T_out *__restrict pAcc,
                                         T_out *__restrict pC)
{
    // Don't change functionality with debug mode since the weight matrix is an identity matrix
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

    event0();

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (unsigned z = 0; z < rowA; z += 1) {

        T_out *__restrict pAcc1 = pAcc + (z * colB) * MMUL::size_C;
        T_out *__restrict pC1 = pC + (z * colB) * MMUL::size_C;

        for (unsigned j = 0; j < colB; j += 1)
#ifdef OPT_PERF_ENABLED
            AIE_LOOP_FLATTEN
#endif
            {

                const T_in *__restrict pA1 = pA + (z * colA) * MMUL::size_A;
                const T_in *__restrict pB1 = pB + (j)*MMUL::size_B;
                aie::vector<T_in, MMUL::size_A> A0;
                aie::vector<T_in, MMUL::size_B> B0;

                // Load partial results from C buffer for accumulation in-place. The
                // zero.cc function handles the zeroing of data when a new
                // accumulation is needed (after the 'K' reduction dimension)
                aie::vector<T_out, MMUL::size_C> acc_C00;
                acc_C00 = aie::load_v<MMUL::size_C>(pAcc1);
                pAcc1 += MMUL::size_C;

                MMUL C00(acc_C00);

                for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
                    AIE_LOOP_FLATTEN
#endif
                    {
                        A0 = aie::load_v<MMUL::size_A>(pA1);
                        pA1 += MMUL::size_A;
                        B0 = aie::load_v<MMUL::size_B>(pB1);
                        pB1 += MMUL::size_B * colB;

                        C00.mac(A0, B0);
                    }
                aie::store_v(pC1, C00.template to_vector<T_out>());
                pC1 += MMUL::size_C;
            }
    }

    event1();
}

/* Blocked MatMul kernel (vectorized) utilizing the aie::mmul class.
 * The matrices are assumed to be pre-tiled with the following shapes
 * for the aie:mmul class: A => rxs, B => sxt, C => rxt.
 *
 * The matrix dimensions of the kernel are defined by rowA, colA and colB.
 * In this particular kernel we expand the aie::mmul two times in each
 * input matrices A (in 'm' dimension, or rowA) and B (in 'n' dimension, or
 * ColB), leading to a 1x4 expansion in output matrix C (see C00, C01, C02, C03
 * below). This expansion helps with accumulator registers usage, which leads in
 * attaining high kernel efficiency (SIMD utilization).
 *
 * Data within each tile (rxs, sxt and rxt) are assumed to be in row-major
 * order. Also, the entire tiles themselves are stored in row-major order, as
 * shown in the example below for matrix A:
 *
 *      <-s->
 *    _  ________________________
 * 	  r |  1 |  2 |  3 | ...
 * 	  _ |____|____|____|
 * 	    |  x | x+1| x+2| ...
 * 	    |____|____|____|
 * 	    |.
 * 	    |.
 * 	    |.
 *
 * A simplified example of this kernel can be found in the AIE-API
 * documentation: https://xilinx.github.io/aie_api/group__group__mmul.html
 */
template <typename T_in,
          typename T_out,
          unsigned rowA,
          unsigned colA,
          unsigned colB,
          unsigned r,
          unsigned s,
          unsigned t>
void matmul_vectorized_1x4_mmul(const T_in *__restrict pHalfA1,
                                const T_in *__restrict pHalfA2,
                                const T_in *__restrict pB,
                                T_out *__restrict pC)
{
    // Don't change functionality with debug mode since the weight matrix is an identity matrix
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;
    const int szA = MMUL::size_A / 2; // The tiles have half of the rows expected for mmul api

    event0();

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (unsigned z = 0; z < rowA; z += 1) {

        T_out *__restrict pC1 = pC + (z * colB) * MMUL::size_C;

        for (unsigned j = 0; j < colB; j += 4)
#ifdef OPT_PERF_ENABLED
            AIE_LOOP_FLATTEN
#endif
            {

                const T_in *__restrict pHA1 = pHalfA1 + (z * colA) * szA;
                const T_in *__restrict pHA2 = pHalfA2 + (z * colA) * szA;
                const T_in *__restrict pB1 = pB + (j)*MMUL::size_B;
                const T_in *__restrict pB2 = pB + (j + 1) * MMUL::size_B;
                const T_in *__restrict pB3 = pB + (j + 2) * MMUL::size_B;
                const T_in *__restrict pB4 = pB + (j + 3) * MMUL::size_B;
                aie::vector<T_in, szA> HA10;
                aie::vector<T_in, szA> HA20;
                aie::vector<T_in, MMUL::size_B> B0;
                aie::vector<T_in, MMUL::size_B> B1;
                aie::vector<T_in, MMUL::size_B> B2;
                aie::vector<T_in, MMUL::size_B> B3;

                // Load partial results from C buffer for accumulation in-place. The
                // zero.cc function handles the zeroing of data when a new
                // accumulation is needed (after the 'K' reduction dimension)
                aie::vector<T_out, MMUL::size_C> acc_C00 = aie::load_v<MMUL::size_C>(pC1);
                aie::vector<T_out, MMUL::size_C> acc_C01 = aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
                aie::vector<T_out, MMUL::size_C> acc_C02 = aie::load_v<MMUL::size_C>(pC1 + 2 * MMUL::size_C);
                aie::vector<T_out, MMUL::size_C> acc_C03 = aie::load_v<MMUL::size_C>(pC1 + 3 * MMUL::size_C);

                MMUL C00(acc_C00);
                MMUL C01(acc_C01);
                MMUL C02(acc_C02);
                MMUL C03(acc_C03);

                for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
                    AIE_LOOP_FLATTEN
#endif
                    {
                        HA10 = aie::load_v<szA>(pHA1);
                        pHA1 += szA;
                        HA20 = aie::load_v<szA>(pHA2);
                        pHA2 += szA;
                        auto A0 = ::aie::concat(HA10, HA20); // Expects HA10 to be the first rows and HA20 the next rows
                        B0 = aie::load_v<MMUL::size_B>(pB1);
                        pB1 += MMUL::size_B * colB;
                        B1 = aie::load_v<MMUL::size_B>(pB2);
                        pB2 += MMUL::size_B * colB;
                        B2 = aie::load_v<MMUL::size_B>(pB3);
                        pB3 += MMUL::size_B * colB;
                        B3 = aie::load_v<MMUL::size_B>(pB4);
                        pB4 += MMUL::size_B * colB;

                        C00.mac(A0, B0);
                        C01.mac(A0, B1);
                        C02.mac(A0, B2);
                        C03.mac(A0, B3);
                    }

                // TODO make shift right here to keep most significat bits
                // when lowering the output
                // example below shows how to shift right 10 bits
                // #define SHIFT 10
                // aie::store_v(pC1, C00.template to_vector<T_out>(SHIFT));

                aie::store_v(pC1, C00.template to_vector<T_out>());
                pC1 += MMUL::size_C;
                aie::store_v(pC1, C01.template to_vector<T_out>());
                pC1 += MMUL::size_C;
                aie::store_v(pC1, C02.template to_vector<T_out>());
                pC1 += MMUL::size_C;
                aie::store_v(pC1, C03.template to_vector<T_out>());
                pC1 += MMUL::size_C;
            }
    }

    event1();
}

/* Blocked MatMul kernel (vectorized) utilizing the aie::mmul class.
 * The matrices are assumed to be pre-tiled with the following shapes
 * for the aie:mmul class: A => rxs, B => sxt, C => rxt.
 *
 * The matrix dimensions of the kernel are defined by rowA, colA and colB.
 * In this particular kernel we expand the aie::mmul two times in each
 * input matrices A (in 'm' dimension, or rowA) and B (in 'n' dimension, or
 * ColB), leading to a 1x4 expansion in output matrix C (see C00, C01, C02, C03
 * below). This expansion helps with accumulator registers usage, which leads in
 * attaining high kernel efficiency (SIMD utilization).
 *
 * Data within each tile (rxs, sxt and rxt) are assumed to be in row-major
 * order. Also, the entire tiles themselves are stored in row-major order, as
 * shown in the example below for matrix A:
 *
 *      <-s->
 *    _  ________________________
 * 	  r |  1 |  2 |  3 | ...
 * 	  _ |____|____|____|
 * 	    |  x | x+1| x+2| ...
 * 	    |____|____|____|
 * 	    |.
 * 	    |.
 * 	    |.
 *
 * A simplified example of this kernel can be found in the AIE-API
 * documentation: https://xilinx.github.io/aie_api/group__group__mmul.html
 */
template <typename T_in,
          typename T_out,
          unsigned rowA,
          unsigned colA,
          unsigned colB,
          unsigned r,
          unsigned s,
          unsigned t>
void matmul_with_acc_vectorized_1x4_mmul(const T_in *__restrict pA,
                                         const T_in *__restrict pB,
                                         T_out *__restrict pAcc,
                                         T_out *__restrict pC)
{
    // Don't change functionality with debug mode since the weight matrix is an identity matrix
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

    event0();

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (unsigned z = 0; z < rowA; z += 1) {

        T_out *__restrict pAcc1 = pAcc + (z * colB) * MMUL::size_C;
        T_out *__restrict pC1 = pC + (z * colB) * MMUL::size_C;

        for (unsigned j = 0; j < colB; j += 4)
#ifdef OPT_PERF_ENABLED
            AIE_LOOP_FLATTEN
#endif
            {

                const T_in *__restrict pA1 = pA + (z * colA) * MMUL::size_A;
                const T_in *__restrict pB1 = pB + (j)*MMUL::size_B;
                const T_in *__restrict pB2 = pB + (j + 1) * MMUL::size_B;
                const T_in *__restrict pB3 = pB + (j + 2) * MMUL::size_B;
                const T_in *__restrict pB4 = pB + (j + 3) * MMUL::size_B;
                aie::vector<T_in, MMUL::size_A> A0;
                aie::vector<T_in, MMUL::size_B> B0;
                aie::vector<T_in, MMUL::size_B> B1;
                aie::vector<T_in, MMUL::size_B> B2;
                aie::vector<T_in, MMUL::size_B> B3;

                // Load partial results from C buffer for accumulation in-place. The
                // zero.cc function handles the zeroing of data when a new
                // accumulation is needed (after the 'K' reduction dimension)
                aie::vector<T_out, MMUL::size_C> acc_C00 = aie::load_v<MMUL::size_C>(pAcc1);
                aie::vector<T_out, MMUL::size_C> acc_C01 = aie::load_v<MMUL::size_C>(pAcc1);
                aie::vector<T_out, MMUL::size_C> acc_C02 = aie::load_v<MMUL::size_C>(pAcc1);
                aie::vector<T_out, MMUL::size_C> acc_C03 = aie::load_v<MMUL::size_C>(pAcc1);
                pAcc1 += MMUL::size_C;
                pAcc1 += MMUL::size_C;
                pAcc1 += MMUL::size_C;
                pAcc1 += MMUL::size_C;

                MMUL C00(acc_C00);
                MMUL C01(acc_C01);
                MMUL C02(acc_C02);
                MMUL C03(acc_C03);

                for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
                    AIE_LOOP_FLATTEN
#endif
                    {
                        A0 = aie::load_v<MMUL::size_A>(pA1);
                        pA1 += MMUL::size_A;
                        B0 = aie::load_v<MMUL::size_B>(pB1);
                        pB1 += MMUL::size_B * colB;
                        B1 = aie::load_v<MMUL::size_B>(pB2);
                        pB2 += MMUL::size_B * colB;
                        B2 = aie::load_v<MMUL::size_B>(pB3);
                        pB3 += MMUL::size_B * colB;
                        B3 = aie::load_v<MMUL::size_B>(pB4);
                        pB4 += MMUL::size_B * colB;

                        C00.mac(A0, B0);
                        C01.mac(A0, B1);
                        C02.mac(A0, B2);
                        C03.mac(A0, B3);
                    }

                // TODO make shift right here to keep most significat bits
                // when lowering the output
                // example below shows how to shift right 10 bits
                // #define SHIFT 10
                // aie::store_v(pC1, C00.template to_vector<T_out>(SHIFT));

                aie::store_v(pC1, C00.template to_vector<T_out>());
                pC1 += MMUL::size_C;
                aie::store_v(pC1, C01.template to_vector<T_out>());
                pC1 += MMUL::size_C;
                aie::store_v(pC1, C02.template to_vector<T_out>());
                pC1 += MMUL::size_C;
                aie::store_v(pC1, C03.template to_vector<T_out>());
                pC1 += MMUL::size_C;
            }
    }

    event1();
}

template <typename T, int N>
void fused_add_layer_norm_1(const T *restrict input,
                            const T *restrict residual,
                            const T *restrict weight,
                            T *restrict output,
                            int32_t cols,
                            int32_t rows_to_process)
{
    event0();
#ifndef DEBUG_AIE_KERNELS
    constexpr float epsilon = 1e-5f;
    int vector_chunks = cols / N;

    AIE_PREPARE_FOR_PIPELINING
    // AIE_LOOP_MIN_ITERATION_COUNT(4)
    for (int row = 0; row < rows_to_process; row++) {

        ::aie::vector<T, N> sum_acc = ::aie::zeros<T, N>();
        ::aie::vector<float, N> sum_sq_acc = ::aie::zeros<float, N>();
        int input_idx = row * cols;

        for (int i = 0; i < vector_chunks; i++) {

            ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + input_idx);
            sum_acc = ::aie::add(sum_acc, reg_a);
            ::aie::vector<float, N> sq_acc = ::aie::mul(reg_a, reg_a);
            sum_sq_acc = ::aie::add(sum_sq_acc, sq_acc);
            input_idx += N;
        }

        input_idx -= cols; // reset pointer to beginning of the row

        float sum_of_vals = ::aie::reduce_add(sum_acc);
        float sum_of_sq_vals = ::aie::reduce_add(sum_sq_acc);

        float mean = sum_of_vals / float(cols);
        float mean_sq = mean * mean;
        float variance = (sum_of_sq_vals / float(cols)) - mean_sq;
        float inv_std = aie::invsqrt(variance + epsilon);

        ::aie::vector<T, N> mean_v = ::aie::broadcast<T, N>(mean);
        ::aie::vector<T, N> inv_std_v = ::aie::broadcast<T, N>(inv_std);

        const T *__restrict pW = weight;
        const T *__restrict pRes = residual + row * cols;
        T *__restrict pOut = output + row * cols;
        for (int i = 0; i < vector_chunks; i++) {

            ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + input_idx);
            ::aie::vector<T, N> reg_weight = ::aie::load_v<N>(pW);
            ::aie::vector<T, N> reg_res = ::aie::load_v<N>(pRes);
            ::aie::vector<T, N> diff_v = ::aie::sub(reg_a, mean_v);
            ::aie::vector<T, N> norm_v = ::aie::mul(diff_v, inv_std_v);
            ::aie::vector<T, N> scaled_v = aie::mul(norm_v, reg_weight);
            // ::aie::vector<T, N> out_v = ::aie::add(scaled_v, beta_v);
            ::aie::vector<T, N> out_v = ::aie::add(scaled_v, reg_res);
            ::aie::store_v(pOut, out_v);
            input_idx += N;
            pW += N;
            pRes += N;
            pOut += N;
        }
    }
#else
#if DEBUG_AIE_KERNELS == 0
    // In debug mode, just copy input to output
    int total_elements = rows_to_process * cols;
    const T *__restrict pIn = input;
    T *__restrict pOut = output;
    AIE_PREPARE_FOR_PIPELINING
    // AIE_LOOP_MIN_ITERATION_COUNT(4)
    for (int i = 0; i < total_elements; i += N) {
        ::aie::vector<T, N> reg_a = ::aie::load_v<N>(pIn);
        ::aie::store_v(pOut, reg_a);
        pIn += N;
        pOut += N;
    }
#elif DEBUG_AIE_KERNELS == 1
    // In debug mode, just copy residual to output
    int total_elements = rows_to_process * cols;
    const T *__restrict pRes = residual;
    T *__restrict pOut = output;
    AIE_PREPARE_FOR_PIPELINING
    // AIE_LOOP_MIN_ITERATION_COUNT(4)
    for (int i = 0; i < total_elements; i += N) {
        ::aie::vector<T, N> reg_a = ::aie::load_v<N>(pRes);
        ::aie::store_v(pOut, reg_a);
        pRes += N;
        pOut += N;
    }
#endif
#endif
    event1();
}

template <typename T, int N>
void fused_add_layer_norm_2(const T *restrict input,
                            const T *restrict residual,
                            const T *restrict weight,
                            T *restrict output1,
                            T *restrict output2,
                            int32_t cols,
                            int32_t rows_to_process)
{
    event0();
#ifndef DEBUG_AIE_KERNELS
    constexpr float epsilon = 1e-5f;
    int vector_chunks = cols / N;

    // AIE_PREPARE_FOR_PIPELINING
    // AIE_LOOP_MIN_ITERATION_COUNT(4)
    for (int row = 0; row < rows_to_process; row++) {

        ::aie::vector<T, N> sum_acc = ::aie::zeros<T, N>();
        ::aie::vector<float, N> sum_sq_acc = ::aie::zeros<float, N>();
        int input_idx = row * cols;

        for (int i = 0; i < vector_chunks; i++) {

            ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + input_idx);
            sum_acc = ::aie::add(sum_acc, reg_a);
            ::aie::vector<float, N> sq_acc = ::aie::mul(reg_a, reg_a);
            sum_sq_acc = ::aie::add(sum_sq_acc, sq_acc);
            input_idx += N;
        }

        input_idx -= cols; // reset pointer to beginning of the row

        float sum_of_vals = ::aie::reduce_add(sum_acc);
        float sum_of_sq_vals = ::aie::reduce_add(sum_sq_acc);

        float mean = sum_of_vals / float(cols);
        float mean_sq = mean * mean;
        float variance = (sum_of_sq_vals / float(cols)) - mean_sq;
        float inv_std = aie::invsqrt(variance + epsilon);

        ::aie::vector<T, N> mean_v = ::aie::broadcast<T, N>(mean);
        ::aie::vector<T, N> inv_std_v = ::aie::broadcast<T, N>(inv_std);

        const T *__restrict pW = weight;
        const T *__restrict pRes = residual + row * cols;
        T *__restrict pOut1 = output1 + row * cols;
        T *__restrict pOut2 = output2 + row * cols;
        for (int i = 0; i < vector_chunks; i++) {

            ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + input_idx);
            ::aie::vector<T, N> reg_weight = ::aie::load_v<N>(pW);
            ::aie::vector<T, N> reg_res = ::aie::load_v<N>(pRes);
            ::aie::vector<T, N> diff_v = ::aie::sub(reg_a, mean_v);
            ::aie::vector<T, N> norm_v = ::aie::mul(diff_v, inv_std_v);
            ::aie::vector<T, N> scaled_v = aie::mul(norm_v, reg_weight);
            // ::aie::vector<T, N> out_v = ::aie::add(scaled_v, beta_v);
            ::aie::vector<T, N> out_v = ::aie::add(scaled_v, reg_res);
            ::aie::store_v(pOut1, out_v);
            ::aie::store_v(pOut2, out_v);
            input_idx += N;
            pW += N;
            pRes += N;
            pOut1 += N;
            pOut2 += N;
        }
    }
#else
#if DEBUG_AIE_KERNELS == 0

    // In debug mode, just copy input to output
    int total_elements = rows_to_process * cols;
    const T *__restrict pIn = input;
    T *__restrict pOut1 = output1;
    T *__restrict pOut2 = output2;
    AIE_PREPARE_FOR_PIPELINING
    // AIE_LOOP_MIN_ITERATION_COUNT(4)
    for (int i = 0; i < total_elements; i += N) {
        ::aie::vector<T, N> reg_a = ::aie::load_v<N>(pIn);
        ::aie::store_v(pOut1, reg_a);
        ::aie::store_v(pOut2, reg_a);
        pIn += N;
        pOut1 += N;
        pOut2 += N;
    }
#elif DEBUG_AIE_KERNELS == 1
    // In debug mode, just copy residual to output
    int total_elements = rows_to_process * cols;
    const T *__restrict pRes = residual;
    T *__restrict pOut1 = output1;
    T *__restrict pOut2 = output2;
    AIE_PREPARE_FOR_PIPELINING
    // AIE_LOOP_MIN_ITERATION_COUNT(4)
    for (int i = 0; i < total_elements; i += N) {
        ::aie::vector<T, N> reg_a = ::aie::load_v<N>(pRes);
        ::aie::store_v(pOut1, reg_a);
        ::aie::store_v(pOut2, reg_a);
        pRes += N;
        pOut1 += N;
        pOut2 += N;
    }
#endif
#endif
    event1();
}

template <typename T, int N>
void ffn_passThrough_aie(const T *restrict in, T *restrict out, int32_t k, int32_t rows_to_process, int32_t row_offset)
{
    event0();

    T *__restrict pOut = out;
    AIE_PREPARE_FOR_PIPELINING
    // AIE_LOOP_MIN_ITERATION_COUNT(4)
    for (unsigned row = 0; row < rows_to_process; row++) {
        // t = 8, r * t = m * 8 = num LN cores * rows per LN core * 8 = 8 * 8
        const T *__restrict pIn = in + row_offset * rows_to_process * 8 + row * (k / N) * (2 * rows_to_process) * 8;
        for (unsigned i = 0; i < k; i += N) {
            auto in_vec = ::aie::load_v<N>(pIn);
            ::aie::store_v(pOut, in_vec);
            pIn += 2 * (rows_to_process * 8); // stride is 64 bfloat16, so increment twice in the loop
            pOut += (rows_to_process * 8);
        }
    }

    event1();
}

template <typename T, int N>
void ln_passThrough_in_aie(const T *restrict in,
                           T *restrict out,
                           int32_t K,
                           int32_t k,
                           int32_t rows_to_process,
                           int32_t col_offset)
{
    event0();

    v64uint8 *restrict outPtr = (v64uint8 *)out;

    AIE_PREPARE_FOR_PIPELINING
    // AIE_LOOP_MIN_ITERATION_COUNT(4)
    for (unsigned row = 0; row < rows_to_process; row++) {

        v64uint8 *restrict inPtr = (v64uint8 *)(in + row * K + col_offset * k);

        for (int j = 0; j < k; j += N) { // Nx samples per loop

            *outPtr++ = *inPtr++;
        }
    }

    event1();
}

template <typename T, int N>
void ln_passThrough_out_aie(T *restrict in,
                            T *restrict out,
                            int32_t K,
                            int32_t k,
                            int32_t rows_to_process,
                            int32_t col_offset)
{
    event0();

    T *__restrict pIn = in;

    AIE_PREPARE_FOR_PIPELINING
    // AIE_LOOP_MIN_ITERATION_COUNT(4)
    for (unsigned row = 0; row < rows_to_process; row++) {

        T *__restrict pOut = out + row * K + col_offset * k;

        for (int j = 0; j < k; j += N) {

            ::aie::vector<T, N> reg_a = ::aie::load_v<N>(pIn);
            ::aie::store_v(pOut, reg_a);
            pOut += N;
            pIn += N;
        }
    }

    event1();
}

extern "C" {

// If you want to compile microkernels with different inner tile sizes,
// define DIM_M, DIM_K and DIM_N at compile time using -DDIM_M 32 etc.
// These dimensions must be divisible by the r, s, t dimensions used in
// the kernels.

#ifndef DIM_M
#define DIM_M 64
#endif

#ifndef DIM_K
#define DIM_K 64
#endif

#ifndef DIM_N
#define DIM_N 64
#endif

#ifdef BUILD_FFN
void zero_bf16_up_proj(bfloat16 *C)
{
    zero_vectorized<bfloat16, DIM_M, DIM_N>(C);
}

void zero_bf16_down_proj(bfloat16 *C)
{
    zero_vectorized<bfloat16, DIM_M, DIM_K>(C);
}

// void matmul_bf16_bf16_up_proj_half_inps(bfloat16 *A1, bfloat16 *A2, bfloat16 *B, bfloat16 *C)
// {
// #ifndef AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
//     static_assert(false, "AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 must be defined for this kernel");
// #endif
//     constexpr int r = 8;
//     constexpr int s = 8;
//     constexpr int t = 8;

//     static_assert(DIM_M % r == 0);
//     static_assert(DIM_K % s == 0);
//     static_assert(DIM_N % (1 * t) == 0);

//     ::aie::set_rounding(aie::rounding_mode::conv_even);

//     // NOTE: Assuming A1 and A2 each have size (DIM_M / 2) x DIM_K
//     matmul_vectorized_1x1_mmul<bfloat16, bfloat16, (DIM_M / r), (DIM_K / s), (DIM_N / t), r, s, t>(A1, A2, B, C);
// }

// void matmul_with_acc_bf16_bf16_down_proj(bfloat16 *A, bfloat16 *B, bfloat16 *pAcc, bfloat16 *C)
// {
// #ifndef AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
//     static_assert(false, "AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 must be defined for this kernel");
// #endif
//     constexpr int r = 8;
//     constexpr int s = 8;
//     constexpr int t = 8;

//     static_assert(DIM_M % r == 0);
//     static_assert(DIM_K % (1 * s) == 0);
//     static_assert(DIM_N % t == 0);

//     ::aie::set_rounding(aie::rounding_mode::conv_even);

//     // NOTE: K and N, s and t are swapped here compared to the up projection since up projection
//     // computes MxK with KxN, while down projection computes MxN with NxK
//     matmul_with_acc_vectorized_1x1_mmul<bfloat16, bfloat16, (DIM_M / r), (DIM_N / t), (DIM_K / s), r, t, s>(
//         A, B, pAcc, C);
// }

void matmul_bf16_bf16_up_proj_half_inps(bfloat16 *A1, bfloat16 *A2, bfloat16 *B, bfloat16 *C)
{
#ifndef AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
    static_assert(false, "AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 must be defined for this kernel");
#endif
    constexpr int r = 8;
    constexpr int s = 8;
    constexpr int t = 8;

    static_assert(DIM_M % r == 0);
    static_assert(DIM_K % s == 0);
    static_assert(DIM_N % (4 * t) == 0);

    ::aie::set_rounding(aie::rounding_mode::conv_even);

    matmul_vectorized_1x4_mmul<bfloat16, bfloat16, (DIM_M / r), (DIM_K / s), (DIM_N / t), r, s, t>(A1, A2, B, C);
}

void matmul_with_acc_bf16_bf16_down_proj(bfloat16 *A, bfloat16 *B, bfloat16 *pAcc, bfloat16 *C)
{
#ifndef AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
    static_assert(false, "AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 must be defined for this kernel");
#endif
    constexpr int r = 8;
    constexpr int s = 8;
    constexpr int t = 8;

    static_assert(DIM_M % r == 0);
    static_assert(DIM_K % (4 * s) == 0);
    static_assert(DIM_N % t == 0);

    ::aie::set_rounding(aie::rounding_mode::conv_even);

    // NOTE: K and N, s and t are swapped here compared to the up projection since up projection
    // computes MxK with KxN, while down projection computes MxN with NxK
    matmul_with_acc_vectorized_1x4_mmul<bfloat16, bfloat16, (DIM_M / r), (DIM_N / t), (DIM_K / s), r, t, s>(
        A, B, pAcc, C);
}

void ffn_passThroughTile_out(bfloat16 *in, bfloat16 *out, int32_t cols, int32_t rows_to_process, int32_t row_offset)
{
    ffn_passThrough_aie<bfloat16, 32>(
        in,
        out,
        cols,
        rows_to_process,
        row_offset); // Assumes input sz is larger than output sz, e.g. 8x96 input to 4x96 output
}

void ffn_gelu_bf16(bfloat16 *restrict input, bfloat16 *restrict output, int input_size)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    // Skip GeLU calculations in debug mode
#ifndef DEBUG_AIE_KERNELS
    gelu_tanh_approx_bf16(input, output, input_size);
#endif
}

void ffn_eltwise_add_bf16_vector(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out, int size)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    eltwise_vadd<bfloat16, bfloat16>(a_in, b_in, c_out, size);
}

#endif

#ifdef BUILD_ADDNORM
void fused_add_layer_norm_1outs(bfloat16 *input,
                                bfloat16 *residual,
                                bfloat16 *weights,
                                bfloat16 *output,
                                int32_t cols,
                                int32_t rows_to_process)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    fused_add_layer_norm_1<bfloat16, 32>(input, residual, weights, output, cols, rows_to_process);
}

void fused_add_layer_norm_2outs(bfloat16 *input,
                                bfloat16 *residual,
                                bfloat16 *weights,
                                bfloat16 *output1,
                                bfloat16 *output2,
                                int32_t cols,
                                int32_t rows_to_process)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    fused_add_layer_norm_2<bfloat16, 32>(input, residual, weights, output1, output2, cols, rows_to_process);
}

void ln_passThroughTile_out(int16_t *in,
                            int16_t *out,
                            int32_t cols,
                            int32_t cols_to_process,
                            int32_t rows_to_process,
                            int32_t col_offset)
{
    ln_passThrough_out_aie<int16_t, 32>(in, out, cols, cols_to_process, rows_to_process, col_offset);
}

void ln_passThroughTile_in(int16_t *in,
                           int16_t *out,
                           int32_t cols,
                           int32_t cols_to_process,
                           int32_t rows_to_process,
                           int32_t col_offset)
{
    ln_passThrough_in_aie<int16_t, 32>(in, out, cols, cols_to_process, rows_to_process, col_offset);
}
#endif
}
