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
 * for the aie::mmul class: A => rxs, B => sxt, C => rxt.
 *
 * The matrix dimensions of the kernel are defined by rowA, colA and colB.
 * In this kernel we expand the aie::mmul two times in both the 'm' dimension
 * and the 'n' dimension, producing a 2x2 block in the output matrix.
 *
 * Data within each tile (rxs, sxt and rxt) are assumed to be in row-major
 * order. Also, the entire tiles themselves are stored in row-major order.
 */
template <typename T_in,
          typename T_out,
          unsigned rowA,
          unsigned colA,
          unsigned colB,
          unsigned r,
          unsigned s,
          unsigned t,
          bool b_row_maj = true,
          bool c_row_maj = true>
void matmul_vectorized_2x2_mmul(const T_in *__restrict pA, const T_in *__restrict pB, T_out *__restrict pC)
{
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

    event0();

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (unsigned z = 0; z < rowA; z += 2) {

        T_out *__restrict pC1;
        T_out *__restrict pC2;
        if constexpr (c_row_maj) {
            pC1 = pC + (z * colB) * MMUL::size_C;
            pC2 = pC + ((z + 1) * colB) * MMUL::size_C;
        }

        for (unsigned j = 0; j < colB; j += 2)
#ifdef OPT_PERF_ENABLED
            AIE_LOOP_FLATTEN
#endif
            {
                if constexpr (!c_row_maj) {
                    pC1 = pC + j * rowA * MMUL::size_C + z * MMUL::size_C;
                    pC2 = pC + (j + 1) * rowA * MMUL::size_C + z * MMUL::size_C;
                }

                const T_in *__restrict pA1 = pA + (z * colA) * MMUL::size_A;
                const T_in *__restrict pA2 = pA + ((z + 1) * colA) * MMUL::size_A;
                const T_in *__restrict pB1 = pB + (j)*MMUL::size_B;
                const T_in *__restrict pB2 = pB + (j + 1) * MMUL::size_B;
                aie::vector<T_in, MMUL::size_A> A0;
                aie::vector<T_in, MMUL::size_A> A1;
                aie::vector<T_in, MMUL::size_B> B0;
                aie::vector<T_in, MMUL::size_B> B1;

                aie::vector<T_out, MMUL::size_C> acc_C00;
                aie::vector<T_out, MMUL::size_C> acc_C01;
                aie::vector<T_out, MMUL::size_C> acc_C10;
                aie::vector<T_out, MMUL::size_C> acc_C11;
                if constexpr (c_row_maj) {
                    acc_C00 = aie::load_v<MMUL::size_C>(pC1);
                    acc_C01 = aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
                    acc_C10 = aie::load_v<MMUL::size_C>(pC2);
                    acc_C11 = aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);
                } else {
                    acc_C00 = aie::transpose(aie::load_v<MMUL::size_C>(pC1), t, r);
                    acc_C01 = aie::transpose(aie::load_v<MMUL::size_C>(pC2), t, r);
                    acc_C10 = aie::transpose(aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C), t, r);
                    acc_C11 = aie::transpose(aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C), t, r);
                }

                MMUL C00(acc_C00);
                MMUL C01(acc_C01);
                MMUL C10(acc_C10);
                MMUL C11(acc_C11);

                for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
                    AIE_LOOP_FLATTEN
#endif
                    {
                        A0 = aie::load_v<MMUL::size_A>(pA1);
                        pA1 += MMUL::size_A;
                        A1 = aie::load_v<MMUL::size_A>(pA2);
                        pA2 += MMUL::size_A;
                        if constexpr (b_row_maj) {
                            B0 = aie::load_v<MMUL::size_B>(pB1);
                            pB1 += MMUL::size_B * colB;
                            B1 = aie::load_v<MMUL::size_B>(pB2);
                            pB2 += MMUL::size_B * colB;
                        } else {
                            B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), t, s);
                            pB1 += MMUL::size_B;
                            B1 = aie::transpose(aie::load_v<MMUL::size_B>(pB2), t, s);
                            pB2 += MMUL::size_B;
                        }

                        C00.mac(A0, B0);
                        C01.mac(A0, B1);
                        C10.mac(A1, B0);
                        C11.mac(A1, B1);
                    }

                if constexpr (c_row_maj) {
                    aie::store_v(pC1, C00.template to_vector<T_out>());
                    pC1 += MMUL::size_C;
                    aie::store_v(pC1, C01.template to_vector<T_out>());
                    pC1 += MMUL::size_C;
                    aie::store_v(pC2, C10.template to_vector<T_out>());
                    pC2 += MMUL::size_C;
                    aie::store_v(pC2, C11.template to_vector<T_out>());
                    pC2 += MMUL::size_C;
                } else {
                    aie::store_v(pC1, aie::transpose(C00.template to_vector<T_out>(), r, t));
                    pC1 += MMUL::size_C;
                    aie::store_v(pC2, aie::transpose(C01.template to_vector<T_out>(), r, t));
                    pC2 += MMUL::size_C;
                    aie::store_v(pC1, aie::transpose(C10.template to_vector<T_out>(), r, t));
                    pC1 += MMUL::size_C;
                    aie::store_v(pC2, aie::transpose(C11.template to_vector<T_out>(), r, t));
                    pC2 += MMUL::size_C;
                }
            }
    }

    event1();
}

/* First-K matmul variant: computes C = A*B without loading prior C values.
 * This avoids an unnecessary accumulator-buffer read when starting a new reduction.
 */
template <typename T_in,
          typename T_out,
          unsigned rowA,
          unsigned colA,
          unsigned colB,
          unsigned r,
          unsigned s,
          unsigned t>
void matmul_init_vectorized_2x2_mmul(const T_in *__restrict pA, const T_in *__restrict pB, T_out *__restrict pC)
{
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

    event0();

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (unsigned z = 0; z < rowA; z += 2) {

        T_out *__restrict pC1 = pC + (z * colB) * MMUL::size_C;
        T_out *__restrict pC2 = pC + ((z + 1) * colB) * MMUL::size_C;

        for (unsigned j = 0; j < colB; j += 2)
#ifdef OPT_PERF_ENABLED
            AIE_LOOP_FLATTEN
#endif
            {

                const T_in *__restrict pA1 = pA + (z * colA) * MMUL::size_A;
                const T_in *__restrict pA2 = pA + ((z + 1) * colA) * MMUL::size_A;
                const T_in *__restrict pB1 = pB + (j)*MMUL::size_B;
                const T_in *__restrict pB2 = pB + (j + 1) * MMUL::size_B;
                aie::vector<T_in, MMUL::size_A> A0;
                aie::vector<T_in, MMUL::size_A> A1;
                aie::vector<T_in, MMUL::size_B> B0;
                aie::vector<T_in, MMUL::size_B> B1;

                aie::vector<T_out, MMUL::size_C> zero_c = aie::zeros<T_out, MMUL::size_C>();
                MMUL C00(zero_c);
                MMUL C01(zero_c);
                MMUL C10(zero_c);
                MMUL C11(zero_c);

                for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
                    AIE_LOOP_FLATTEN
#endif
                    {
                        A0 = aie::load_v<MMUL::size_A>(pA1);
                        pA1 += MMUL::size_A;
                        A1 = aie::load_v<MMUL::size_A>(pA2);
                        pA2 += MMUL::size_A;
                        B0 = aie::load_v<MMUL::size_B>(pB1);
                        pB1 += MMUL::size_B * colB;
                        B1 = aie::load_v<MMUL::size_B>(pB2);
                        pB2 += MMUL::size_B * colB;

                        C00.mac(A0, B0);
                        C01.mac(A0, B1);
                        C10.mac(A1, B0);
                        C11.mac(A1, B1);
                    }

                aie::store_v(pC1, C00.template to_vector<T_out>());
                pC1 += MMUL::size_C;
                aie::store_v(pC1, C01.template to_vector<T_out>());
                aie::store_v(pC2, C10.template to_vector<T_out>());
                pC2 += MMUL::size_C;
                aie::store_v(pC2, C11.template to_vector<T_out>());
            }
    }

    event1();
}

template <typename T_in,
          typename T_out,
          unsigned rowA,
          unsigned colA,
          unsigned colB,
          unsigned r,
          unsigned s,
          unsigned t,
          bool b_row_maj = true,
          bool c_row_maj = true>
void matmul_init_vectorized_2x1_mmul(const T_in *__restrict pA, const T_in *__restrict pB, T_out *__restrict pC)
{
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

    event0();

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (unsigned z = 0; z < rowA; z += 2) {

        T_out *__restrict pC1;
        T_out *__restrict pC2;
        if constexpr (c_row_maj) {
            pC1 = pC + (z * colB) * MMUL::size_C;
            pC2 = pC + ((z + 1) * colB) * MMUL::size_C;
        }

        for (unsigned j = 0; j < colB; ++j)
#ifdef OPT_PERF_ENABLED
            AIE_LOOP_FLATTEN
#endif
            {
                if constexpr (!c_row_maj) {
                    pC1 = pC + j * rowA * MMUL::size_C + z * MMUL::size_C;
                    pC2 = pC + j * rowA * MMUL::size_C + (z + 1) * MMUL::size_C;
                }

                const T_in *__restrict pA1 = pA + (z * colA) * MMUL::size_A;
                const T_in *__restrict pA2 = pA + ((z + 1) * colA) * MMUL::size_A;
                const T_in *__restrict pB1;
                if constexpr (b_row_maj) {
                    pB1 = pB + j * MMUL::size_B;
                } else {
                    pB1 = pB + (j * colA) * MMUL::size_B;
                }
                aie::vector<T_in, MMUL::size_A> A0;
                aie::vector<T_in, MMUL::size_A> A1;
                aie::vector<T_in, MMUL::size_B> B0;

                aie::vector<T_out, MMUL::size_C> zero_c = aie::zeros<T_out, MMUL::size_C>();
                MMUL C00(zero_c);
                MMUL C10(zero_c);

                for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
                    AIE_LOOP_FLATTEN
#endif
                    {
                        A0 = aie::load_v<MMUL::size_A>(pA1);
                        pA1 += MMUL::size_A;
                        A1 = aie::load_v<MMUL::size_A>(pA2);
                        pA2 += MMUL::size_A;
                        if constexpr (b_row_maj) {
                            B0 = aie::load_v<MMUL::size_B>(pB1);
                            pB1 += MMUL::size_B * colB;
                        } else {
                            B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), t, s);
                            pB1 += MMUL::size_B;
                        }

                        C00.mac(A0, B0);
                        C10.mac(A1, B0);
                    }

                if constexpr (c_row_maj) {
                    aie::store_v(pC1, C00.template to_vector<T_out>());
                    pC1 += MMUL::size_C;
                    aie::store_v(pC2, C10.template to_vector<T_out>());
                    pC2 += MMUL::size_C;
                } else {
                    aie::store_v(pC1, aie::transpose(C00.template to_vector<T_out>(), r, t));
                    pC1 += MMUL::size_C;
                    aie::store_v(pC2, aie::transpose(C10.template to_vector<T_out>(), r, t));
                    pC2 += MMUL::size_C;
                }
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
          unsigned t,
          bool b_row_maj = true,
          bool c_row_maj = true>
void matmul_with_acc_vectorized_2x2_mmul(const T_in *__restrict pA,
                                         const T_in *__restrict pB,
                                         T_out *__restrict pAcc,
                                         T_out *__restrict pC)
{
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

    event0();

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (unsigned z = 0; z < rowA; z += 2) {

        T_out *__restrict pAcc1;
        T_out *__restrict pAcc2;
        T_out *__restrict pC1;
        T_out *__restrict pC2;
        if constexpr (c_row_maj) {
            pAcc1 = pAcc + (z * colB) * MMUL::size_C;
            pAcc2 = pAcc + ((z + 1) * colB) * MMUL::size_C;
            pC1 = pC + (z * colB) * MMUL::size_C;
            pC2 = pC + ((z + 1) * colB) * MMUL::size_C;
        }

        for (unsigned j = 0; j < colB; j += 2)
#ifdef OPT_PERF_ENABLED
            AIE_LOOP_FLATTEN
#endif
            {
                if constexpr (!c_row_maj) {
                    pAcc1 = pAcc + j * rowA * MMUL::size_C + z * MMUL::size_C;
                    pAcc2 = pAcc + (j + 1) * rowA * MMUL::size_C + z * MMUL::size_C;
                    pC1 = pC + j * rowA * MMUL::size_C + z * MMUL::size_C;
                    pC2 = pC + (j + 1) * rowA * MMUL::size_C + z * MMUL::size_C;
                }

                const T_in *__restrict pA1 = pA + (z * colA) * MMUL::size_A;
                const T_in *__restrict pA2 = pA + ((z + 1) * colA) * MMUL::size_A;
                const T_in *__restrict pB1 = pB + (j)*MMUL::size_B;
                const T_in *__restrict pB2 = pB + (j + 1) * MMUL::size_B;
                aie::vector<T_in, MMUL::size_A> A0;
                aie::vector<T_in, MMUL::size_A> A1;
                aie::vector<T_in, MMUL::size_B> B0;
                aie::vector<T_in, MMUL::size_B> B1;

                aie::vector<T_out, MMUL::size_C> acc_C00;
                aie::vector<T_out, MMUL::size_C> acc_C01;
                aie::vector<T_out, MMUL::size_C> acc_C10;
                aie::vector<T_out, MMUL::size_C> acc_C11;
                if constexpr (c_row_maj) {
                    acc_C00 = aie::load_v<MMUL::size_C>(pAcc1);
                    pAcc1 += MMUL::size_C;
                    acc_C01 = aie::load_v<MMUL::size_C>(pAcc1);
                    pAcc1 += MMUL::size_C;
                    acc_C10 = aie::load_v<MMUL::size_C>(pAcc2);
                    pAcc2 += MMUL::size_C;
                    acc_C11 = aie::load_v<MMUL::size_C>(pAcc2);
                    pAcc2 += MMUL::size_C;
                } else {
                    acc_C00 = aie::transpose(aie::load_v<MMUL::size_C>(pAcc1), t, r);
                    pAcc1 += MMUL::size_C;
                    acc_C01 = aie::transpose(aie::load_v<MMUL::size_C>(pAcc2), t, r);
                    pAcc2 += MMUL::size_C;
                    acc_C10 = aie::transpose(aie::load_v<MMUL::size_C>(pAcc1), t, r);
                    pAcc1 += MMUL::size_C;
                    acc_C11 = aie::transpose(aie::load_v<MMUL::size_C>(pAcc2), t, r);
                    pAcc2 += MMUL::size_C;
                }

                MMUL C00(acc_C00);
                MMUL C01(acc_C01);
                MMUL C10(acc_C10);
                MMUL C11(acc_C11);

                for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
                    AIE_LOOP_FLATTEN
#endif
                    {
                        A0 = aie::load_v<MMUL::size_A>(pA1);
                        pA1 += MMUL::size_A;
                        A1 = aie::load_v<MMUL::size_A>(pA2);
                        pA2 += MMUL::size_A;
                        if constexpr (b_row_maj) {
                            B0 = aie::load_v<MMUL::size_B>(pB1);
                            pB1 += MMUL::size_B * colB;
                            B1 = aie::load_v<MMUL::size_B>(pB2);
                            pB2 += MMUL::size_B * colB;
                        } else {
                            B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), t, s);
                            pB1 += MMUL::size_B;
                            B1 = aie::transpose(aie::load_v<MMUL::size_B>(pB2), t, s);
                            pB2 += MMUL::size_B;
                        }

                        C00.mac(A0, B0);
                        C01.mac(A0, B1);
                        C10.mac(A1, B0);
                        C11.mac(A1, B1);
                    }

                if constexpr (c_row_maj) {
                    aie::store_v(pC1, C00.template to_vector<T_out>());
                    pC1 += MMUL::size_C;
                    aie::store_v(pC1, C01.template to_vector<T_out>());
                    pC1 += MMUL::size_C;
                    aie::store_v(pC2, C10.template to_vector<T_out>());
                    pC2 += MMUL::size_C;
                    aie::store_v(pC2, C11.template to_vector<T_out>());
                    pC2 += MMUL::size_C;
                } else {
                    aie::store_v(pC1, aie::transpose(C00.template to_vector<T_out>(), r, t));
                    pC1 += MMUL::size_C;
                    aie::store_v(pC2, aie::transpose(C01.template to_vector<T_out>(), r, t));
                    pC2 += MMUL::size_C;
                    aie::store_v(pC1, aie::transpose(C10.template to_vector<T_out>(), r, t));
                    pC1 += MMUL::size_C;
                    aie::store_v(pC2, aie::transpose(C11.template to_vector<T_out>(), r, t));
                    pC2 += MMUL::size_C;
                }
            }
    }

    event1();
}

template <typename T_in,
          typename T_out,
          unsigned rowA,
          unsigned colA,
          unsigned colB,
          unsigned r,
          unsigned s,
          unsigned t,
          bool b_row_maj = true,
          bool c_row_maj = true>
void matmul_with_acc_vectorized_2x1_mmul(const T_in *__restrict pA,
                                         const T_in *__restrict pB,
                                         T_out *__restrict pAcc,
                                         T_out *__restrict pC)
{
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

    event0();

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (unsigned z = 0; z < rowA; z += 2) {

        T_out *__restrict pAcc1;
        T_out *__restrict pAcc2;
        T_out *__restrict pC1;
        T_out *__restrict pC2;
        if constexpr (c_row_maj) {
            pAcc1 = pAcc + (z * colB) * MMUL::size_C;
            pAcc2 = pAcc + ((z + 1) * colB) * MMUL::size_C;
            pC1 = pC + (z * colB) * MMUL::size_C;
            pC2 = pC + ((z + 1) * colB) * MMUL::size_C;
        }

        for (unsigned j = 0; j < colB; ++j)
#ifdef OPT_PERF_ENABLED
            AIE_LOOP_FLATTEN
#endif
            {
                if constexpr (!c_row_maj) {
                    pAcc1 = pAcc + j * rowA * MMUL::size_C + z * MMUL::size_C;
                    pAcc2 = pAcc + j * rowA * MMUL::size_C + (z + 1) * MMUL::size_C;
                    pC1 = pC + j * rowA * MMUL::size_C + z * MMUL::size_C;
                    pC2 = pC + j * rowA * MMUL::size_C + (z + 1) * MMUL::size_C;
                }

                const T_in *__restrict pA1 = pA + (z * colA) * MMUL::size_A;
                const T_in *__restrict pA2 = pA + ((z + 1) * colA) * MMUL::size_A;
                const T_in *__restrict pB1;
                if constexpr (b_row_maj) {
                    pB1 = pB + j * MMUL::size_B;
                } else {
                    pB1 = pB + (j * colA) * MMUL::size_B;
                }
                aie::vector<T_in, MMUL::size_A> A0;
                aie::vector<T_in, MMUL::size_A> A1;
                aie::vector<T_in, MMUL::size_B> B0;

                aie::vector<T_out, MMUL::size_C> acc_C00;
                aie::vector<T_out, MMUL::size_C> acc_C10;
                if constexpr (c_row_maj) {
                    acc_C00 = aie::load_v<MMUL::size_C>(pAcc1);
                    pAcc1 += MMUL::size_C;
                    acc_C10 = aie::load_v<MMUL::size_C>(pAcc2);
                    pAcc2 += MMUL::size_C;
                } else {
                    acc_C00 = aie::transpose(aie::load_v<MMUL::size_C>(pAcc1), t, r);
                    pAcc1 += MMUL::size_C;
                    acc_C10 = aie::transpose(aie::load_v<MMUL::size_C>(pAcc2), t, r);
                    pAcc2 += MMUL::size_C;
                }

                MMUL C00(acc_C00);
                MMUL C10(acc_C10);

                for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
                    AIE_LOOP_FLATTEN
#endif
                    {
                        A0 = aie::load_v<MMUL::size_A>(pA1);
                        pA1 += MMUL::size_A;
                        A1 = aie::load_v<MMUL::size_A>(pA2);
                        pA2 += MMUL::size_A;
                        if constexpr (b_row_maj) {
                            B0 = aie::load_v<MMUL::size_B>(pB1);
                            pB1 += MMUL::size_B * colB;
                        } else {
                            B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), t, s);
                            pB1 += MMUL::size_B;
                        }

                        C00.mac(A0, B0);
                        C10.mac(A1, B0);
                    }

                if constexpr (c_row_maj) {
                    aie::store_v(pC1, C00.template to_vector<T_out>());
                    pC1 += MMUL::size_C;
                    aie::store_v(pC2, C10.template to_vector<T_out>());
                    pC2 += MMUL::size_C;
                } else {
                    aie::store_v(pC1, aie::transpose(C00.template to_vector<T_out>(), r, t));
                    pC1 += MMUL::size_C;
                    aie::store_v(pC2, aie::transpose(C10.template to_vector<T_out>(), r, t));
                    pC2 += MMUL::size_C;
                }
            }
    }

    event1();
}

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
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;
    constexpr int szA = MMUL::size_A;

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
                const T_in *__restrict pB1 = pB + (j)*MMUL::size_B;
                const T_in *__restrict pB2 = pB + (j + 1) * MMUL::size_B;
                const T_in *__restrict pB3 = pB + (j + 2) * MMUL::size_B;
                const T_in *__restrict pB4 = pB + (j + 3) * MMUL::size_B;
                aie::vector<T_in, szA> HA10;
                aie::vector<T_in, MMUL::size_B> B0;
                aie::vector<T_in, MMUL::size_B> B1;
                aie::vector<T_in, MMUL::size_B> B2;
                aie::vector<T_in, MMUL::size_B> B3;

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
                        B0 = aie::load_v<MMUL::size_B>(pB1);
                        pB1 += MMUL::size_B * colB;
                        B1 = aie::load_v<MMUL::size_B>(pB2);
                        pB2 += MMUL::size_B * colB;
                        B2 = aie::load_v<MMUL::size_B>(pB3);
                        pB3 += MMUL::size_B * colB;
                        B3 = aie::load_v<MMUL::size_B>(pB4);
                        pB4 += MMUL::size_B * colB;

                        C00.mac(HA10, B0);
                        C01.mac(HA10, B1);
                        C02.mac(HA10, B2);
                        C03.mac(HA10, B3);
                    }

                aie::store_v(pC1, C00.template to_vector<T_out>());
                pC1 += MMUL::size_C;
                aie::store_v(pC1, C01.template to_vector<T_out>());
                pC1 += MMUL::size_C;
                aie::store_v(pC1, C02.template to_vector<T_out>());
                pC1 += MMUL::size_C;
                aie::store_v(pC1, C03.template to_vector<T_out>());
            }

        T_out *__restrict pC2 = pC + ((z + rowA) * colB) * MMUL::size_C;

        for (unsigned j = 0; j < colB; j += 4)
#ifdef OPT_PERF_ENABLED
            AIE_LOOP_FLATTEN
#endif
            {
                const T_in *__restrict pHA2 = pHalfA2 + (z * colA) * szA;
                const T_in *__restrict pB1 = pB + (j)*MMUL::size_B;
                const T_in *__restrict pB2 = pB + (j + 1) * MMUL::size_B;
                const T_in *__restrict pB3 = pB + (j + 2) * MMUL::size_B;
                const T_in *__restrict pB4 = pB + (j + 3) * MMUL::size_B;
                aie::vector<T_in, szA> HA20;
                aie::vector<T_in, MMUL::size_B> B0;
                aie::vector<T_in, MMUL::size_B> B1;
                aie::vector<T_in, MMUL::size_B> B2;
                aie::vector<T_in, MMUL::size_B> B3;

                aie::vector<T_out, MMUL::size_C> acc_C10 = aie::load_v<MMUL::size_C>(pC2);
                aie::vector<T_out, MMUL::size_C> acc_C11 = aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);
                aie::vector<T_out, MMUL::size_C> acc_C12 = aie::load_v<MMUL::size_C>(pC2 + 2 * MMUL::size_C);
                aie::vector<T_out, MMUL::size_C> acc_C13 = aie::load_v<MMUL::size_C>(pC2 + 3 * MMUL::size_C);

                MMUL C10(acc_C10);
                MMUL C11(acc_C11);
                MMUL C12(acc_C12);
                MMUL C13(acc_C13);

                for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
                    AIE_LOOP_FLATTEN
#endif
                    {
                        HA20 = aie::load_v<szA>(pHA2);
                        pHA2 += szA;
                        B0 = aie::load_v<MMUL::size_B>(pB1);
                        pB1 += MMUL::size_B * colB;
                        B1 = aie::load_v<MMUL::size_B>(pB2);
                        pB2 += MMUL::size_B * colB;
                        B2 = aie::load_v<MMUL::size_B>(pB3);
                        pB3 += MMUL::size_B * colB;
                        B3 = aie::load_v<MMUL::size_B>(pB4);
                        pB4 += MMUL::size_B * colB;

                        C10.mac(HA20, B0);
                        C11.mac(HA20, B1);
                        C12.mac(HA20, B2);
                        C13.mac(HA20, B3);
                    }

                aie::store_v(pC2, C10.template to_vector<T_out>());
                pC2 += MMUL::size_C;
                aie::store_v(pC2, C11.template to_vector<T_out>());
                pC2 += MMUL::size_C;
                aie::store_v(pC2, C12.template to_vector<T_out>());
                pC2 += MMUL::size_C;
                aie::store_v(pC2, C13.template to_vector<T_out>());
            }
    }

    event1();
}

template <typename T_in,
          typename T_out,
          unsigned rowA,
          unsigned colA,
          unsigned colB,
          unsigned r,
          unsigned s,
          unsigned t>
void matmul_vectorized_1x4_mmul_MLessThanr(const T_in *__restrict pHalfA1,
                                           const T_in *__restrict pHalfA2,
                                           const T_in *__restrict pB,
                                           T_out *__restrict pC)
{
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;
    constexpr int szA = MMUL::size_A / 2;

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
                        auto A0 = ::aie::concat(HA10, HA20);
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

                aie::store_v(pC1, C00.template to_vector<T_out>());
                pC1 += MMUL::size_C;
                aie::store_v(pC1, C01.template to_vector<T_out>());
                pC1 += MMUL::size_C;
                aie::store_v(pC1, C02.template to_vector<T_out>());
                pC1 += MMUL::size_C;
                aie::store_v(pC1, C03.template to_vector<T_out>());
            }
    }

    event1();
}

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

                aie::vector<T_out, MMUL::size_C> acc_C00 = aie::load_v<MMUL::size_C>(pAcc1);
                pAcc1 += MMUL::size_C;
                aie::vector<T_out, MMUL::size_C> acc_C01 = aie::load_v<MMUL::size_C>(pAcc1);
                pAcc1 += MMUL::size_C;
                aie::vector<T_out, MMUL::size_C> acc_C02 = aie::load_v<MMUL::size_C>(pAcc1);
                pAcc1 += MMUL::size_C;
                aie::vector<T_out, MMUL::size_C> acc_C03 = aie::load_v<MMUL::size_C>(pAcc1);
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

                aie::store_v(pC1, C00.template to_vector<T_out>());
                pC1 += MMUL::size_C;
                aie::store_v(pC1, C01.template to_vector<T_out>());
                pC1 += MMUL::size_C;
                aie::store_v(pC1, C02.template to_vector<T_out>());
                pC1 += MMUL::size_C;
                aie::store_v(pC1, C03.template to_vector<T_out>());
            }
    }

    event1();
}

template <typename T, unsigned rowA, unsigned colA, unsigned r, unsigned s>
void fused_add_layer_norm_1(const T *restrict input,
                            const T *restrict residual,
                            const T *restrict weight,
                            const float *restrict sum,
                            const float *restrict sumsq,
                            T *restrict output,
                            const int32_t cols,
                            const int32_t col_idx) // For offset into weight vector
{
    event0();
    (void)residual;

    constexpr float epsilon = 1e-5f;
    constexpr unsigned mmul_c_size = r * s; // This is the number of elements in each C tile (microtile)

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (unsigned z = 0; z < rowA; z += 1) {

        const T *__restrict pA1 = input + (z * colA) * mmul_c_size;
        T *__restrict pC1 = output + (z * colA) * mmul_c_size;

        // Each row has an accumulator for sum and sum-squared, so offset by z*r to get to the correct row,
        // and then we'll index into the correct one within that row with the loop below
        const float *__restrict pSum1 = sum + z * r;
        const float *__restrict pSumSq1 = sumsq + z * r;

        // For each row within this microtile, apply layer norm and add residual
        for (unsigned ri = 0; ri < r; ri++) {
            float mean = aie::div(*pSum1, aie::to_float(cols));
            float mean_sq = mean * mean;
            float variance = aie::div(*pSumSq1, aie::to_float(cols)) - mean_sq;
            if (variance < 0.0f) {
                variance = 0.0f;
            }
            float inv_std = aie::invsqrt(variance + epsilon);

            const T *__restrict pW = weight + (col_idx * colA * s);

            for (unsigned j = 0; j < colA; j += 2) {
                aie::vector<T, s> A0 = aie::load_v<s>(pA1);
                pA1 += mmul_c_size; // Move pointer to the start of the next microtile in the same row
                aie::vector<T, s> A1 = aie::load_v<s>(pA1);
                pA1 += mmul_c_size; // Move pointer to the start of the next microtile in the same row
                auto A01 = aie::concat(A0, A1);
                aie::accum<accfloat, 2 * s> a_acc;
                a_acc.from_vector(A01);
                aie::accum<accfloat, 2 * s> diff_acc = aie::sub(a_acc, mean);
                aie::accum<accfloat, 2 * s> norm_acc = aie::mul(diff_acc.template to_vector<float>(), inv_std);
                aie::vector<T, 2 * s> weight_v = aie::load_v<2 * s>(pW);
                pW += 2 * s; // Move weight pointer to the columns
                aie::vector<T, 2 * s> scaled_acc = aie::mul(norm_acc.template to_vector<T>(), weight_v);
                aie::vector<T, 2 * s> out_acc = scaled_acc;

#ifndef DEBUG_AIE_KERNELS
                // Write s elements to one row of four r x s tiles in output
                aie::store_v(pC1, out_acc.template extract<s>(0));
                pC1 += mmul_c_size; // Move pointer to the start of the next microtile in the same row
                aie::store_v(pC1, out_acc.template extract<s>(1));
#else
#if DEBUG_AIE_KERNELS == 0
                // Input values
                aie::store_v(pC1, A0);
                pC1 += mmul_c_size; // Move pointer to the start of the next microtile in the same row
                aie::store_v(pC1, A1);
#elif DEBUG_AIE_KERNELS == 1
                // This helper has no residual input; mirror the input path.
                aie::store_v(pC1, A0);
                pC1 += mmul_c_size; // Move pointer to the start of the next microtile in the same row
                aie::store_v(pC1, A1);
#endif
#endif
                pC1 += mmul_c_size; // Move pointer to the start of the next microtile in the same row
            }

            pSum1++;
            pSumSq1++;
            pA1 -= colA * mmul_c_size; // Move pointer back to the start of the row
            pA1 += s;                  // Move pointer to the next row within the same microtile
            pC1 -= colA * mmul_c_size; // Move pointer back to the start of the row
            pC1 += s;                  // Move pointer to the next row within the same microtile
        }
    }
    event1();
}

#ifdef BUILD_ADDNORM_REPLAY_FASTPATH
template <typename T, unsigned rowA, unsigned colA, unsigned r, unsigned s>
void add_1_from_inputs(const T *restrict input, const T *restrict residual, T *restrict output)
{
    event0();

    constexpr unsigned mmul_c_size = r * s;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (unsigned z = 0; z < rowA; z += 1) {
        const T *__restrict pA1 = input + (z * colA) * mmul_c_size;
        const T *__restrict pR1 = residual + (z * colA) * mmul_c_size;
        T *__restrict pC1 = output + (z * colA) * mmul_c_size;

        for (unsigned ri = 0; ri < r; ri++) {
            for (unsigned j = 0; j < colA; j += 2) {
                aie::vector<T, s> A0 = aie::load_v<s>(pA1);
                pA1 += mmul_c_size;
                aie::vector<T, s> A1 = aie::load_v<s>(pA1);
                pA1 += mmul_c_size;
                aie::vector<T, s> R0 = aie::load_v<s>(pR1);
                pR1 += mmul_c_size;
                aie::vector<T, s> R1 = aie::load_v<s>(pR1);
                pR1 += mmul_c_size;
                aie::store_v(pC1, aie::add(A0, R0));
                pC1 += mmul_c_size;
                aie::store_v(pC1, aie::add(A1, R1));
                pC1 += mmul_c_size;
            }

            pA1 -= colA * mmul_c_size;
            pA1 += s;
            pR1 -= colA * mmul_c_size;
            pR1 += s;
            pC1 -= colA * mmul_c_size;
            pC1 += s;
        }
    }
    event1();
}

template <typename T, unsigned rowA, unsigned colA, unsigned r, unsigned s>
void fused_add_layer_norm_1_from_inputs(const T *restrict input,
                                        const T *restrict residual,
                                        const T *restrict weight,
                                        const float *restrict sum,
                                        const float *restrict sumsq,
                                        T *restrict output,
                                        const int32_t cols,
                                        const int32_t col_idx)
{
    event0();

    constexpr float epsilon = 1e-5f;
    constexpr unsigned mmul_c_size = r * s; // This is the number of elements in each C tile (microtile)

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (unsigned z = 0; z < rowA; z += 1) {

        const T *__restrict pA1 = input + (z * colA) * mmul_c_size;
        const T *__restrict pR1 = residual + (z * colA) * mmul_c_size;
        T *__restrict pC1 = output + (z * colA) * mmul_c_size;

        const float *__restrict pSum1 = sum + z * r;
        const float *__restrict pSumSq1 = sumsq + z * r;

        for (unsigned ri = 0; ri < r; ri++) {
            float mean = aie::div(*pSum1, aie::to_float(cols));
            float mean_sq = mean * mean;
            float variance = aie::div(*pSumSq1, aie::to_float(cols)) - mean_sq;
            if (variance < 0.0f) {
                variance = 0.0f;
            }
            float inv_std = aie::invsqrt(variance + epsilon);

            const T *__restrict pW = weight + (col_idx * colA * s);

            for (unsigned j = 0; j < colA; j += 2) {
                aie::vector<T, s> A0 = aie::load_v<s>(pA1);
                pA1 += mmul_c_size;
                aie::vector<T, s> A1 = aie::load_v<s>(pA1);
                pA1 += mmul_c_size;
                aie::vector<T, s> R0 = aie::load_v<s>(pR1);
                pR1 += mmul_c_size;
                aie::vector<T, s> R1 = aie::load_v<s>(pR1);
                pR1 += mmul_c_size;
                aie::vector<T, s> merged0 = aie::add(A0, R0);
                aie::vector<T, s> merged1 = aie::add(A1, R1);
                auto merged01 = aie::concat(merged0, merged1);
                aie::accum<accfloat, 2 * s> a_acc;
                a_acc.from_vector(merged01);
                aie::accum<accfloat, 2 * s> diff_acc = aie::sub(a_acc, mean);
                aie::accum<accfloat, 2 * s> norm_acc = aie::mul(diff_acc.template to_vector<float>(), inv_std);
                aie::vector<T, 2 * s> weight_v = aie::load_v<2 * s>(pW);
                pW += 2 * s;
                aie::vector<T, 2 * s> scaled_acc = aie::mul(norm_acc.template to_vector<T>(), weight_v);
                aie::vector<T, 2 * s> out_acc = scaled_acc;

#ifndef DEBUG_AIE_KERNELS
                aie::store_v(pC1, out_acc.template extract<s>(0));
                pC1 += mmul_c_size;
                aie::store_v(pC1, out_acc.template extract<s>(1));
#else
#if DEBUG_AIE_KERNELS == 0
                aie::store_v(pC1, merged0);
                pC1 += mmul_c_size;
                aie::store_v(pC1, merged1);
#elif DEBUG_AIE_KERNELS == 1
                aie::store_v(pC1, R0);
                pC1 += mmul_c_size;
                aie::store_v(pC1, R1);
#endif
#endif
                pC1 += mmul_c_size;
            }

            pSum1++;
            pSumSq1++;
            pA1 -= colA * mmul_c_size;
            pA1 += s;
            pR1 -= colA * mmul_c_size;
            pR1 += s;
            pC1 -= colA * mmul_c_size;
            pC1 += s;
        }
    }
    event1();
}

template <typename T, unsigned rowA, unsigned colA, unsigned r, unsigned s>
void fused_add_layer_norm_2_from_inputs(const T *restrict input,
                                        const T *restrict residual,
                                        const T *restrict weight,
                                        const float *restrict sum,
                                        const float *restrict sumsq,
                                        T *restrict output,
                                        T *restrict merged_output,
                                        const int32_t cols,
                                        const int32_t col_idx)
{
    event0();

    constexpr float epsilon = 1e-5f;
    constexpr unsigned mmul_c_size = r * s;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (unsigned z = 0; z < rowA; z += 1) {

        const T *__restrict pA1 = input + (z * colA) * mmul_c_size;
        const T *__restrict pR1 = residual + (z * colA) * mmul_c_size;
        T *__restrict pC1 = output + (z * colA) * mmul_c_size;
        T *__restrict pMerged1 = merged_output + (z * colA) * mmul_c_size;

        const float *__restrict pSum1 = sum + z * r;
        const float *__restrict pSumSq1 = sumsq + z * r;

        for (unsigned ri = 0; ri < r; ri++) {
            float mean = aie::div(*pSum1, aie::to_float(cols));
            float mean_sq = mean * mean;
            float variance = aie::div(*pSumSq1, aie::to_float(cols)) - mean_sq;
            if (variance < 0.0f) {
                variance = 0.0f;
            }
            float inv_std = aie::invsqrt(variance + epsilon);

            const T *__restrict pW = weight + (col_idx * colA * s);

            for (unsigned j = 0; j < colA; j += 2) {
                aie::vector<T, s> A0 = aie::load_v<s>(pA1);
                pA1 += mmul_c_size;
                aie::vector<T, s> A1 = aie::load_v<s>(pA1);
                pA1 += mmul_c_size;
                aie::vector<T, s> R0 = aie::load_v<s>(pR1);
                pR1 += mmul_c_size;
                aie::vector<T, s> R1 = aie::load_v<s>(pR1);
                pR1 += mmul_c_size;
                aie::vector<T, s> merged0 = aie::add(A0, R0);
                aie::vector<T, s> merged1 = aie::add(A1, R1);
                auto merged01 = aie::concat(merged0, merged1);
                aie::accum<accfloat, 2 * s> a_acc;
                a_acc.from_vector(merged01);
                aie::accum<accfloat, 2 * s> diff_acc = aie::sub(a_acc, mean);
                aie::accum<accfloat, 2 * s> norm_acc = aie::mul(diff_acc.template to_vector<float>(), inv_std);
                aie::vector<T, 2 * s> weight_v = aie::load_v<2 * s>(pW);
                pW += 2 * s;
                aie::vector<T, 2 * s> scaled_acc = aie::mul(norm_acc.template to_vector<T>(), weight_v);

                aie::store_v(pMerged1, merged0);
                pMerged1 += mmul_c_size;
                aie::store_v(pMerged1, merged1);
                pMerged1 += mmul_c_size;

#ifndef DEBUG_AIE_KERNELS
                aie::store_v(pC1, scaled_acc.template extract<s>(0));
                pC1 += mmul_c_size;
                aie::store_v(pC1, scaled_acc.template extract<s>(1));
#else
#if DEBUG_AIE_KERNELS == 0
                aie::store_v(pC1, merged0);
                pC1 += mmul_c_size;
                aie::store_v(pC1, merged1);
#elif DEBUG_AIE_KERNELS == 1
                aie::store_v(pC1, R0);
                pC1 += mmul_c_size;
                aie::store_v(pC1, R1);
#endif
#endif
                pC1 += mmul_c_size;
            }

            pSum1++;
            pSumSq1++;
            pA1 -= colA * mmul_c_size;
            pA1 += s;
            pR1 -= colA * mmul_c_size;
            pR1 += s;
            pC1 -= colA * mmul_c_size;
            pC1 += s;
            pMerged1 -= colA * mmul_c_size;
            pMerged1 += s;
        }
    }
    event1();
}
#endif

template <typename T, unsigned rowA, unsigned colA, unsigned r, unsigned s>
void fused_add_layer_norm_1_fp32weights(const T *restrict input,
                                        const T *restrict residual,
                                        const int32_t *restrict weight_bits,
                                        const float *restrict sum,
                                        const float *restrict sumsq,
                                        T *restrict output,
                                        const int32_t cols,
                                        const int32_t col_idx)
{
    event0();
    (void)residual;

    constexpr float epsilon = 1e-5f;
    constexpr unsigned mmul_c_size = r * s;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (unsigned z = 0; z < rowA; z += 1) {

        const T *__restrict pA1 = input + (z * colA) * mmul_c_size;
        T *__restrict pC1 = output + (z * colA) * mmul_c_size;

        const float *__restrict pSum1 = sum + z * r;
        const float *__restrict pSumSq1 = sumsq + z * r;

        for (unsigned ri = 0; ri < r; ri++) {
            float mean = aie::div(*pSum1, aie::to_float(cols));
            float mean_sq = mean * mean;
            float variance = aie::div(*pSumSq1, aie::to_float(cols)) - mean_sq;
            if (variance < 0.0f) {
                variance = 0.0f;
            }
            float inv_std = aie::invsqrt(variance + epsilon);

            const float *__restrict pW = reinterpret_cast<const float *>(weight_bits) + (col_idx * colA * s);

            for (unsigned j = 0; j < colA; j += 2) {
                aie::vector<T, s> A0 = aie::load_v<s>(pA1);
                pA1 += mmul_c_size;
                aie::vector<T, s> A1 = aie::load_v<s>(pA1);
                pA1 += mmul_c_size;
                auto A01 = aie::concat(A0, A1);
                aie::accum<accfloat, 2 * s> a_acc;
                a_acc.from_vector(A01);
                aie::accum<accfloat, 2 * s> diff_acc = aie::sub(a_acc, mean);
                aie::accum<accfloat, 2 * s> norm_acc = aie::mul(diff_acc.template to_vector<float>(), inv_std);
                aie::vector<float, 2 * s> weight_v = aie::load_v<2 * s>(pW);
                pW += 2 * s;
                aie::accum<accfloat, 2 * s> scaled_acc = aie::mul(norm_acc.template to_vector<float>(), weight_v);
                aie::vector<T, 2 * s> out_acc = scaled_acc.template to_vector<T>();

#ifndef DEBUG_AIE_KERNELS
                aie::store_v(pC1, out_acc.template extract<s>(0));
                pC1 += mmul_c_size;
                aie::store_v(pC1, out_acc.template extract<s>(1));
#else
#if DEBUG_AIE_KERNELS == 0
                aie::store_v(pC1, A0);
                pC1 += mmul_c_size;
                aie::store_v(pC1, A1);
#endif
#endif
                pC1 += mmul_c_size;
            }

            pSum1++;
            pSumSq1++;
            pA1 -= colA * mmul_c_size;
            pA1 += s;
            pC1 -= colA * mmul_c_size;
            pC1 += s;
        }
    }
    event1();
}

#ifdef BUILD_ADDNORM_REPLAY_FASTPATH
template <typename T, unsigned rowA, unsigned colA, unsigned r, unsigned s>
void fused_add_layer_norm_1_from_inputs_fp32weights(const T *restrict input,
                                                    const T *restrict residual,
                                                    const int32_t *restrict weight_bits,
                                                    const float *restrict sum,
                                                    const float *restrict sumsq,
                                                    T *restrict output,
                                                    const int32_t cols,
                                                    const int32_t col_idx)
{
    event0();

    constexpr float epsilon = 1e-5f;
    constexpr unsigned mmul_c_size = r * s;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (unsigned z = 0; z < rowA; z += 1) {

        const T *__restrict pA1 = input + (z * colA) * mmul_c_size;
        const T *__restrict pR1 = residual + (z * colA) * mmul_c_size;
        T *__restrict pC1 = output + (z * colA) * mmul_c_size;

        const float *__restrict pSum1 = sum + z * r;
        const float *__restrict pSumSq1 = sumsq + z * r;

        for (unsigned ri = 0; ri < r; ri++) {
            float mean = aie::div(*pSum1, aie::to_float(cols));
            float mean_sq = mean * mean;
            float variance = aie::div(*pSumSq1, aie::to_float(cols)) - mean_sq;
            if (variance < 0.0f) {
                variance = 0.0f;
            }
            float inv_std = aie::invsqrt(variance + epsilon);

            const float *__restrict pW = reinterpret_cast<const float *>(weight_bits) + (col_idx * colA * s);

            for (unsigned j = 0; j < colA; j += 2) {
                aie::vector<T, s> A0 = aie::load_v<s>(pA1);
                pA1 += mmul_c_size;
                aie::vector<T, s> A1 = aie::load_v<s>(pA1);
                pA1 += mmul_c_size;
                aie::vector<T, s> R0 = aie::load_v<s>(pR1);
                pR1 += mmul_c_size;
                aie::vector<T, s> R1 = aie::load_v<s>(pR1);
                pR1 += mmul_c_size;
                aie::vector<T, s> merged0 = aie::add(A0, R0);
                aie::vector<T, s> merged1 = aie::add(A1, R1);
                auto merged01 = aie::concat(merged0, merged1);
                aie::accum<accfloat, 2 * s> a_acc;
                a_acc.from_vector(merged01);
                aie::accum<accfloat, 2 * s> diff_acc = aie::sub(a_acc, mean);
                aie::accum<accfloat, 2 * s> norm_acc = aie::mul(diff_acc.template to_vector<float>(), inv_std);
                aie::vector<float, 2 * s> weight_v = aie::load_v<2 * s>(pW);
                pW += 2 * s;
                aie::accum<accfloat, 2 * s> scaled_acc = aie::mul(norm_acc.template to_vector<float>(), weight_v);
                aie::vector<T, 2 * s> out_acc = scaled_acc.template to_vector<T>();

#ifndef DEBUG_AIE_KERNELS
                aie::store_v(pC1, out_acc.template extract<s>(0));
                pC1 += mmul_c_size;
                aie::store_v(pC1, out_acc.template extract<s>(1));
#else
#if DEBUG_AIE_KERNELS == 0
                aie::store_v(pC1, merged0);
                pC1 += mmul_c_size;
                aie::store_v(pC1, merged1);
#elif DEBUG_AIE_KERNELS == 1
                aie::store_v(pC1, R0);
                pC1 += mmul_c_size;
                aie::store_v(pC1, R1);
#endif
#endif
                pC1 += mmul_c_size;
            }

            pSum1++;
            pSumSq1++;
            pA1 -= colA * mmul_c_size;
            pA1 += s;
            pR1 -= colA * mmul_c_size;
            pR1 += s;
            pC1 -= colA * mmul_c_size;
            pC1 += s;
        }
    }
    event1();
}
#endif

template <typename T, unsigned rowA, unsigned colA, unsigned r, unsigned s>
void fused_layer_norm_1(const T *restrict input,
                        const float *restrict sum,
                        const float *restrict sumsq,
                        T *restrict output,
                        const int32_t cols)
{
    event0();

    constexpr float epsilon = 1e-5f;
    constexpr unsigned mmul_c_size = r * s; // Number of elements in each C microtile

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (unsigned z = 0; z < rowA; z += 1) {
        const T *__restrict pA1 = input + (z * colA) * mmul_c_size;
        T *__restrict pC1 = output + (z * colA) * mmul_c_size;

        const float *__restrict pSum1 = sum + z * r;
        const float *__restrict pSumSq1 = sumsq + z * r;

        // For each row within this microtile, apply layer norm (no weight/residual).
        for (unsigned ri = 0; ri < r; ri++) {
            float mean = aie::div(*pSum1, aie::to_float(cols));
            float mean_sq = mean * mean;
            float variance = aie::div(*pSumSq1, aie::to_float(cols)) - mean_sq;
            if (variance < 0.0f) {
                variance = 0.0f;
            }
            float inv_std = aie::invsqrt(variance + epsilon);

            for (unsigned j = 0; j < colA; j += 2) {
                aie::vector<T, s> A0 = aie::load_v<s>(pA1);
                pA1 += mmul_c_size;
                aie::vector<T, s> A1 = aie::load_v<s>(pA1);
                pA1 += mmul_c_size;
                auto A01 = aie::concat(A0, A1);

                aie::accum<accfloat, 2 * s> a_acc;
                a_acc.from_vector(A01);
                aie::accum<accfloat, 2 * s> diff_acc = aie::sub(a_acc, mean);
                aie::accum<accfloat, 2 * s> norm_acc = aie::mul(diff_acc.template to_vector<float>(), inv_std);
                aie::vector<T, 2 * s> out_acc = norm_acc.template to_vector<T>();

                aie::store_v(pC1, out_acc.template extract<s>(0));
                pC1 += mmul_c_size;
                aie::store_v(pC1, out_acc.template extract<s>(1));
                pC1 += mmul_c_size;
            }

            pSum1++;
            pSumSq1++;
            pA1 -= colA * mmul_c_size;
            pA1 += s;
            pC1 -= colA * mmul_c_size;
            pC1 += s;
        }
    }
    event1();
}

template <typename T, unsigned rowA, unsigned colA, unsigned r, unsigned s>
void ln_mul_weights_1(const T *restrict input, const T *restrict weight, T *restrict output, const int32_t col_idx)
{
    event0();

    constexpr unsigned mmul_c_size = r * s; // Number of elements in each C microtile

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (unsigned z = 0; z < rowA; z += 1) {
        const T *__restrict pA1 = input + (z * colA) * mmul_c_size;
        T *__restrict pC1 = output + (z * colA) * mmul_c_size;
        const T *__restrict pW_row = weight + (col_idx * colA * s);

        for (unsigned ri = 0; ri < r; ri++) {
            const T *__restrict pW = pW_row;
            for (unsigned j = 0; j < colA; j += 2) {
                aie::vector<T, s> A0 = aie::load_v<s>(pA1);
                pA1 += mmul_c_size;
                aie::vector<T, s> A1 = aie::load_v<s>(pA1);
                pA1 += mmul_c_size;
                auto A01 = aie::concat(A0, A1);

                aie::vector<T, 2 * s> weight_v = aie::load_v<2 * s>(pW);
                pW += 2 * s;
                aie::vector<T, 2 * s> out_acc = aie::mul(A01, weight_v);

                aie::store_v(pC1, out_acc.template extract<s>(0));
                pC1 += mmul_c_size;
                aie::store_v(pC1, out_acc.template extract<s>(1));
                pC1 += mmul_c_size;
            }

            pA1 -= colA * mmul_c_size;
            pA1 += s;
            pC1 -= colA * mmul_c_size;
            pC1 += s;
        }
    }
    event1();
}

template <typename T, unsigned rowA, unsigned colA, unsigned r, unsigned s>
void ln_mul_add_1(const T *restrict input,
                  const T *restrict residual,
                  const T *restrict weight,
                  T *restrict output,
                  const int32_t col_idx)
{
    event0();

    constexpr unsigned mmul_c_size = r * s; // Number of elements in each C microtile

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (unsigned z = 0; z < rowA; z += 1) {
        const T *__restrict pA1 = input + (z * colA) * mmul_c_size;
        const T *__restrict pR1 = residual + (z * colA) * mmul_c_size;
        T *__restrict pC1 = output + (z * colA) * mmul_c_size;
        const T *__restrict pW_row = weight + (col_idx * colA * s);

        for (unsigned ri = 0; ri < r; ri++) {
            const T *__restrict pW = pW_row;
            for (unsigned j = 0; j < colA; j += 2) {
                aie::vector<T, s> A0 = aie::load_v<s>(pA1);
                pA1 += mmul_c_size;
                aie::vector<T, s> A1 = aie::load_v<s>(pA1);
                pA1 += mmul_c_size;
                auto A01 = aie::concat(A0, A1);

                aie::vector<T, s> R0 = aie::load_v<s>(pR1);
                pR1 += mmul_c_size;
                aie::vector<T, s> R1 = aie::load_v<s>(pR1);
                pR1 += mmul_c_size;
                auto R01 = aie::concat(R0, R1);

                aie::vector<T, 2 * s> weight_v = aie::load_v<2 * s>(pW);
                pW += 2 * s;
                aie::vector<T, 2 * s> scaled = aie::mul(A01, weight_v);
                aie::vector<T, 2 * s> out_acc = aie::add(scaled, R01);

                aie::store_v(pC1, out_acc.template extract<s>(0));
                pC1 += mmul_c_size;
                aie::store_v(pC1, out_acc.template extract<s>(1));
                pC1 += mmul_c_size;
            }

            pA1 -= colA * mmul_c_size;
            pA1 += s;
            pR1 -= colA * mmul_c_size;
            pR1 += s;
            pC1 -= colA * mmul_c_size;
            pC1 += s;
        }
    }
    event1();
}

template <typename T, int N>
void fused_add_layer_norm_2(const T *restrict input,
                            const T *restrict residual,
                            const T *restrict weight,
                            T *restrict output1,
                            T *restrict output2,
                            const int32_t cols,
                            const int32_t rows_to_process)
{
    event0();
    (void)residual;
#ifndef DEBUG_AIE_KERNELS
    constexpr float epsilon = 1e-5f;
    int vector_chunks = cols / N;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(4)
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
        if (variance < 0.0f) {
            variance = 0.0f;
        }
        float inv_std = aie::invsqrt(variance + epsilon);

        ::aie::vector<T, N> mean_v = ::aie::broadcast<T, N>(mean);
        ::aie::vector<T, N> inv_std_v = ::aie::broadcast<T, N>(inv_std);

        const T *__restrict pW = weight;
        T *__restrict pOut1 = output1 + row * cols;
        T *__restrict pOut2 = output2 + row * cols;
        for (int i = 0; i < vector_chunks; i++) {

            ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + input_idx);
            ::aie::vector<T, N> reg_weight = ::aie::load_v<N>(pW);
            ::aie::vector<T, N> diff_v = ::aie::sub(reg_a, mean_v);
            ::aie::vector<T, N> norm_v = ::aie::mul(diff_v, inv_std_v);
            ::aie::vector<T, N> scaled_v = aie::mul(norm_v, reg_weight);
            ::aie::vector<T, N> out_v = scaled_v;
            ::aie::store_v(pOut1, out_v);
            ::aie::store_v(pOut2, out_v);
            input_idx += N;
            pW += N;
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

template <typename T, int N> void ln_zero_vectorized(T *__restrict c, int size)
{
    event0();

    T *__restrict pOut1 = c;
    auto zero_v = ::aie::zeros<T, N>();

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int j = 0; j < size; j += N) {
        ::aie::store_v(pOut1, zero_v);
        pOut1 += N;
    }

    event1();
}

template <typename T, unsigned rowA, unsigned colA, unsigned r, unsigned s>
void ln_calc_sum_sumsq_vectorized(const T *__restrict pA, float *__restrict pSum, float *__restrict pSumSq)
{
    event0();

    constexpr unsigned mmul_c_size = r * s; // This is the number of elements in each C tile (microtile)

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (unsigned z = 0; z < rowA; z += 1) {
        const T *__restrict pA1 = pA + (z * colA) * mmul_c_size;
        // Each row has an accumulator for sum and sum-squared, so offset by z*r to get to the correct row,
        // and then we'll index into the correct one within that row with the loop below
        float *__restrict pSum1 = pSum + z * r;
        float *__restrict pSumSq1 = pSumSq + z * r;

        // Per-row accumulators for sum and sum-squared across all column tiles
        for (unsigned i = 0; i < r; i++) {
            float row_sum = *pSum1;
            float row_sumsq = *pSumSq1;
            for (unsigned j = 0; j < colA; j += 2) {
                aie::vector<T, s> A0 = aie::load_v<s>(pA1);
                pA1 += mmul_c_size; // Move pointer to the start of the next microtile in the same row
                aie::vector<T, s> A1 = aie::load_v<s>(pA1);
                pA1 += mmul_c_size; // Move pointer to the start of the next microtile in the same row

                auto A01 = aie::concat(A0, A1);
                aie::accum<accfloat, 2 * s> sum_acc;
                sum_acc.from_vector(A01);
                float sum = aie::reduce_add(sum_acc.template to_vector<float>());
                row_sum += sum;

                aie::vector<float, s> a_acc_sq0 = aie::mul(A0, A0);
                float sumsq = aie::reduce_add(a_acc_sq0);
                row_sumsq += sumsq;
                aie::vector<float, s> a_acc_sq1 = aie::mul(A1, A1);
                sumsq = aie::reduce_add(a_acc_sq1);
                row_sumsq += sumsq;
            }
            *pSum1 = row_sum;
            *pSumSq1 = row_sumsq;
            pSum1++;
            pSumSq1++;
            pA1 -= colA * mmul_c_size; // Move pointer back to the start of the row
            pA1 += s;                  // Move pointer to the next row within the same microtile
        }
    }

    event1();
}

#ifdef BUILD_ADDNORM_REPLAY_FASTPATH
template <typename T, unsigned rowA, unsigned colA, unsigned r, unsigned s>
void ln_add_calc_sum_sumsq_vectorized(const T *__restrict pA,
                                      const T *__restrict pB,
                                      float *__restrict pSum,
                                      float *__restrict pSumSq)
{
    event0();

    constexpr unsigned mmul_c_size = r * s; // This is the number of elements in each C tile (microtile)

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (unsigned z = 0; z < rowA; z += 1) {
        const T *__restrict pA1 = pA + (z * colA) * mmul_c_size;
        const T *__restrict pB1 = pB + (z * colA) * mmul_c_size;
        float *__restrict pSum1 = pSum + z * r;
        float *__restrict pSumSq1 = pSumSq + z * r;

        for (unsigned i = 0; i < r; i++) {
            float row_sum = *pSum1;
            float row_sumsq = *pSumSq1;
            for (unsigned j = 0; j < colA; j += 2) {
                aie::vector<T, s> A0 = aie::load_v<s>(pA1);
                pA1 += mmul_c_size;
                aie::vector<T, s> A1 = aie::load_v<s>(pA1);
                pA1 += mmul_c_size;
                aie::vector<T, s> B0 = aie::load_v<s>(pB1);
                pB1 += mmul_c_size;
                aie::vector<T, s> B1 = aie::load_v<s>(pB1);
                pB1 += mmul_c_size;

                aie::vector<T, s> merged0 = aie::add(A0, B0);
                aie::vector<T, s> merged1 = aie::add(A1, B1);
                auto merged01 = aie::concat(merged0, merged1);
                aie::accum<accfloat, 2 * s> sum_acc;
                sum_acc.from_vector(merged01);
                row_sum += aie::reduce_add(sum_acc.template to_vector<float>());

                aie::vector<float, s> merged_sq0 = aie::mul(merged0, merged0);
                row_sumsq += aie::reduce_add(merged_sq0);
                aie::vector<float, s> merged_sq1 = aie::mul(merged1, merged1);
                row_sumsq += aie::reduce_add(merged_sq1);
            }
            *pSum1 = row_sum;
            *pSumSq1 = row_sumsq;
            pSum1++;
            pSumSq1++;
            pA1 -= colA * mmul_c_size;
            pA1 += s;
            pB1 -= colA * mmul_c_size;
            pB1 += s;
        }
    }

    event1();
}
#endif

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
void ffn_zero_bf16_up_proj(bfloat16 *C)
{
    zero_vectorized<bfloat16, DIM_M, DIM_N>(C);
}

void ffn_zero_bf16_down_proj(bfloat16 *C)
{
    zero_vectorized<bfloat16, DIM_M, DIM_K>(C);
}

void ffn_matmul_init_bf16_bf16_up_proj(const bfloat16 *A, const bfloat16 *B, bfloat16 *C)
{
#ifndef AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
    static_assert(false, "AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 must be defined for this kernel");
#endif
    constexpr int r = 8;
    constexpr int s = 8;
    constexpr int t = 8;

    static_assert(DIM_M % r == 0);
    static_assert(DIM_K % s == 0);

    ::aie::set_rounding(aie::rounding_mode::conv_even);

    zero_vectorized<bfloat16, DIM_M, DIM_N>(C);
    if constexpr (DIM_M < (2 * r)) {
        static_assert(DIM_N % (4 * t) == 0);
        constexpr int half_tile_elems = (DIM_M / 2) * DIM_K;
        matmul_vectorized_1x4_mmul_MLessThanr<bfloat16, bfloat16, (DIM_M / r), (DIM_K / s), (DIM_N / t), r, s, t>(
            A, A + half_tile_elems, B, C);
    } else {
        static_assert(DIM_N % (2 * t) == 0);
        matmul_vectorized_2x2_mmul<bfloat16, bfloat16, (DIM_M / r), (DIM_K / s), (DIM_N / t), r, s, t>(A, B, C);
    }
}

void ffn_matmul_bf16_bf16_up_proj(const bfloat16 *A, const bfloat16 *B, bfloat16 *C)
{
#ifndef AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
    static_assert(false, "AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 must be defined for this kernel");
#endif
    constexpr int r = 8;
    constexpr int s = 8;
    constexpr int t = 8;

    static_assert(DIM_M % r == 0);
    static_assert(DIM_K % s == 0);

    ::aie::set_rounding(aie::rounding_mode::conv_even);

    if constexpr (DIM_M < (2 * r)) {
        static_assert(DIM_N % (4 * t) == 0);
        constexpr int half_tile_elems = (DIM_M / 2) * DIM_K;
        matmul_vectorized_1x4_mmul_MLessThanr<bfloat16, bfloat16, (DIM_M / r), (DIM_K / s), (DIM_N / t), r, s, t>(
            A, A + half_tile_elems, B, C);
    } else {
        static_assert(DIM_N % (2 * t) == 0);
        matmul_vectorized_2x2_mmul<bfloat16, bfloat16, (DIM_M / r), (DIM_K / s), (DIM_N / t), r, s, t>(A, B, C);
    }
}

void ffn_matmul_bf16_bf16_up_proj_half_inps(const bfloat16 *A1, const bfloat16 *A2, const bfloat16 *B, bfloat16 *C)
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

    if constexpr (DIM_M <= r) {
        matmul_vectorized_1x4_mmul_MLessThanr<bfloat16, bfloat16, (DIM_M / r), (DIM_K / s), (DIM_N / t), r, s, t>(
            A1, A2, B, C);
    } else {
        matmul_vectorized_1x4_mmul<bfloat16, bfloat16, (DIM_M / (2 * r)), (DIM_K / s), (DIM_N / t), r, s, t>(
            A1, A2, B, C);
    }
}

void matmul_bf16_bf16_up_proj_half_inps(const bfloat16 *A1, const bfloat16 *A2, const bfloat16 *B, bfloat16 *C)
{
    ffn_matmul_bf16_bf16_up_proj_half_inps(A1, A2, B, C);
}

void ffn_matmul_init_bf16_bf16_down_proj(const bfloat16 *A, const bfloat16 *B, bfloat16 *C)
{
#ifndef AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
    static_assert(false, "AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 must be defined for this kernel");
#endif
    constexpr int r = 8;
    constexpr int s = 8;
    constexpr int t = 8;

    static_assert(DIM_M % r == 0);
    static_assert(DIM_N % t == 0);

    ::aie::set_rounding(aie::rounding_mode::conv_even);

    zero_vectorized<bfloat16, DIM_M, DIM_K>(C);
    if constexpr ((DIM_N % (2 * t) == 0) && (DIM_K == 24)) {
        static_assert(DIM_M % (2 * r) == 0);
        matmul_init_vectorized_2x1_mmul<bfloat16, bfloat16, (DIM_M / r), (DIM_N / t), (DIM_K / s), r, s, t>(A, B, C);
    } else if constexpr (DIM_M < (2 * r)) {
        static_assert(DIM_K % (4 * s) == 0);
        matmul_with_acc_vectorized_1x4_mmul<bfloat16, bfloat16, (DIM_M / r), (DIM_N / t), (DIM_K / s), r, t, s>(
            A, B, C, C);
    } else if constexpr (DIM_N % (2 * t) == 0) {
        matmul_vectorized_2x2_mmul<bfloat16, bfloat16, (DIM_M / r), (DIM_N / t), (DIM_K / s), r, s, t>(A, B, C);
    } else {
        static_assert(DIM_M % (2 * r) == 0);
        matmul_init_vectorized_2x1_mmul<bfloat16, bfloat16, (DIM_M / r), (DIM_N / t), (DIM_K / s), r, s, t>(A, B, C);
    }
}

void ffn_matmul_with_acc_bf16_bf16_down_proj(const bfloat16 *A, const bfloat16 *B, bfloat16 *pAcc, bfloat16 *C)
{
#ifndef AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
    static_assert(false, "AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 must be defined for this kernel");
#endif
    constexpr int r = 8;
    constexpr int s = 8;
    constexpr int t = 8;

    static_assert(DIM_M % r == 0);
    static_assert(DIM_N % t == 0);

    ::aie::set_rounding(aie::rounding_mode::conv_even);

    if constexpr ((DIM_N % (2 * t) == 0) && (DIM_K == 24)) {
        static_assert(DIM_M % (2 * r) == 0);
        matmul_with_acc_vectorized_2x1_mmul<bfloat16, bfloat16, (DIM_M / r), (DIM_N / t), (DIM_K / s), r, s, t>(
            A, B, pAcc, C);
    } else if constexpr (DIM_M < (2 * r)) {
        static_assert(DIM_K % (4 * s) == 0);
        matmul_with_acc_vectorized_1x4_mmul<bfloat16, bfloat16, (DIM_M / r), (DIM_N / t), (DIM_K / s), r, t, s>(
            A, B, pAcc, C);
    } else if constexpr (DIM_N % (2 * t) == 0) {
        matmul_with_acc_vectorized_2x2_mmul<bfloat16, bfloat16, (DIM_M / r), (DIM_N / t), (DIM_K / s), r, s, t>(
            A, B, pAcc, C);
    } else {
        static_assert(DIM_M % (2 * r) == 0);
        matmul_with_acc_vectorized_2x1_mmul<bfloat16, bfloat16, (DIM_M / r), (DIM_N / t), (DIM_K / s), r, s, t>(
            A, B, pAcc, C);
    }
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
void ln_passThroughTile_in(const bfloat16 *input,
                           bfloat16 *output,
                           const int32_t cols,
                           const int32_t tileWidth,
                           const int32_t tileHeight,
                           const int32_t col_idx)
{
    event0();
    const bfloat16 *pIn = input + col_idx * tileWidth;
    bfloat16 *pOut = output;
    AIE_PREPARE_FOR_PIPELINING
    for (int row = 0; row < tileHeight; ++row) {
        const bfloat16 *pInRow = pIn + row * cols;
        for (int col = 0; col < tileWidth; col += 32) {
            auto reg = ::aie::load_v<32>(pInRow + col);
            ::aie::store_v(pOut + col, reg);
        }
        pOut += tileWidth;
    }
    event1();
}

void fused_add_layer_norm_1outs(const bfloat16 *input,
                                const bfloat16 *residual,
                                const bfloat16 *weights,
                                const float *sum,
                                const float *sumsq,
                                bfloat16 *output,
                                const int32_t cols,
                                const int32_t col_idx)
{
    constexpr int r = 8;
    constexpr int s = 8;

    static_assert(DIM_M % r == 0);
    static_assert(DIM_K % s == 0);

    ::aie::set_rounding(aie::rounding_mode::conv_even);
    fused_add_layer_norm_1<bfloat16, (DIM_M / r), (DIM_K / s), r, s>(
        input, residual, weights, sum, sumsq, output, cols, col_idx);
}

void fused_add_layer_norm_1outs_fp32weights(const bfloat16 *input,
                                            const bfloat16 *residual,
                                            const int32_t *weights,
                                            const float *sum,
                                            const float *sumsq,
                                            bfloat16 *output,
                                            const int32_t cols,
                                            const int32_t col_idx)
{
    constexpr int r = 8;
    constexpr int s = 8;

    static_assert(DIM_M % r == 0);
    static_assert(DIM_K % s == 0);

    ::aie::set_rounding(aie::rounding_mode::conv_even);
    fused_add_layer_norm_1_fp32weights<bfloat16, (DIM_M / r), (DIM_K / s), r, s>(
        input, residual, weights, sum, sumsq, output, cols, col_idx);
}

#ifdef BUILD_ADDNORM_REPLAY_FASTPATH
void add_1outs_from_inputs(const bfloat16 *input, const bfloat16 *residual, bfloat16 *output);

void packed_add_1outs_from_inputs(const bfloat16 *packed_input_and_residual, bfloat16 *output)
{
    constexpr int packed_tile_elems = DIM_M * DIM_K;
    add_1outs_from_inputs(packed_input_and_residual, packed_input_and_residual + packed_tile_elems, output);
}

void add_1outs_from_inputs(const bfloat16 *input, const bfloat16 *residual, bfloat16 *output)
{
    constexpr int r = 8;
    constexpr int s = 8;

    static_assert(DIM_M % r == 0);
    static_assert(DIM_K % s == 0);

    ::aie::set_rounding(aie::rounding_mode::conv_even);
    add_1_from_inputs<bfloat16, (DIM_M / r), (DIM_K / s), r, s>(input, residual, output);
}

void fused_add_layer_norm_1outs_from_inputs(const bfloat16 *input,
                                            const bfloat16 *residual,
                                            const bfloat16 *weights,
                                            const float *sum,
                                            const float *sumsq,
                                            bfloat16 *output,
                                            const int32_t cols,
                                            const int32_t col_idx);

void packed_fused_add_layer_norm_1outs_from_inputs(const bfloat16 *packed_input_and_residual,
                                                   const bfloat16 *weights,
                                                   const float *sum,
                                                   const float *sumsq,
                                                   bfloat16 *output,
                                                   const int32_t cols,
                                                   const int32_t col_idx)
{
    constexpr int packed_tile_elems = DIM_M * DIM_K;
    fused_add_layer_norm_1outs_from_inputs(packed_input_and_residual,
                                           packed_input_and_residual + packed_tile_elems,
                                           weights,
                                           sum,
                                           sumsq,
                                           output,
                                           cols,
                                           col_idx);
}

void fused_add_layer_norm_1outs_from_inputs(const bfloat16 *input,
                                            const bfloat16 *residual,
                                            const bfloat16 *weights,
                                            const float *sum,
                                            const float *sumsq,
                                            bfloat16 *output,
                                            const int32_t cols,
                                            const int32_t col_idx)
{
    constexpr int r = 8;
    constexpr int s = 8;

    static_assert(DIM_M % r == 0);
    static_assert(DIM_K % s == 0);

    ::aie::set_rounding(aie::rounding_mode::conv_even);
    fused_add_layer_norm_1_from_inputs<bfloat16, (DIM_M / r), (DIM_K / s), r, s>(
        input, residual, weights, sum, sumsq, output, cols, col_idx);
}

void fused_add_layer_norm_1outs_from_inputs_fp32weights(const bfloat16 *input,
                                                        const bfloat16 *residual,
                                                        const int32_t *weights,
                                                        const float *sum,
                                                        const float *sumsq,
                                                        bfloat16 *output,
                                                        const int32_t cols,
                                                        const int32_t col_idx)
{
    constexpr int r = 8;
    constexpr int s = 8;

    static_assert(DIM_M % r == 0);
    static_assert(DIM_K % s == 0);

    ::aie::set_rounding(aie::rounding_mode::conv_even);
    fused_add_layer_norm_1_from_inputs_fp32weights<bfloat16, (DIM_M / r), (DIM_K / s), r, s>(
        input, residual, weights, sum, sumsq, output, cols, col_idx);
}

void fused_add_layer_norm_2outs_from_inputs(const bfloat16 *input,
                                            const bfloat16 *residual,
                                            const bfloat16 *weights,
                                            const float *sum,
                                            const float *sumsq,
                                            bfloat16 *output,
                                            bfloat16 *merged_output,
                                            const int32_t cols,
                                            const int32_t col_idx);

void packed_fused_add_layer_norm_2outs_from_inputs(const bfloat16 *packed_input_and_residual,
                                                   const bfloat16 *weights,
                                                   const float *sum,
                                                   const float *sumsq,
                                                   bfloat16 *output,
                                                   bfloat16 *merged_output,
                                                   const int32_t cols,
                                                   const int32_t col_idx)
{
    constexpr int packed_tile_elems = DIM_M * DIM_K;
    fused_add_layer_norm_2outs_from_inputs(packed_input_and_residual,
                                           packed_input_and_residual + packed_tile_elems,
                                           weights,
                                           sum,
                                           sumsq,
                                           output,
                                           merged_output,
                                           cols,
                                           col_idx);
}

void fused_add_layer_norm_2outs_from_inputs(const bfloat16 *input,
                                            const bfloat16 *residual,
                                            const bfloat16 *weights,
                                            const float *sum,
                                            const float *sumsq,
                                            bfloat16 *output,
                                            bfloat16 *merged_output,
                                            const int32_t cols,
                                            const int32_t col_idx)
{
    constexpr int r = 8;
    constexpr int s = 8;

    static_assert(DIM_M % r == 0);
    static_assert(DIM_K % s == 0);

    ::aie::set_rounding(aie::rounding_mode::conv_even);
    fused_add_layer_norm_2_from_inputs<bfloat16, (DIM_M / r), (DIM_K / s), r, s>(
        input, residual, weights, sum, sumsq, output, merged_output, cols, col_idx);
}
#endif

void fused_layer_norm_1outs(const bfloat16 *input,
                            const float *sum,
                            const float *sumsq,
                            bfloat16 *output,
                            const int32_t cols)
{
    constexpr int r = 8;
    constexpr int s = 8;

    static_assert(DIM_M % r == 0);
    static_assert(DIM_K % s == 0);

    ::aie::set_rounding(aie::rounding_mode::conv_even);
    fused_layer_norm_1<bfloat16, (DIM_M / r), (DIM_K / s), r, s>(input, sum, sumsq, output, cols);
}

void ln_mul_weights_1outs(const bfloat16 *input, const bfloat16 *weights, bfloat16 *output, const int32_t col_idx)
{
    constexpr int r = 8;
    constexpr int s = 8;

    static_assert(DIM_M % r == 0);
    static_assert(DIM_K % s == 0);

    ::aie::set_rounding(aie::rounding_mode::conv_even);
    ln_mul_weights_1<bfloat16, (DIM_M / r), (DIM_K / s), r, s>(input, weights, output, col_idx);
}

void ln_mul_add_1outs(const bfloat16 *input,
                      const bfloat16 *residual,
                      const bfloat16 *weights,
                      bfloat16 *output,
                      const int32_t col_idx)
{
    constexpr int r = 8;
    constexpr int s = 8;

    static_assert(DIM_M % r == 0);
    static_assert(DIM_K % s == 0);

    ::aie::set_rounding(aie::rounding_mode::conv_even);
    ln_mul_add_1<bfloat16, (DIM_M / r), (DIM_K / s), r, s>(input, residual, weights, output, col_idx);
}

void fused_add_layer_norm_2outs(const bfloat16 *input,
                                const bfloat16 *residual,
                                const bfloat16 *weights,
                                bfloat16 *output1,
                                bfloat16 *output2,
                                const int32_t cols,
                                const int32_t rows_to_process)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    fused_add_layer_norm_2<bfloat16, 32>(input, residual, weights, output1, output2, cols, rows_to_process);
}

void ln_zero_bf16(bfloat16 *C, int size)
{
    ln_zero_vectorized<bfloat16, 16>(C, size);
}

void ln_zero_f32(float *C, int size)
{
    ln_zero_vectorized<float, 8>(C, size);
}

void ln_calc_sum_sumsq(const bfloat16 *A, float *pSum, float *pSumSq)
{
    constexpr int r = 8;
    constexpr int s = 8;

    static_assert(DIM_M % r == 0);
    static_assert(DIM_K % s == 0);

    ::aie::set_rounding(aie::rounding_mode::conv_even);

    ln_calc_sum_sumsq_vectorized<bfloat16, (DIM_M / r), (DIM_K / s), r, s>(A, pSum, pSumSq);
}

#ifdef BUILD_ADDNORM_REPLAY_FASTPATH
void ln_add_calc_sum_sumsq(const bfloat16 *A, const bfloat16 *B, float *pSum, float *pSumSq);

void packed_ln_add_calc_sum_sumsq(const bfloat16 *packed_input_and_residual, float *pSum, float *pSumSq)
{
    constexpr int packed_tile_elems = DIM_M * DIM_K;
    ln_add_calc_sum_sumsq(packed_input_and_residual, packed_input_and_residual + packed_tile_elems, pSum, pSumSq);
}

void ln_add_calc_sum_sumsq(const bfloat16 *A, const bfloat16 *B, float *pSum, float *pSumSq)
{
    constexpr int r = 8;
    constexpr int s = 8;

    static_assert(DIM_M % r == 0);
    static_assert(DIM_K % s == 0);

    ::aie::set_rounding(aie::rounding_mode::conv_even);

    ln_add_calc_sum_sumsq_vectorized<bfloat16, (DIM_M / r), (DIM_K / s), r, s>(A, B, pSum, pSumSq);
}
#endif

#endif
}
