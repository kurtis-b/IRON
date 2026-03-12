module {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_1_5 = aie.tile(1, 5)
    %tile_1_2 = aie.tile(1, 2)
    %tile_5_5 = aie.tile(5, 5)
    %tile_6_5 = aie.tile(6, 5)
    %tile_7_5 = aie.tile(7, 5)
    %mem_tile_7_1 = aie.tile(7, 1)
    %shim_noc_tile_5_0 = aie.tile(5, 0)
    %mem_tile_5_1 = aie.tile(5, 1)
    %shim_noc_tile_6_0 = aie.tile(6, 0)
    %mem_tile_6_1 = aie.tile(6, 1)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %mem_tile_1_1 = aie.tile(1, 1)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %mem_tile_3_1 = aie.tile(3, 1)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %shim_noc_tile_7_0 = aie.tile(7, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %mem_tile_2_1 = aie.tile(2, 1)
    %mem_tile_4_1 = aie.tile(4, 1)
    aie.objectfifo @ffnDownOut(%tile_6_5, {%tile_7_5}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @ffnRIn(%mem_tile_7_1, {%tile_7_5}, 12 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @ffnROut(%tile_1_2, {%mem_tile_7_1}, [1 : i32, 12 : i32]) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo.link [@ffnROut] -> [@ffnRIn]([] [0])
    aie.objectfifo @ffnUpOut(%tile_5_5, {%tile_6_5}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @inBDown(%shim_noc_tile_5_0, {%mem_tile_5_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @memBDown(%mem_tile_5_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_6_5}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo.link [@inBDown] -> [@memBDown]([] [0])
    aie.objectfifo @inBUp(%shim_noc_tile_6_0, {%mem_tile_6_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @memBUp(%mem_tile_6_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_5_5}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo.link [@inBUp] -> [@memBUp]([] [0])
    aie.objectfifo @inK(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @memK0(%mem_tile_1_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo.link [@inK] -> [@memK0]([] [0])
    aie.objectfifo @inOW(%shim_noc_tile_3_0, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @memOW0(%mem_tile_3_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo.link [@inOW] -> [@memOW0]([] [0])
    aie.objectfifo @inQ(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @memQ0(%mem_tile_0_1 dimensionsToStream [<size = 4, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo.link [@inQ] -> [@memQ0]([] [0])
    aie.objectfifo @inR(%shim_noc_tile_7_0, {%mem_tile_7_1}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @memR(%mem_tile_7_1 dimensionsToStream [<size = 4, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo.link [@inR] -> [@memR]([] [0])
    aie.objectfifo @inV(%shim_noc_tile_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @memV0(%mem_tile_2_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo.link [@inV] -> [@memV0]([] [0])
    aie.objectfifo @ln1Norm(%tile_1_5, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @ln2Replay(%mem_tile_4_1, {%tile_7_5}, [12 : i32, 1 : i32]) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @ln2ReplayPart(%tile_7_5, {%mem_tile_4_1}, [1 : i32, 12 : i32]) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo.link [@ln2ReplayPart] -> [@ln2Replay]([] [0])
    aie.objectfifo @memA0(%tile_0_2 dimensionsToStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], {%tile_0_3 dimensionsFromStream [<size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @memLN2(%mem_tile_7_1 dimensionsToStream [<size = 4, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_7_0}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @outLN2(%tile_7_5, {%mem_tile_7_1}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo.link [@outLN2] -> [@memLN2]([] [0])
    aie.objectfifo @memP0(%tile_0_3 dimensionsToStream [<size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>], {%tile_0_4 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @outLNBroadcast(%tile_1_2, {%tile_5_5}, [1 : i32, 2 : i32]) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @outOProj0(%tile_0_4, {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @outOProjInput(%tile_0_5, {%tile_1_5}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @scaleOF0(%tile_0_3, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<128xbf16>> 
    func.func private @zero_bf16(memref<32x64xbf16>)
    func.func private @matmul_bf16_bf16_wrapper(memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<2xi32>)
    %idx_buffer_qk_0 = aie.buffer(%tile_0_2) {sym_name = "idx_buffer_qk_0"} : memref<2xi32> = dense<0>
    func.func private @partial_softmax(memref<32x64xbf16>, memref<32x64xbf16>, memref<128xbf16>, memref<2xi32>, bf16, i32, i32, i32, i32)
    func.func private @init_scale_buffer(memref<128xbf16>, i32)
    func.func private @passThroughLine(memref<128xbf16>, memref<128xbf16>, i32)
    %idx_buffer_softmax_0 = aie.buffer(%tile_0_3) {sym_name = "idx_buffer_softmax_0"} : memref<2xi32> = dense<0>
    %scale_buffer_softmax_0 = aie.buffer(%tile_0_3) {sym_name = "scale_buffer_softmax_0"} : memref<128xbf16> = dense<0.000000e+00>
    func.func private @matmul_PV(memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<128xbf16>, i32, i32, memref<2xi32>)
    func.func private @rescale_O(memref<32x64xbf16>, memref<128xbf16>, i32, memref<2xi32>)
    %idx_buffer_pv_0 = aie.buffer(%tile_0_4) {sym_name = "idx_buffer_pv_0"} : memref<2xi32> = dense<0>
    %mem_tile_4_1_0 = aie.tile(4, 1)
    %o_proj_partial_scratch_0 = aie.buffer(%tile_0_5) {sym_name = "o_proj_partial_scratch_0"} : memref<32x64xbf16> 
    %o_proj_zero_scratch_0 = aie.buffer(%tile_0_5) {sym_name = "o_proj_zero_scratch_0"} : memref<32x64xbf16> 
    %o_proj_stats_sum_0 = aie.buffer(%tile_0_5) {sym_name = "o_proj_stats_sum_0"} : memref<32xf32> 
    %o_proj_stats_sumsq_0 = aie.buffer(%tile_0_5) {sym_name = "o_proj_stats_sumsq_0"} : memref<32xf32> 
    func.func private @ln_zero_f32(memref<32xf32>, i32)
    func.func private @ln_calc_sum_sumsq(memref<32x64xbf16>, memref<32xf32>, memref<32xf32>)
    func.func private @pack_stats_f32_to_bf16_packet(memref<32xf32>, memref<32xf32>, memref<32x64xbf16>, i32)
    func.func private @zero_bf16_o_proj(memref<32x64xbf16>)
    func.func private @matmul_with_acc_bf16_bf16_o_proj(memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>)
    func.func private @eltwise_add_bf16_vector_o_proj(memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>, i32)
    func.func private @passThroughLine_o_proj(memref<32x64xbf16>, memref<32x64xbf16>, i32)
    %mem_tile_5_1_1 = aie.tile(5, 1)
    %ln1_norm_sum_buffer = aie.buffer(%tile_1_5) {sym_name = "ln1_norm_sum_buffer"} : memref<32xf32> 
    %ln1_norm_sumsq_buffer = aie.buffer(%tile_1_5) {sym_name = "ln1_norm_sumsq_buffer"} : memref<32xf32> 
    func.func private @fused_layer_norm_1outs(memref<32x64xbf16>, memref<32xf32>, memref<32xf32>, memref<32x64xbf16>, i32)
    func.func private @unpack_stats_bf16_packet_to_f32(memref<32x64xbf16>, memref<32xf32>, memref<32xf32>, i32)
    %static_ln1_weights = aie.buffer(%tile_1_2) {sym_name = "static_ln1_weights"} : memref<768xbf16> = dense<1.000000e+00>
    func.func private @ln_mul_add_1outs(memref<32x64xbf16>, memref<32x64xbf16>, memref<768xbf16>, memref<32x64xbf16>, i32)
    func.func private @ffn_zero_bf16_up_proj(memref<32x64xbf16>)
    func.func private @ffn_matmul_init_bf16_bf16_up_proj(memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>)
    func.func private @ffn_matmul_bf16_bf16_up_proj(memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>)
    func.func private @ffn_gelu_bf16(memref<32x64xbf16>, memref<32x64xbf16>, i32)
    %mem_tile_6_1_2 = aie.tile(6, 1)
    func.func private @ffn_zero_bf16_down_proj(memref<32x64xbf16>)
    func.func private @ffn_matmul_init_bf16_bf16_down_proj(memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>)
    func.func private @ffn_matmul_with_acc_bf16_bf16_down_proj(memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>)
    %ln2_sum_buffer = aie.buffer(%tile_7_5) {sym_name = "ln2_sum_buffer"} : memref<32xf32> 
    %ln2_sumsq_buffer = aie.buffer(%tile_7_5) {sym_name = "ln2_sumsq_buffer"} : memref<32xf32> 
    %static_ln2_weights = aie.buffer(%tile_7_5) {sym_name = "static_ln2_weights"} : memref<768xbf16> = dense<1.000000e+00>
    func.func private @fused_add_layer_norm_1outs(memref<32x64xbf16>, memref<32x64xbf16>, memref<768xbf16>, memref<32xf32>, memref<32xf32>, memref<32x64xbf16>, i32, i32)
    %outOProjAccum0_src = aie.buffer(%tile_0_5) {sym_name = "outOProjAccum0_src"} : memref<32x64xbf16> 
    %outOProjAccum0_dst = aie.buffer(%tile_0_5) {sym_name = "outOProjAccum0_dst"} : memref<32x64xbf16> 
    %outOProjAccum0_src_empty = aie.lock(%tile_0_5) {init = 1 : i32, sym_name = "outOProjAccum0_src_empty"}
    %outOProjAccum0_src_full = aie.lock(%tile_0_5) {init = 0 : i32, sym_name = "outOProjAccum0_src_full"}
    %outOProjAccum0_dst_empty = aie.lock(%tile_0_5) {init = 1 : i32, sym_name = "outOProjAccum0_dst_empty"}
    %outOProjAccum0_dst_full = aie.lock(%tile_0_5) {init = 0 : i32, sym_name = "outOProjAccum0_dst_full"}
    %outOProjAccum0_row_0 = aie.buffer(%mem_tile_4_1_0) {sym_name = "outOProjAccum0_row_0"} : memref<24576xbf16> 
    %outOProjAccum0_row_0_empty = aie.lock(%mem_tile_4_1_0) {init = 1 : i32, sym_name = "outOProjAccum0_row_0_empty"}
    %outOProjAccum0_row_0_full = aie.lock(%mem_tile_4_1_0) {init = 0 : i32, sym_name = "outOProjAccum0_row_0_full"}
    %outOProjAccum0_row_1 = aie.buffer(%mem_tile_4_1_0) {sym_name = "outOProjAccum0_row_1"} : memref<24576xbf16> 
    %outOProjAccum0_row_1_empty = aie.lock(%mem_tile_4_1_0) {init = 1 : i32, sym_name = "outOProjAccum0_row_1_empty"}
    %outOProjAccum0_row_1_full = aie.lock(%mem_tile_4_1_0) {init = 0 : i32, sym_name = "outOProjAccum0_row_1_full"}
    aie.flow(%tile_0_5, DMA : 0, %mem_tile_4_1_0, DMA : 2)
    aie.flow(%mem_tile_4_1_0, DMA : 2, %tile_0_5, DMA : 1)
    %ln1Replay_src = aie.buffer(%tile_1_5) {sym_name = "ln1Replay_src"} : memref<32x64xbf16> 
    %ln1Replay_dst = aie.buffer(%tile_1_5) {sym_name = "ln1Replay_dst"} : memref<32x64xbf16> 
    %ln1Replay_src_empty = aie.lock(%tile_1_5) {init = 1 : i32, sym_name = "ln1Replay_src_empty"}
    %ln1Replay_src_full = aie.lock(%tile_1_5) {init = 0 : i32, sym_name = "ln1Replay_src_full"}
    %ln1Replay_dst_empty = aie.lock(%tile_1_5) {init = 1 : i32, sym_name = "ln1Replay_dst_empty"}
    %ln1Replay_dst_full = aie.lock(%tile_1_5) {init = 0 : i32, sym_name = "ln1Replay_dst_full"}
    %ln1Replay_row_0 = aie.buffer(%mem_tile_5_1_1) {sym_name = "ln1Replay_row_0"} : memref<24576xbf16> 
    %ln1Replay_row_0_empty = aie.lock(%mem_tile_5_1_1) {init = 1 : i32, sym_name = "ln1Replay_row_0_empty"}
    %ln1Replay_row_0_full = aie.lock(%mem_tile_5_1_1) {init = 0 : i32, sym_name = "ln1Replay_row_0_full"}
    %ln1Replay_row_1 = aie.buffer(%mem_tile_5_1_1) {sym_name = "ln1Replay_row_1"} : memref<24576xbf16> 
    %ln1Replay_row_1_empty = aie.lock(%mem_tile_5_1_1) {init = 1 : i32, sym_name = "ln1Replay_row_1_empty"}
    %ln1Replay_row_1_full = aie.lock(%mem_tile_5_1_1) {init = 0 : i32, sym_name = "ln1Replay_row_1_full"}
    aie.flow(%tile_1_5, DMA : 0, %mem_tile_5_1_1, DMA : 0)
    aie.flow(%mem_tile_5_1_1, DMA : 1, %tile_1_5, DMA : 0)
    %ffnDownAccum_src = aie.buffer(%tile_6_5) {sym_name = "ffnDownAccum_src"} : memref<32x64xbf16> 
    %ffnDownAccum_dst = aie.buffer(%tile_6_5) {sym_name = "ffnDownAccum_dst"} : memref<32x64xbf16> 
    %ffnDownAccum_src_empty = aie.lock(%tile_6_5) {init = 1 : i32, sym_name = "ffnDownAccum_src_empty"}
    %ffnDownAccum_src_full = aie.lock(%tile_6_5) {init = 0 : i32, sym_name = "ffnDownAccum_src_full"}
    %ffnDownAccum_dst_empty = aie.lock(%tile_6_5) {init = 1 : i32, sym_name = "ffnDownAccum_dst_empty"}
    %ffnDownAccum_dst_full = aie.lock(%tile_6_5) {init = 0 : i32, sym_name = "ffnDownAccum_dst_full"}
    %ffnDownAccum_row_0 = aie.buffer(%mem_tile_6_1_2) {sym_name = "ffnDownAccum_row_0"} : memref<24576xbf16> 
    %ffnDownAccum_row_0_empty = aie.lock(%mem_tile_6_1_2) {init = 1 : i32, sym_name = "ffnDownAccum_row_0_empty"}
    %ffnDownAccum_row_0_full = aie.lock(%mem_tile_6_1_2) {init = 0 : i32, sym_name = "ffnDownAccum_row_0_full"}
    %ffnDownAccum_row_1 = aie.buffer(%mem_tile_6_1_2) {sym_name = "ffnDownAccum_row_1"} : memref<24576xbf16> 
    %ffnDownAccum_row_1_empty = aie.lock(%mem_tile_6_1_2) {init = 1 : i32, sym_name = "ffnDownAccum_row_1_empty"}
    %ffnDownAccum_row_1_full = aie.lock(%mem_tile_6_1_2) {init = 0 : i32, sym_name = "ffnDownAccum_row_1_full"}
    aie.flow(%tile_6_5, DMA : 0, %mem_tile_6_1_2, DMA : 2)
    aie.flow(%mem_tile_6_1_2, DMA : 2, %tile_6_5, DMA : 0)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_3 = arith.constant 0 : index
        %c0_i32 = arith.constant 0 : i32
        memref.store %c0_i32, %idx_buffer_qk_0[%c0_3] : memref<2xi32>
        %c1_4 = arith.constant 1 : index
        %c0_i32_5 = arith.constant 0 : i32
        memref.store %c0_i32_5, %idx_buffer_qk_0[%c1_4] : memref<2xi32>
        %c0_6 = arith.constant 0 : index
        %c12 = arith.constant 12 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %c12 step %c1_7 {
          %0 = aie.objectfifo.acquire @memQ0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          %c1_10 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %c1_9 step %c1_10 {
            %4 = aie.objectfifo.acquire @memK0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %6 = aie.objectfifo.acquire @memA0(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @zero_bf16(%7) : (memref<32x64xbf16>) -> ()
            func.call @matmul_bf16_bf16_wrapper(%1, %5, %7, %idx_buffer_qk_0) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<2xi32>) -> ()
            aie.objectfifo.release @memK0(Consume, 1)
            aie.objectfifo.release @memA0(Produce, 1)
            %c0_16 = arith.constant 0 : index
            %8 = memref.load %idx_buffer_qk_0[%c0_16] : memref<2xi32>
            %c0_i32_17 = arith.constant 0 : i32
            %9 = arith.addi %8, %c0_i32_17 : i32
            %c0_18 = arith.constant 0 : index
            memref.store %9, %idx_buffer_qk_0[%c0_18] : memref<2xi32>
          }
          %c0_11 = arith.constant 0 : index
          %c0_i32_12 = arith.constant 0 : i32
          memref.store %c0_i32_12, %idx_buffer_qk_0[%c0_11] : memref<2xi32>
          %c1_13 = arith.constant 1 : index
          %2 = memref.load %idx_buffer_qk_0[%c1_13] : memref<2xi32>
          %c0_i32_14 = arith.constant 0 : i32
          %3 = arith.addi %2, %c0_i32_14 : i32
          %c1_15 = arith.constant 1 : index
          memref.store %3, %idx_buffer_qk_0[%c1_15] : memref<2xi32>
          aie.objectfifo.release @memQ0(Consume, 1)
        }
      }
      aie.end
    } {link_with = "ep_12h_64s_64e_1ph_12pa_1g_1pf_d-1_m0_fa_n1-1_n2-1_me_150ff6c44c7b_kernels.a", stack_size = 3328 : i32}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_3 = arith.constant 0 : index
        %c0_i32 = arith.constant 0 : i32
        memref.store %c0_i32, %idx_buffer_softmax_0[%c0_3] : memref<2xi32>
        %c1_4 = arith.constant 1 : index
        %c0_i32_5 = arith.constant 0 : i32
        memref.store %c0_i32_5, %idx_buffer_softmax_0[%c1_4] : memref<2xi32>
        %c0_6 = arith.constant 0 : index
        %c12 = arith.constant 12 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %c12 step %c1_7 {
          %c32_i32 = arith.constant 32 : i32
          func.call @init_scale_buffer(%scale_buffer_softmax_0, %c32_i32) : (memref<128xbf16>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          %c1_10 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %c1_9 step %c1_10 {
            %2 = aie.objectfifo.acquire @memP0(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            %4 = aie.objectfifo.acquire @memA0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            %6 = aie.objectfifo.acquire @scaleOF0(Produce, 1) : !aie.objectfifosubview<memref<128xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
            %cst = arith.constant 1.806640e-01 : bf16
            %c32_i32_16 = arith.constant 32 : i32
            %c64_i32 = arith.constant 64 : i32
            %c64_i32_17 = arith.constant 64 : i32
            %c64_i32_18 = arith.constant 64 : i32
            func.call @partial_softmax(%5, %3, %scale_buffer_softmax_0, %idx_buffer_softmax_0, %cst, %c32_i32_16, %c64_i32, %c64_i32_17, %c64_i32_18) : (memref<32x64xbf16>, memref<32x64xbf16>, memref<128xbf16>, memref<2xi32>, bf16, i32, i32, i32, i32) -> ()
            %c128_i32 = arith.constant 128 : i32
            func.call @passThroughLine(%scale_buffer_softmax_0, %7, %c128_i32) : (memref<128xbf16>, memref<128xbf16>, i32) -> ()
            aie.objectfifo.release @memA0(Consume, 1)
            aie.objectfifo.release @memP0(Produce, 1)
            aie.objectfifo.release @scaleOF0(Produce, 1)
            %c0_19 = arith.constant 0 : index
            %8 = memref.load %idx_buffer_softmax_0[%c0_19] : memref<2xi32>
            %c0_i32_20 = arith.constant 0 : i32
            %9 = arith.addi %8, %c0_i32_20 : i32
            %c0_21 = arith.constant 0 : index
            memref.store %9, %idx_buffer_softmax_0[%c0_21] : memref<2xi32>
          }
          %c0_11 = arith.constant 0 : index
          %c0_i32_12 = arith.constant 0 : i32
          memref.store %c0_i32_12, %idx_buffer_softmax_0[%c0_11] : memref<2xi32>
          %c1_13 = arith.constant 1 : index
          %0 = memref.load %idx_buffer_softmax_0[%c1_13] : memref<2xi32>
          %c0_i32_14 = arith.constant 0 : i32
          %1 = arith.addi %0, %c0_i32_14 : i32
          %c1_15 = arith.constant 1 : index
          memref.store %1, %idx_buffer_softmax_0[%c1_15] : memref<2xi32>
        }
      }
      aie.end
    } {link_with = "ep_12h_64s_64e_1ph_12pa_1g_1pf_d-1_m0_fa_n1-1_n2-1_me_150ff6c44c7b_kernels.a", stack_size = 3328 : i32}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_3 = arith.constant 0 : index
        %c0_i32 = arith.constant 0 : i32
        memref.store %c0_i32, %idx_buffer_pv_0[%c0_3] : memref<2xi32>
        %c1_4 = arith.constant 1 : index
        %c0_i32_5 = arith.constant 0 : i32
        memref.store %c0_i32_5, %idx_buffer_pv_0[%c1_4] : memref<2xi32>
        %c0_6 = arith.constant 0 : index
        %c12 = arith.constant 12 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %c12 step %c1_7 {
          %0 = aie.objectfifo.acquire @outOProj0(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @zero_bf16(%1) : (memref<32x64xbf16>) -> ()
          %2 = aie.objectfifo.acquire @memP0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %4 = aie.objectfifo.acquire @memV0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
          %6 = aie.objectfifo.acquire @scaleOF0(Consume, 1) : !aie.objectfifosubview<memref<128xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
          %c32_i32 = arith.constant 32 : i32
          %c0_i32_8 = arith.constant 0 : i32
          func.call @matmul_PV(%3, %5, %1, %7, %c32_i32, %c0_i32_8, %idx_buffer_pv_0) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<128xbf16>, i32, i32, memref<2xi32>) -> ()
          aie.objectfifo.release @memP0(Consume, 1)
          aie.objectfifo.release @memV0(Consume, 1)
          aie.objectfifo.release @scaleOF0(Consume, 1)
          %c0_9 = arith.constant 0 : index
          %8 = memref.load %idx_buffer_pv_0[%c0_9] : memref<2xi32>
          %c0_i32_10 = arith.constant 0 : i32
          %9 = arith.addi %8, %c0_i32_10 : i32
          %c0_11 = arith.constant 0 : index
          memref.store %9, %idx_buffer_pv_0[%c0_11] : memref<2xi32>
          %c32_i32_12 = arith.constant 32 : i32
          func.call @rescale_O(%1, %7, %c32_i32_12, %idx_buffer_pv_0) : (memref<32x64xbf16>, memref<128xbf16>, i32, memref<2xi32>) -> ()
          %c0_13 = arith.constant 0 : index
          %10 = memref.load %idx_buffer_pv_0[%c0_13] : memref<2xi32>
          %c0_i32_14 = arith.constant 0 : i32
          %11 = arith.addi %10, %c0_i32_14 : i32
          %c0_15 = arith.constant 0 : index
          memref.store %11, %idx_buffer_pv_0[%c0_15] : memref<2xi32>
          %c0_16 = arith.constant 0 : index
          %c0_i32_17 = arith.constant 0 : i32
          memref.store %c0_i32_17, %idx_buffer_pv_0[%c0_16] : memref<2xi32>
          %c1_18 = arith.constant 1 : index
          %12 = memref.load %idx_buffer_pv_0[%c1_18] : memref<2xi32>
          %c0_i32_19 = arith.constant 0 : i32
          %13 = arith.addi %12, %c0_i32_19 : i32
          %c1_20 = arith.constant 1 : index
          memref.store %13, %idx_buffer_pv_0[%c1_20] : memref<2xi32>
          aie.objectfifo.release @outOProj0(Produce, 1)
        }
      }
      aie.end
    } {link_with = "ep_12h_64s_64e_1ph_12pa_1g_1pf_d-1_m0_fa_n1-1_n2-1_me_150ff6c44c7b_kernels.a", stack_size = 3328 : i32}
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c32_i32 = arith.constant 32 : i32
        func.call @ln_zero_f32(%o_proj_stats_sum_0, %c32_i32) : (memref<32xf32>, i32) -> ()
        %c32_i32_3 = arith.constant 32 : i32
        func.call @ln_zero_f32(%o_proj_stats_sumsq_0, %c32_i32_3) : (memref<32xf32>, i32) -> ()
        %c0_4 = arith.constant 0 : index
        %c12 = arith.constant 12 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c12 step %c1_5 {
          aie.use_lock(%outOProjAccum0_src_empty, AcquireGreaterEqual, 1)
          func.call @zero_bf16_o_proj(%outOProjAccum0_src) : (memref<32x64xbf16>) -> ()
          aie.use_lock(%outOProjAccum0_src_full, Release, 1)
        }
        %c0_6 = arith.constant 0 : index
        %c12_7 = arith.constant 12 : index
        %c1_8 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %c12_7 step %c1_8 {
          %2 = aie.objectfifo.acquire @outOProj0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %c0_13 = arith.constant 0 : index
          %c12_14 = arith.constant 12 : index
          %c1_15 = arith.constant 1 : index
          scf.for %arg2 = %c0_13 to %c12_14 step %c1_15 {
            aie.use_lock(%outOProjAccum0_dst_full, AcquireGreaterEqual, 1)
            %4 = aie.objectfifo.acquire @memOW0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.use_lock(%outOProjAccum0_src_empty, AcquireGreaterEqual, 1)
            func.call @matmul_with_acc_bf16_bf16_o_proj(%3, %5, %outOProjAccum0_dst, %outOProjAccum0_src) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
            aie.use_lock(%outOProjAccum0_src_full, Release, 1)
            aie.objectfifo.release @memOW0(Consume, 1)
            aie.use_lock(%outOProjAccum0_dst_empty, Release, 1)
          }
          aie.objectfifo.release @outOProj0(Consume, 1)
        }
        %c0_9 = arith.constant 0 : index
        %c12_10 = arith.constant 12 : index
        %c1_11 = arith.constant 1 : index
        scf.for %arg1 = %c0_9 to %c12_10 step %c1_11 {
          aie.use_lock(%outOProjAccum0_dst_full, AcquireGreaterEqual, 1)
          func.call @ln_calc_sum_sumsq(%outOProjAccum0_dst, %o_proj_stats_sum_0, %o_proj_stats_sumsq_0) : (memref<32x64xbf16>, memref<32xf32>, memref<32xf32>) -> ()
          %2 = aie.objectfifo.acquire @outOProjInput(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %c2048_i32 = arith.constant 2048 : i32
          func.call @passThroughLine_o_proj(%outOProjAccum0_dst, %3, %c2048_i32) : (memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
          aie.objectfifo.release @outOProjInput(Produce, 1)
          aie.use_lock(%outOProjAccum0_dst_empty, Release, 1)
        }
        %0 = aie.objectfifo.acquire @outOProjInput(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        %c32_i32_12 = arith.constant 32 : i32
        func.call @pack_stats_f32_to_bf16_packet(%o_proj_stats_sum_0, %o_proj_stats_sumsq_0, %1, %c32_i32_12) : (memref<32xf32>, memref<32xf32>, memref<32x64xbf16>, i32) -> ()
        aie.objectfifo.release @outOProjInput(Produce, 1)
      }
      aie.end
    } {link_with = "ep_12h_64s_64e_1ph_12pa_1g_1pf_d-1_m0_fa_n1-1_n2-1_me_150ff6c44c7b_kernels.a", stack_size = 3328 : i32}
    %core_1_5 = aie.core(%tile_1_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_3 = arith.constant 0 : index
        %c12 = arith.constant 12 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_3 to %c12 step %c1_4 {
          %2 = aie.objectfifo.acquire @outOProjInput(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          aie.use_lock(%ln1Replay_src_empty, AcquireGreaterEqual, 1)
          %c2048_i32 = arith.constant 2048 : i32
          func.call @passThroughLine_o_proj(%3, %ln1Replay_src, %c2048_i32) : (memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
          aie.use_lock(%ln1Replay_src_full, Release, 1)
          aie.objectfifo.release @outOProjInput(Consume, 1)
        }
        %0 = aie.objectfifo.acquire @outOProjInput(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        %c32_i32 = arith.constant 32 : i32
        func.call @unpack_stats_bf16_packet_to_f32(%1, %ln1_norm_sum_buffer, %ln1_norm_sumsq_buffer, %c32_i32) : (memref<32x64xbf16>, memref<32xf32>, memref<32xf32>, i32) -> ()
        aie.objectfifo.release @outOProjInput(Consume, 1)
        %c0_5 = arith.constant 0 : index
        %c12_6 = arith.constant 12 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_5 to %c12_6 step %c1_7 {
          aie.use_lock(%ln1Replay_dst_full, AcquireGreaterEqual, 1)
          aie.use_lock(%ln1Replay_src_empty, AcquireGreaterEqual, 1)
          %c768_i32 = arith.constant 768 : i32
          func.call @fused_layer_norm_1outs(%ln1Replay_dst, %ln1_norm_sum_buffer, %ln1_norm_sumsq_buffer, %ln1Replay_src, %c768_i32) : (memref<32x64xbf16>, memref<32xf32>, memref<32xf32>, memref<32x64xbf16>, i32) -> ()
          %2 = aie.objectfifo.acquire @ln1Norm(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %c2048_i32 = arith.constant 2048 : i32
          func.call @passThroughLine_o_proj(%ln1Replay_src, %3, %c2048_i32) : (memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
          aie.objectfifo.release @ln1Norm(Produce, 1)
          aie.use_lock(%ln1Replay_src_full, Release, 1)
          aie.use_lock(%ln1Replay_dst_empty, Release, 1)
        }
        %c0_8 = arith.constant 0 : index
        %c46 = arith.constant 46 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_8 to %c46 step %c1_9 {
          %c0_13 = arith.constant 0 : index
          %c12_14 = arith.constant 12 : index
          %c1_15 = arith.constant 1 : index
          scf.for %arg2 = %c0_13 to %c12_14 step %c1_15 {
            aie.use_lock(%ln1Replay_dst_full, AcquireGreaterEqual, 1)
            %2 = aie.objectfifo.acquire @ln1Norm(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            %c2048_i32 = arith.constant 2048 : i32
            func.call @passThroughLine_o_proj(%ln1Replay_dst, %3, %c2048_i32) : (memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
            aie.objectfifo.release @ln1Norm(Produce, 1)
            aie.use_lock(%ln1Replay_src_empty, AcquireGreaterEqual, 1)
            %c2048_i32_16 = arith.constant 2048 : i32
            func.call @passThroughLine_o_proj(%ln1Replay_dst, %ln1Replay_src, %c2048_i32_16) : (memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
            aie.use_lock(%ln1Replay_src_full, Release, 1)
            aie.use_lock(%ln1Replay_dst_empty, Release, 1)
          }
        }
        %c0_10 = arith.constant 0 : index
        %c12_11 = arith.constant 12 : index
        %c1_12 = arith.constant 1 : index
        scf.for %arg1 = %c0_10 to %c12_11 step %c1_12 {
          aie.use_lock(%ln1Replay_dst_full, AcquireGreaterEqual, 1)
          %2 = aie.objectfifo.acquire @ln1Norm(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %c2048_i32 = arith.constant 2048 : i32
          func.call @passThroughLine_o_proj(%ln1Replay_dst, %3, %c2048_i32) : (memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
          aie.objectfifo.release @ln1Norm(Produce, 1)
          aie.use_lock(%ln1Replay_dst_empty, Release, 1)
        }
      }
      aie.end
    } {link_with = "ep_12h_64s_64e_1ph_12pa_1g_1pf_d-1_m0_fa_n1-1_n2-1_me_150ff6c44c7b_kernels.a"}
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_3 = arith.constant 0 : index
        %c12 = arith.constant 12 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_3 to %c12 step %c1_4 {
          %0 = index.casts %arg1 : index to i32
          %1 = aie.objectfifo.acquire @ln1Norm(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %2 = aie.objectfifo.subview.access %1[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %3 = aie.objectfifo.acquire @memR(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %5 = aie.objectfifo.acquire @outLNBroadcast(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @ln_mul_add_1outs(%2, %4, %static_ln1_weights, %6, %0) : (memref<32x64xbf16>, memref<32x64xbf16>, memref<768xbf16>, memref<32x64xbf16>, i32) -> ()
          %7 = aie.objectfifo.acquire @ffnROut(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %c2048_i32 = arith.constant 2048 : i32
          func.call @passThroughLine_o_proj(%6, %8, %c2048_i32) : (memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
          aie.objectfifo.release @ffnROut(Produce, 1)
          aie.objectfifo.release @outLNBroadcast(Produce, 1)
          aie.objectfifo.release @ln1Norm(Consume, 1)
          aie.objectfifo.release @memR(Consume, 1)
        }
        %c0_5 = arith.constant 0 : index
        %c47 = arith.constant 47 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_5 to %c47 step %c1_6 {
          %c0_10 = arith.constant 0 : index
          %c12_11 = arith.constant 12 : index
          %c1_12 = arith.constant 1 : index
          scf.for %arg2 = %c0_10 to %c12_11 step %c1_12 {
            %0 = index.casts %arg2 : index to i32
            %1 = aie.objectfifo.acquire @ln1Norm(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %2 = aie.objectfifo.subview.access %1[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            %3 = aie.objectfifo.acquire @memR(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            %5 = aie.objectfifo.acquire @outLNBroadcast(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @ln_mul_add_1outs(%2, %4, %static_ln1_weights, %6, %0) : (memref<32x64xbf16>, memref<32x64xbf16>, memref<768xbf16>, memref<32x64xbf16>, i32) -> ()
            aie.objectfifo.release @outLNBroadcast(Produce, 1)
            aie.objectfifo.release @ln1Norm(Consume, 1)
            aie.objectfifo.release @memR(Consume, 1)
          }
        }
        %c0_7 = arith.constant 0 : index
        %c0_8 = arith.constant 0 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c0_8 step %c1_9 {
          %c0_10 = arith.constant 0 : index
          %c12_11 = arith.constant 12 : index
          %c1_12 = arith.constant 1 : index
          scf.for %arg2 = %c0_10 to %c12_11 step %c1_12 {
            %0 = aie.objectfifo.acquire @ln1Norm(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            %2 = aie.objectfifo.acquire @memR(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            aie.objectfifo.release @ln1Norm(Consume, 1)
            aie.objectfifo.release @memR(Consume, 1)
          }
        }
      }
      aie.end
    } {link_with = "ep_12h_64s_64e_1ph_12pa_1g_1pf_d-1_m0_fa_n1-1_n2-1_me_150ff6c44c7b_kernels.a"}
    %core_5_5 = aie.core(%tile_5_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_3 = arith.constant 0 : index
        %c48 = arith.constant 48 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_3 to %c48 step %c1_4 {
          %0 = index.casts %arg1 : index to i32
          %c48_i32 = arith.constant 48 : i32
          %1 = arith.cmpi slt, %0, %c48_i32 : i32
          scf.if %1 {
            %2 = aie.objectfifo.acquire @ffnUpOut(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            %c0_5 = arith.constant 0 : index
            %c12 = arith.constant 12 : index
            %c1_6 = arith.constant 1 : index
            scf.for %arg2 = %c0_5 to %c12 step %c1_6 {
              %4 = aie.objectfifo.acquire @outLNBroadcast(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
              %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
              %6 = aie.objectfifo.acquire @memBUp(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %8 = index.casts %arg2 : index to i32
              %c0_i32 = arith.constant 0 : i32
              %9 = arith.cmpi eq, %8, %c0_i32 : i32
              scf.if %9 {
                func.call @ffn_matmul_init_bf16_bf16_up_proj(%5, %7, %3) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
              } else {
                func.call @ffn_matmul_bf16_bf16_up_proj(%5, %7, %3) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
              }
              aie.objectfifo.release @memBUp(Consume, 1)
              aie.objectfifo.release @outLNBroadcast(Consume, 1)
            }
            %c2048_i32 = arith.constant 2048 : i32
            func.call @ffn_gelu_bf16(%3, %3, %c2048_i32) : (memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
            aie.objectfifo.release @ffnUpOut(Produce, 1)
          } else {
            %c0_5 = arith.constant 0 : index
            %c12 = arith.constant 12 : index
            %c1_6 = arith.constant 1 : index
            scf.for %arg2 = %c0_5 to %c12 step %c1_6 {
              %2 = aie.objectfifo.acquire @outLNBroadcast(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
              %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
              aie.objectfifo.release @outLNBroadcast(Consume, 1)
            }
          }
        }
      }
      aie.end
    } {link_with = "ep_12h_64s_64e_1ph_12pa_1g_1pf_d-1_m0_fa_n1-1_n2-1_me_150ff6c44c7b_kernels.a", stack_size = 1792 : i32}
    %core_6_5 = aie.core(%tile_6_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @ffnUpOut(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        %c0_3 = arith.constant 0 : index
        %c12 = arith.constant 12 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_3 to %c12 step %c1_4 {
          aie.use_lock(%ffnDownAccum_src_empty, AcquireGreaterEqual, 1)
          %2 = aie.objectfifo.acquire @memBDown(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
          func.call @ffn_matmul_init_bf16_bf16_down_proj(%1, %3, %ffnDownAccum_src) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
          aie.objectfifo.release @memBDown(Consume, 1)
          aie.use_lock(%ffnDownAccum_src_full, Release, 1)
        }
        aie.objectfifo.release @ffnUpOut(Consume, 1)
        %c0_5 = arith.constant 0 : index
        %c47 = arith.constant 47 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_5 to %c47 step %c1_6 {
          %2 = aie.objectfifo.acquire @ffnUpOut(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %c0_10 = arith.constant 0 : index
          %c12_11 = arith.constant 12 : index
          %c1_12 = arith.constant 1 : index
          scf.for %arg2 = %c0_10 to %c12_11 step %c1_12 {
            aie.use_lock(%ffnDownAccum_dst_full, AcquireGreaterEqual, 1)
            aie.use_lock(%ffnDownAccum_src_empty, AcquireGreaterEqual, 1)
            %4 = aie.objectfifo.acquire @memBDown(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            func.call @ffn_matmul_with_acc_bf16_bf16_down_proj(%3, %5, %ffnDownAccum_dst, %ffnDownAccum_src) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
            aie.objectfifo.release @memBDown(Consume, 1)
            aie.use_lock(%ffnDownAccum_src_full, Release, 1)
            aie.use_lock(%ffnDownAccum_dst_empty, Release, 1)
          }
          aie.objectfifo.release @ffnUpOut(Consume, 1)
        }
        %c0_7 = arith.constant 0 : index
        %c12_8 = arith.constant 12 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c12_8 step %c1_9 {
          %2 = aie.objectfifo.acquire @ffnDownOut(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          aie.use_lock(%ffnDownAccum_dst_full, AcquireGreaterEqual, 1)
          %c2048_i32 = arith.constant 2048 : i32
          func.call @passThroughLine_o_proj(%ffnDownAccum_dst, %3, %c2048_i32) : (memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
          aie.use_lock(%ffnDownAccum_dst_empty, Release, 1)
          aie.objectfifo.release @ffnDownOut(Produce, 1)
        }
      }
      aie.end
    } {link_with = "ep_12h_64s_64e_1ph_12pa_1g_1pf_d-1_m0_fa_n1-1_n2-1_me_150ff6c44c7b_kernels.a", stack_size = 3840 : i32}
    %core_7_5 = aie.core(%tile_7_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c32_i32 = arith.constant 32 : i32
        func.call @ln_zero_f32(%ln2_sum_buffer, %c32_i32) : (memref<32xf32>, i32) -> ()
        %c32_i32_3 = arith.constant 32 : i32
        func.call @ln_zero_f32(%ln2_sumsq_buffer, %c32_i32_3) : (memref<32xf32>, i32) -> ()
        %c0_4 = arith.constant 0 : index
        %c12 = arith.constant 12 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c12 step %c1_5 {
          %0 = aie.objectfifo.acquire @ffnDownOut(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @ln_calc_sum_sumsq(%1, %ln2_sum_buffer, %ln2_sumsq_buffer) : (memref<32x64xbf16>, memref<32xf32>, memref<32xf32>) -> ()
          %2 = aie.objectfifo.acquire @ln2ReplayPart(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %c2048_i32 = arith.constant 2048 : i32
          func.call @passThroughLine_o_proj(%1, %3, %c2048_i32) : (memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
          aie.objectfifo.release @ln2ReplayPart(Produce, 1)
          aie.objectfifo.release @ffnDownOut(Consume, 1)
        }
        %c0_6 = arith.constant 0 : index
        %c12_7 = arith.constant 12 : index
        %c1_8 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %c12_7 step %c1_8 {
          %0 = index.casts %arg1 : index to i32
          %1 = aie.objectfifo.acquire @ln2Replay(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %2 = aie.objectfifo.subview.access %1[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %3 = aie.objectfifo.acquire @ffnRIn(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %5 = aie.objectfifo.acquire @outLN2(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %c768_i32 = arith.constant 768 : i32
          func.call @fused_add_layer_norm_1outs(%2, %4, %static_ln2_weights, %ln2_sum_buffer, %ln2_sumsq_buffer, %6, %c768_i32, %0) : (memref<32x64xbf16>, memref<32x64xbf16>, memref<768xbf16>, memref<32xf32>, memref<32xf32>, memref<32x64xbf16>, i32, i32) -> ()
          aie.objectfifo.release @outLN2(Produce, 1)
          aie.objectfifo.release @ln2Replay(Consume, 1)
          aie.objectfifo.release @ffnRIn(Consume, 1)
        }
      }
      aie.end
    } {link_with = "ep_12h_64s_64e_1ph_12pa_1g_1pf_d-1_m0_fa_n1-1_n2-1_me_150ff6c44c7b_kernels.a", stack_size = 3840 : i32}
    aie.runtime_sequence(%arg0: memref<768x768xbf16>, %arg1: memref<192x768xbf16>, %arg2: memref<128x768xbf16>, %arg3: memref<2359296xbf16>, %arg4: memref<2359296xbf16>) {
      %0 = aiex.dma_configure_task_for @inQ {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 0, 24576, [<size = 1, stride = 0>, <size = 12, stride = 64>, <size = 32, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      aiex.dma_free_task(%0)
      %1 = aiex.dma_configure_task_for @inK {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 49152, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @inV {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 98304, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @inOW {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 0, 4096, [<size = 12, stride = 64>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 11 : i32}
      aiex.dma_start_task(%3)
      aiex.dma_await_task(%1)
      aiex.dma_await_task(%2)
      aiex.dma_await_task(%3)
      %4 = aiex.dma_configure_task_for @inK {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 49216, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @inV {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 98368, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @inOW {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 49152, 4096, [<size = 12, stride = 64>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 11 : i32}
      aiex.dma_start_task(%6)
      aiex.dma_await_task(%4)
      aiex.dma_await_task(%5)
      aiex.dma_await_task(%6)
      %7 = aiex.dma_configure_task_for @inK {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 49280, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @inV {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 98432, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @inOW {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 98304, 4096, [<size = 12, stride = 64>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 11 : i32}
      aiex.dma_start_task(%9)
      aiex.dma_await_task(%7)
      aiex.dma_await_task(%8)
      aiex.dma_await_task(%9)
      %10 = aiex.dma_configure_task_for @inK {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 49344, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @inV {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 98496, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @inOW {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 147456, 4096, [<size = 12, stride = 64>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 11 : i32}
      aiex.dma_start_task(%12)
      aiex.dma_await_task(%10)
      aiex.dma_await_task(%11)
      aiex.dma_await_task(%12)
      %13 = aiex.dma_configure_task_for @inK {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 49408, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @inV {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 98560, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @inOW {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 196608, 4096, [<size = 12, stride = 64>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 11 : i32}
      aiex.dma_start_task(%15)
      aiex.dma_await_task(%13)
      aiex.dma_await_task(%14)
      aiex.dma_await_task(%15)
      %16 = aiex.dma_configure_task_for @inK {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 49472, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @inV {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 98624, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @inOW {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 245760, 4096, [<size = 12, stride = 64>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 11 : i32}
      aiex.dma_start_task(%18)
      aiex.dma_await_task(%16)
      aiex.dma_await_task(%17)
      aiex.dma_await_task(%18)
      %19 = aiex.dma_configure_task_for @inK {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 49536, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @inV {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 98688, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @inOW {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 294912, 4096, [<size = 12, stride = 64>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 11 : i32}
      aiex.dma_start_task(%21)
      aiex.dma_await_task(%19)
      aiex.dma_await_task(%20)
      aiex.dma_await_task(%21)
      %22 = aiex.dma_configure_task_for @inK {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 49600, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @inV {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 98752, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @inOW {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 344064, 4096, [<size = 12, stride = 64>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 11 : i32}
      aiex.dma_start_task(%24)
      aiex.dma_await_task(%22)
      aiex.dma_await_task(%23)
      aiex.dma_await_task(%24)
      %25 = aiex.dma_configure_task_for @inK {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 49664, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @inV {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 98816, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%26)
      %27 = aiex.dma_configure_task_for @inOW {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 393216, 4096, [<size = 12, stride = 64>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 11 : i32}
      aiex.dma_start_task(%27)
      aiex.dma_await_task(%25)
      aiex.dma_await_task(%26)
      aiex.dma_await_task(%27)
      %28 = aiex.dma_configure_task_for @inK {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 49728, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%28)
      %29 = aiex.dma_configure_task_for @inV {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 98880, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%29)
      %30 = aiex.dma_configure_task_for @inOW {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 442368, 4096, [<size = 12, stride = 64>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 11 : i32}
      aiex.dma_start_task(%30)
      aiex.dma_await_task(%28)
      aiex.dma_await_task(%29)
      aiex.dma_await_task(%30)
      %31 = aiex.dma_configure_task_for @inK {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 49792, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%31)
      %32 = aiex.dma_configure_task_for @inV {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 98944, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%32)
      %33 = aiex.dma_configure_task_for @inOW {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 491520, 4096, [<size = 12, stride = 64>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 11 : i32}
      aiex.dma_start_task(%33)
      aiex.dma_await_task(%31)
      aiex.dma_await_task(%32)
      aiex.dma_await_task(%33)
      %34 = aiex.dma_configure_task_for @inK {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 49856, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%34)
      %35 = aiex.dma_configure_task_for @inV {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 99008, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%35)
      %36 = aiex.dma_configure_task_for @inOW {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 540672, 4096, [<size = 12, stride = 64>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 11 : i32}
      aiex.dma_start_task(%36)
      aiex.dma_await_task(%34)
      aiex.dma_await_task(%35)
      aiex.dma_await_task(%36)
      %37 = aiex.dma_configure_task_for @inR {
        aie.dma_bd(%arg2 : memref<128x768xbf16>, 49152, 24576, [<size = 22, stride = 0>, <size = 12, stride = 64>, <size = 32, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 21 : i32}
      aiex.dma_start_task(%37)
      %38 = aiex.dma_configure_task_for @inR {
        aie.dma_bd(%arg2 : memref<128x768xbf16>, 49152, 24576, [<size = 22, stride = 0>, <size = 12, stride = 64>, <size = 32, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 21 : i32}
      aiex.dma_start_task(%38)
      %39 = aiex.dma_configure_task_for @inR {
        aie.dma_bd(%arg2 : memref<128x768xbf16>, 49152, 24576, [<size = 4, stride = 0>, <size = 12, stride = 64>, <size = 32, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%39)
      %40 = aiex.dma_configure_task_for @inBUp {
        aie.dma_bd(%arg3 : memref<2359296xbf16>, 0, 49152, [<size = 48, stride = 64>, <size = 12, stride = 196608>, <size = 64, stride = 3072>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 47 : i32}
      aiex.dma_start_task(%40)
      %41 = aiex.dma_configure_task_for @inBDown {
        aie.dma_bd(%arg4 : memref<2359296xbf16>, 0, 49152, [<size = 48, stride = 49152>, <size = 12, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 47 : i32}
      aiex.dma_start_task(%41)
      %42 = aiex.dma_configure_task_for @memLN2 {
        aie.dma_bd(%arg2 : memref<128x768xbf16>, 0, 24576, [<size = 1, stride = 0>, <size = 12, stride = 64>, <size = 32, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%42)
      aiex.dma_await_task(%42)
      aiex.dma_await_task(%37)
      aiex.dma_await_task(%38)
      aiex.dma_await_task(%39)
      aiex.dma_free_task(%40)
      aiex.dma_free_task(%41)
      %43 = aiex.dma_configure_task_for @inQ {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 24576, 24576, [<size = 1, stride = 0>, <size = 12, stride = 64>, <size = 32, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%43)
      aiex.dma_free_task(%43)
      %44 = aiex.dma_configure_task_for @inK {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 49152, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%44)
      %45 = aiex.dma_configure_task_for @inV {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 98304, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%45)
      %46 = aiex.dma_configure_task_for @inOW {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 0, 4096, [<size = 12, stride = 64>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 11 : i32}
      aiex.dma_start_task(%46)
      aiex.dma_await_task(%44)
      aiex.dma_await_task(%45)
      aiex.dma_await_task(%46)
      %47 = aiex.dma_configure_task_for @inK {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 49216, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%47)
      %48 = aiex.dma_configure_task_for @inV {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 98368, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%48)
      %49 = aiex.dma_configure_task_for @inOW {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 49152, 4096, [<size = 12, stride = 64>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 11 : i32}
      aiex.dma_start_task(%49)
      aiex.dma_await_task(%47)
      aiex.dma_await_task(%48)
      aiex.dma_await_task(%49)
      %50 = aiex.dma_configure_task_for @inK {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 49280, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%50)
      %51 = aiex.dma_configure_task_for @inV {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 98432, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%51)
      %52 = aiex.dma_configure_task_for @inOW {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 98304, 4096, [<size = 12, stride = 64>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 11 : i32}
      aiex.dma_start_task(%52)
      aiex.dma_await_task(%50)
      aiex.dma_await_task(%51)
      aiex.dma_await_task(%52)
      %53 = aiex.dma_configure_task_for @inK {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 49344, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%53)
      %54 = aiex.dma_configure_task_for @inV {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 98496, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%54)
      %55 = aiex.dma_configure_task_for @inOW {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 147456, 4096, [<size = 12, stride = 64>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 11 : i32}
      aiex.dma_start_task(%55)
      aiex.dma_await_task(%53)
      aiex.dma_await_task(%54)
      aiex.dma_await_task(%55)
      %56 = aiex.dma_configure_task_for @inK {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 49408, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%56)
      %57 = aiex.dma_configure_task_for @inV {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 98560, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%57)
      %58 = aiex.dma_configure_task_for @inOW {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 196608, 4096, [<size = 12, stride = 64>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 11 : i32}
      aiex.dma_start_task(%58)
      aiex.dma_await_task(%56)
      aiex.dma_await_task(%57)
      aiex.dma_await_task(%58)
      %59 = aiex.dma_configure_task_for @inK {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 49472, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%59)
      %60 = aiex.dma_configure_task_for @inV {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 98624, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%60)
      %61 = aiex.dma_configure_task_for @inOW {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 245760, 4096, [<size = 12, stride = 64>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 11 : i32}
      aiex.dma_start_task(%61)
      aiex.dma_await_task(%59)
      aiex.dma_await_task(%60)
      aiex.dma_await_task(%61)
      %62 = aiex.dma_configure_task_for @inK {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 49536, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%62)
      %63 = aiex.dma_configure_task_for @inV {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 98688, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%63)
      %64 = aiex.dma_configure_task_for @inOW {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 294912, 4096, [<size = 12, stride = 64>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 11 : i32}
      aiex.dma_start_task(%64)
      aiex.dma_await_task(%62)
      aiex.dma_await_task(%63)
      aiex.dma_await_task(%64)
      %65 = aiex.dma_configure_task_for @inK {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 49600, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%65)
      %66 = aiex.dma_configure_task_for @inV {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 98752, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%66)
      %67 = aiex.dma_configure_task_for @inOW {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 344064, 4096, [<size = 12, stride = 64>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 11 : i32}
      aiex.dma_start_task(%67)
      aiex.dma_await_task(%65)
      aiex.dma_await_task(%66)
      aiex.dma_await_task(%67)
      %68 = aiex.dma_configure_task_for @inK {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 49664, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%68)
      %69 = aiex.dma_configure_task_for @inV {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 98816, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%69)
      %70 = aiex.dma_configure_task_for @inOW {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 393216, 4096, [<size = 12, stride = 64>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 11 : i32}
      aiex.dma_start_task(%70)
      aiex.dma_await_task(%68)
      aiex.dma_await_task(%69)
      aiex.dma_await_task(%70)
      %71 = aiex.dma_configure_task_for @inK {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 49728, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%71)
      %72 = aiex.dma_configure_task_for @inV {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 98880, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%72)
      %73 = aiex.dma_configure_task_for @inOW {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 442368, 4096, [<size = 12, stride = 64>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 11 : i32}
      aiex.dma_start_task(%73)
      aiex.dma_await_task(%71)
      aiex.dma_await_task(%72)
      aiex.dma_await_task(%73)
      %74 = aiex.dma_configure_task_for @inK {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 49792, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%74)
      %75 = aiex.dma_configure_task_for @inV {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 98944, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%75)
      %76 = aiex.dma_configure_task_for @inOW {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 491520, 4096, [<size = 12, stride = 64>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 11 : i32}
      aiex.dma_start_task(%76)
      aiex.dma_await_task(%74)
      aiex.dma_await_task(%75)
      aiex.dma_await_task(%76)
      %77 = aiex.dma_configure_task_for @inK {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 49856, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%77)
      %78 = aiex.dma_configure_task_for @inV {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 99008, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%78)
      %79 = aiex.dma_configure_task_for @inOW {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 540672, 4096, [<size = 12, stride = 64>, <size = 1, stride = 0>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 11 : i32}
      aiex.dma_start_task(%79)
      aiex.dma_await_task(%77)
      aiex.dma_await_task(%78)
      aiex.dma_await_task(%79)
      %80 = aiex.dma_configure_task_for @inR {
        aie.dma_bd(%arg2 : memref<128x768xbf16>, 73728, 24576, [<size = 22, stride = 0>, <size = 12, stride = 64>, <size = 32, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 21 : i32}
      aiex.dma_start_task(%80)
      %81 = aiex.dma_configure_task_for @inR {
        aie.dma_bd(%arg2 : memref<128x768xbf16>, 73728, 24576, [<size = 22, stride = 0>, <size = 12, stride = 64>, <size = 32, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 21 : i32}
      aiex.dma_start_task(%81)
      %82 = aiex.dma_configure_task_for @inR {
        aie.dma_bd(%arg2 : memref<128x768xbf16>, 73728, 24576, [<size = 4, stride = 0>, <size = 12, stride = 64>, <size = 32, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%82)
      %83 = aiex.dma_configure_task_for @inBUp {
        aie.dma_bd(%arg3 : memref<2359296xbf16>, 0, 49152, [<size = 48, stride = 64>, <size = 12, stride = 196608>, <size = 64, stride = 3072>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 47 : i32}
      aiex.dma_start_task(%83)
      %84 = aiex.dma_configure_task_for @inBDown {
        aie.dma_bd(%arg4 : memref<2359296xbf16>, 0, 49152, [<size = 48, stride = 49152>, <size = 12, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 47 : i32}
      aiex.dma_start_task(%84)
      %85 = aiex.dma_configure_task_for @memLN2 {
        aie.dma_bd(%arg2 : memref<128x768xbf16>, 24576, 24576, [<size = 1, stride = 0>, <size = 12, stride = 64>, <size = 32, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%85)
      aiex.dma_await_task(%85)
      aiex.dma_await_task(%80)
      aiex.dma_await_task(%81)
      aiex.dma_await_task(%82)
      aiex.dma_free_task(%83)
      aiex.dma_free_task(%84)
    }
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%outOProjAccum0_src_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOProjAccum0_src : memref<32x64xbf16>, 0, 2048)
      aie.use_lock(%outOProjAccum0_src_empty, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%outOProjAccum0_dst_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOProjAccum0_dst : memref<32x64xbf16>, 0, 2048)
      aie.use_lock(%outOProjAccum0_dst_full, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %memtile_dma_4_1 = aie.memtile_dma(%mem_tile_4_1_0) {
      %0 = aie.dma_start(S2MM, 2, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%outOProjAccum0_row_0_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOProjAccum0_row_0 : memref<24576xbf16>, 0, 24576)
      aie.use_lock(%outOProjAccum0_row_0_full, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%outOProjAccum0_row_1_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOProjAccum0_row_1 : memref<24576xbf16>, 0, 24576)
      aie.use_lock(%outOProjAccum0_row_1_full, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 2, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%outOProjAccum0_row_0_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOProjAccum0_row_0 : memref<24576xbf16>, 0, 24576)
      aie.use_lock(%outOProjAccum0_row_0_empty, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%outOProjAccum0_row_1_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOProjAccum0_row_1 : memref<24576xbf16>, 0, 24576)
      aie.use_lock(%outOProjAccum0_row_1_empty, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%ln1Replay_src_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%ln1Replay_src : memref<32x64xbf16>, 0, 2048)
      aie.use_lock(%ln1Replay_src_empty, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%ln1Replay_dst_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%ln1Replay_dst : memref<32x64xbf16>, 0, 2048)
      aie.use_lock(%ln1Replay_dst_full, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %memtile_dma_5_1 = aie.memtile_dma(%mem_tile_5_1_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%ln1Replay_row_0_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%ln1Replay_row_0 : memref<24576xbf16>, 0, 24576)
      aie.use_lock(%ln1Replay_row_0_full, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%ln1Replay_row_1_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%ln1Replay_row_1 : memref<24576xbf16>, 0, 24576)
      aie.use_lock(%ln1Replay_row_1_full, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%ln1Replay_row_0_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%ln1Replay_row_0 : memref<24576xbf16>, 0, 24576)
      aie.use_lock(%ln1Replay_row_0_empty, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%ln1Replay_row_1_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%ln1Replay_row_1 : memref<24576xbf16>, 0, 24576)
      aie.use_lock(%ln1Replay_row_1_empty, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    %mem_6_5 = aie.mem(%tile_6_5) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%ffnDownAccum_src_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownAccum_src : memref<32x64xbf16>, 0, 2048)
      aie.use_lock(%ffnDownAccum_src_empty, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%ffnDownAccum_dst_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownAccum_dst : memref<32x64xbf16>, 0, 2048)
      aie.use_lock(%ffnDownAccum_dst_full, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %memtile_dma_6_1 = aie.memtile_dma(%mem_tile_6_1_2) {
      %0 = aie.dma_start(S2MM, 2, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%ffnDownAccum_row_0_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownAccum_row_0 : memref<24576xbf16>, 0, 24576)
      aie.use_lock(%ffnDownAccum_row_0_full, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%ffnDownAccum_row_1_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownAccum_row_1 : memref<24576xbf16>, 0, 24576)
      aie.use_lock(%ffnDownAccum_row_1_full, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 2, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%ffnDownAccum_row_0_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownAccum_row_0 : memref<24576xbf16>, 0, 24576)
      aie.use_lock(%ffnDownAccum_row_0_empty, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%ffnDownAccum_row_1_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownAccum_row_1 : memref<24576xbf16>, 0, 24576)
      aie.use_lock(%ffnDownAccum_row_1_empty, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
  }
}

