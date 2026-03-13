module {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_0_3 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_0_4 = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_0_5 = aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_1_2 = aie.tile(1, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_1_3 = aie.tile(1, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_1_4 = aie.tile(1, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_1_5 = aie.tile(1, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_2_2 = aie.tile(2, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_2_3 = aie.tile(2, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_2_4 = aie.tile(2, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_2_5 = aie.tile(2, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_3_2 = aie.tile(3, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_3_3 = aie.tile(3, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_3_4 = aie.tile(3, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_3_5 = aie.tile(3, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_6_5 = aie.tile(6, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_4_2 = aie.tile(4, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_6_4 = aie.tile(6, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_7_4 = aie.tile(7, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_7_5 = aie.tile(7, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %mem_tile_6_1 = aie.tile(6, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_7_1 = aie.tile(7, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %mem_tile_1_1 = aie.tile(1, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %shim_noc_tile_3_0 = aie.tile(3, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %mem_tile_3_1 = aie.tile(3, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %shim_noc_tile_7_0 = aie.tile(7, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_2_0 = aie.tile(2, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %mem_tile_2_1 = aie.tile(2, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_4_1 = aie.tile(4, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    aie.flow(%mem_tile_6_1, DMA : 0, %tile_7_4, DMA : 0)
    aie.flow(%tile_7_4, DMA : 0, %mem_tile_6_1, DMA : 0)
    aie.flow(%mem_tile_7_1, DMA : 0, %tile_7_5, DMA : 0)
    aie.flow(%tile_4_2, DMA : 0, %mem_tile_7_1, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %mem_tile_1_1, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_1_2, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 2, %tile_2_2, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 3, %tile_3_2, DMA : 0)
    aie.flow(%shim_noc_tile_3_0, DMA : 0, %mem_tile_3_1, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 0, %tile_0_5, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 1, %tile_1_5, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 2, %tile_2_5, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 3, %tile_3_5, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_1_2, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 2, %tile_2_2, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 3, %tile_3_2, DMA : 1)
    aie.flow(%shim_noc_tile_7_0, DMA : 0, %mem_tile_7_1, DMA : 1)
    aie.flow(%mem_tile_7_1, DMA : 1, %tile_4_2, DMA : 0)
    aie.flow(%shim_noc_tile_2_0, DMA : 0, %mem_tile_2_1, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 0, %tile_0_4, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_1_4, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 2, %tile_2_4, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 3, %tile_3_4, DMA : 0)
    aie.flow(%tile_6_5, DMA : 1, %tile_4_2, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_3, DMA : 0)
    aie.flow(%tile_1_2, DMA : 0, %tile_1_3, DMA : 0)
    aie.flow(%tile_2_2, DMA : 0, %tile_2_3, DMA : 0)
    aie.flow(%tile_3_2, DMA : 0, %tile_3_3, DMA : 0)
    aie.flow(%mem_tile_7_1, DMA : 2, %shim_noc_tile_7_0, DMA : 0)
    aie.flow(%tile_7_5, DMA : 0, %mem_tile_7_1, DMA : 2)
    aie.flow(%tile_0_3, DMA : 0, %tile_0_4, DMA : 1)
    aie.flow(%tile_1_3, DMA : 0, %tile_1_4, DMA : 1)
    aie.flow(%tile_2_3, DMA : 0, %tile_2_4, DMA : 1)
    aie.flow(%tile_3_3, DMA : 0, %tile_3_4, DMA : 1)
    aie.flow(%tile_4_2, DMA : 1, %tile_6_4, DMA : 0)
    aie.flow(%mem_tile_4_1, DMA : 0, %tile_3_5, DMA : 1)
    aie.flow(%tile_3_5, DMA : 0, %mem_tile_4_1, DMA : 0)
    aie.flow(%tile_3_5, DMA : 1, %tile_6_5, DMA : 1)
    func.func private @zero_bf16(memref<32x64xbf16>)
    func.func private @matmul_bf16_bf16_wrapper(memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<2xi32>)
    %idx_buffer_qk_0 = aie.buffer(%tile_0_2) {address = 53248 : i32, mem_bank = 3 : i32, sym_name = "idx_buffer_qk_0"} : memref<2xi32> = dense<0>
    func.func private @partial_softmax(memref<32x64xbf16>, memref<32x64xbf16>, memref<128xbf16>, memref<2xi32>, bf16, i32, i32, i32, i32)
    func.func private @init_scale_buffer(memref<128xbf16>, i32)
    func.func private @passThroughLine(memref<128xbf16>, memref<128xbf16>, i32)
    %idx_buffer_softmax_0 = aie.buffer(%tile_0_3) {address = 7680 : i32, mem_bank = 0 : i32, sym_name = "idx_buffer_softmax_0"} : memref<2xi32> = dense<0>
    %scale_buffer_softmax_0 = aie.buffer(%tile_0_3) {address = 7424 : i32, mem_bank = 0 : i32, sym_name = "scale_buffer_softmax_0"} : memref<128xbf16> = dense<0.000000e+00>
    func.func private @matmul_PV(memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<128xbf16>, i32, i32, memref<2xi32>)
    func.func private @rescale_O(memref<32x64xbf16>, memref<128xbf16>, i32, memref<2xi32>)
    %idx_buffer_pv_0 = aie.buffer(%tile_0_4) {address = 53248 : i32, mem_bank = 3 : i32, sym_name = "idx_buffer_pv_0"} : memref<2xi32> = dense<0>
    %o_proj_partial_scratch_0 = aie.buffer(%tile_0_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "o_proj_partial_scratch_0"} : memref<32x96xbf16> 
    %o_proj_zero_scratch_0 = aie.buffer(%tile_0_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "o_proj_zero_scratch_0"} : memref<32x96xbf16> 
    %o_proj_stats_sum_0 = aie.buffer(%tile_0_5) {address = 15616 : i32, mem_bank = 0 : i32, sym_name = "o_proj_stats_sum_0"} : memref<32xf32> 
    %o_proj_stats_sumsq_0 = aie.buffer(%tile_0_5) {address = 28672 : i32, mem_bank = 1 : i32, sym_name = "o_proj_stats_sumsq_0"} : memref<32xf32> 
    func.func private @ln_zero_f32(memref<32xf32>, i32)
    func.func private @ln_calc_sum_sumsq(memref<32x96xbf16>, memref<32xf32>, memref<32xf32>)
    func.func private @pack_stats_f32_to_bf16_packet(memref<32xf32>, memref<32xf32>, memref<32x96xbf16>, i32)
    func.func private @zero_bf16_o_proj(memref<32x96xbf16>)
    func.func private @matmul_with_acc_bf16_bf16_o_proj(memref<32x64xbf16>, memref<64x96xbf16>, memref<32x96xbf16>, memref<32x96xbf16>)
    func.func private @eltwise_add_bf16_vector_o_proj(memref<32x96xbf16>, memref<32x96xbf16>, memref<32x96xbf16>, i32)
    func.func private @passThroughLine_o_proj(memref<32x96xbf16>, memref<32x96xbf16>, i32)
    %idx_buffer_qk_1 = aie.buffer(%tile_1_2) {address = 53248 : i32, mem_bank = 3 : i32, sym_name = "idx_buffer_qk_1"} : memref<2xi32> = dense<0>
    %idx_buffer_softmax_1 = aie.buffer(%tile_1_3) {address = 7680 : i32, mem_bank = 0 : i32, sym_name = "idx_buffer_softmax_1"} : memref<2xi32> = dense<0>
    %scale_buffer_softmax_1 = aie.buffer(%tile_1_3) {address = 7424 : i32, mem_bank = 0 : i32, sym_name = "scale_buffer_softmax_1"} : memref<128xbf16> = dense<0.000000e+00>
    %idx_buffer_pv_1 = aie.buffer(%tile_1_4) {address = 53248 : i32, mem_bank = 3 : i32, sym_name = "idx_buffer_pv_1"} : memref<2xi32> = dense<0>
    %o_proj_partial_scratch_1 = aie.buffer(%tile_1_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "o_proj_partial_scratch_1"} : memref<32x96xbf16> 
    %o_proj_zero_scratch_1 = aie.buffer(%tile_1_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "o_proj_zero_scratch_1"} : memref<32x96xbf16> 
    %o_proj_stats_sum_1 = aie.buffer(%tile_1_5) {address = 15616 : i32, mem_bank = 0 : i32, sym_name = "o_proj_stats_sum_1"} : memref<32xf32> 
    %o_proj_stats_sumsq_1 = aie.buffer(%tile_1_5) {address = 28672 : i32, mem_bank = 1 : i32, sym_name = "o_proj_stats_sumsq_1"} : memref<32xf32> 
    %idx_buffer_qk_2 = aie.buffer(%tile_2_2) {address = 53248 : i32, mem_bank = 3 : i32, sym_name = "idx_buffer_qk_2"} : memref<2xi32> = dense<0>
    %idx_buffer_softmax_2 = aie.buffer(%tile_2_3) {address = 7680 : i32, mem_bank = 0 : i32, sym_name = "idx_buffer_softmax_2"} : memref<2xi32> = dense<0>
    %scale_buffer_softmax_2 = aie.buffer(%tile_2_3) {address = 7424 : i32, mem_bank = 0 : i32, sym_name = "scale_buffer_softmax_2"} : memref<128xbf16> = dense<0.000000e+00>
    %idx_buffer_pv_2 = aie.buffer(%tile_2_4) {address = 53248 : i32, mem_bank = 3 : i32, sym_name = "idx_buffer_pv_2"} : memref<2xi32> = dense<0>
    %o_proj_partial_scratch_2 = aie.buffer(%tile_2_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "o_proj_partial_scratch_2"} : memref<32x96xbf16> 
    %o_proj_zero_scratch_2 = aie.buffer(%tile_2_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "o_proj_zero_scratch_2"} : memref<32x96xbf16> 
    %o_proj_stats_sum_2 = aie.buffer(%tile_2_5) {address = 15616 : i32, mem_bank = 0 : i32, sym_name = "o_proj_stats_sum_2"} : memref<32xf32> 
    %o_proj_stats_sumsq_2 = aie.buffer(%tile_2_5) {address = 28672 : i32, mem_bank = 1 : i32, sym_name = "o_proj_stats_sumsq_2"} : memref<32xf32> 
    %idx_buffer_qk_3 = aie.buffer(%tile_3_2) {address = 53248 : i32, mem_bank = 3 : i32, sym_name = "idx_buffer_qk_3"} : memref<2xi32> = dense<0>
    %idx_buffer_softmax_3 = aie.buffer(%tile_3_3) {address = 7680 : i32, mem_bank = 0 : i32, sym_name = "idx_buffer_softmax_3"} : memref<2xi32> = dense<0>
    %scale_buffer_softmax_3 = aie.buffer(%tile_3_3) {address = 7424 : i32, mem_bank = 0 : i32, sym_name = "scale_buffer_softmax_3"} : memref<128xbf16> = dense<0.000000e+00>
    %idx_buffer_pv_3 = aie.buffer(%tile_3_4) {address = 53248 : i32, mem_bank = 3 : i32, sym_name = "idx_buffer_pv_3"} : memref<2xi32> = dense<0>
    %o_proj_partial_scratch_3 = aie.buffer(%tile_3_5) {address = 27904 : i32, sym_name = "o_proj_partial_scratch_3"} : memref<32x96xbf16> 
    %o_proj_zero_scratch_3 = aie.buffer(%tile_3_5) {address = 34048 : i32, sym_name = "o_proj_zero_scratch_3"} : memref<32x96xbf16> 
    %o_proj_stats_sum_3 = aie.buffer(%tile_3_5) {address = 64768 : i32, sym_name = "o_proj_stats_sum_3"} : memref<32xf32> 
    %o_proj_stats_sumsq_3 = aie.buffer(%tile_3_5) {address = 64896 : i32, sym_name = "o_proj_stats_sumsq_3"} : memref<32xf32> 
    %mem_tile_5_1 = aie.tile(5, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %scaleOF3_buff_0 = aie.buffer(%tile_3_3) {address = 20480 : i32, mem_bank = 1 : i32, sym_name = "scaleOF3_buff_0"} : memref<128xbf16> 
    %scaleOF3_buff_1 = aie.buffer(%tile_3_3) {address = 36864 : i32, mem_bank = 2 : i32, sym_name = "scaleOF3_buff_1"} : memref<128xbf16> 
    %scaleOF3_prod_lock_0 = aie.lock(%tile_3_3, 4) {init = 2 : i32, sym_name = "scaleOF3_prod_lock_0"}
    %scaleOF3_cons_lock_0 = aie.lock(%tile_3_3, 5) {init = 0 : i32, sym_name = "scaleOF3_cons_lock_0"}
    %scaleOF2_buff_0 = aie.buffer(%tile_2_3) {address = 20480 : i32, mem_bank = 1 : i32, sym_name = "scaleOF2_buff_0"} : memref<128xbf16> 
    %scaleOF2_buff_1 = aie.buffer(%tile_2_3) {address = 36864 : i32, mem_bank = 2 : i32, sym_name = "scaleOF2_buff_1"} : memref<128xbf16> 
    %scaleOF2_prod_lock_0 = aie.lock(%tile_2_3, 4) {init = 2 : i32, sym_name = "scaleOF2_prod_lock_0"}
    %scaleOF2_cons_lock_0 = aie.lock(%tile_2_3, 5) {init = 0 : i32, sym_name = "scaleOF2_cons_lock_0"}
    %scaleOF1_buff_0 = aie.buffer(%tile_1_3) {address = 20480 : i32, mem_bank = 1 : i32, sym_name = "scaleOF1_buff_0"} : memref<128xbf16> 
    %scaleOF1_buff_1 = aie.buffer(%tile_1_3) {address = 36864 : i32, mem_bank = 2 : i32, sym_name = "scaleOF1_buff_1"} : memref<128xbf16> 
    %scaleOF1_prod_lock_0 = aie.lock(%tile_1_3, 4) {init = 2 : i32, sym_name = "scaleOF1_prod_lock_0"}
    %scaleOF1_cons_lock_0 = aie.lock(%tile_1_3, 5) {init = 0 : i32, sym_name = "scaleOF1_cons_lock_0"}
    %scaleOF0_buff_0 = aie.buffer(%tile_0_3) {address = 20480 : i32, mem_bank = 1 : i32, sym_name = "scaleOF0_buff_0"} : memref<128xbf16> 
    %scaleOF0_buff_1 = aie.buffer(%tile_0_3) {address = 36864 : i32, mem_bank = 2 : i32, sym_name = "scaleOF0_buff_1"} : memref<128xbf16> 
    %scaleOF0_prod_lock_0 = aie.lock(%tile_0_3, 4) {init = 2 : i32, sym_name = "scaleOF0_prod_lock_0"}
    %scaleOF0_cons_lock_0 = aie.lock(%tile_0_3, 5) {init = 0 : i32, sym_name = "scaleOF0_cons_lock_0"}
    %outOProjInput_cons_buff_0 = aie.buffer(%tile_6_5) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "outOProjInput_cons_buff_0"} : memref<32x96xbf16> 
    %outOProjInput_cons_buff_1 = aie.buffer(%tile_6_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "outOProjInput_cons_buff_1"} : memref<32x96xbf16> 
    %outOProjInput_cons_prod_lock_0 = aie.lock(%tile_6_5, 6) {init = 2 : i32, sym_name = "outOProjInput_cons_prod_lock_0"}
    %outOProjInput_cons_cons_lock_0 = aie.lock(%tile_6_5, 7) {init = 0 : i32, sym_name = "outOProjInput_cons_cons_lock_0"}
    %outOProjInput_buff_0 = aie.buffer(%tile_3_5) {address = 40192 : i32, sym_name = "outOProjInput_buff_0"} : memref<32x96xbf16> 
    %outOProjInput_buff_1 = aie.buffer(%tile_3_5) {address = 46336 : i32, sym_name = "outOProjInput_buff_1"} : memref<32x96xbf16> 
    %outOProjInput_prod_lock_0 = aie.lock(%tile_3_5, 6) {init = 2 : i32, sym_name = "outOProjInput_prod_lock_0"}
    %outOProjInput_cons_lock_0 = aie.lock(%tile_3_5, 7) {init = 0 : i32, sym_name = "outOProjInput_cons_lock_0"}
    %outOProjAccumOut3_cons_buff_0 = aie.buffer(%mem_tile_4_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "outOProjAccumOut3_cons_buff_0"} : memref<32x96xbf16> 
    %outOProjAccumOut3_cons_buff_1 = aie.buffer(%mem_tile_4_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "outOProjAccumOut3_cons_buff_1"} : memref<32x96xbf16> 
    %outOProjAccumOut3_cons_buff_2 = aie.buffer(%mem_tile_4_1) {address = 131072 : i32, mem_bank = 2 : i32, sym_name = "outOProjAccumOut3_cons_buff_2"} : memref<32x96xbf16> 
    %outOProjAccumOut3_cons_buff_3 = aie.buffer(%mem_tile_4_1) {address = 196608 : i32, mem_bank = 3 : i32, sym_name = "outOProjAccumOut3_cons_buff_3"} : memref<32x96xbf16> 
    %outOProjAccumOut3_cons_buff_4 = aie.buffer(%mem_tile_4_1) {address = 262144 : i32, mem_bank = 4 : i32, sym_name = "outOProjAccumOut3_cons_buff_4"} : memref<32x96xbf16> 
    %outOProjAccumOut3_cons_buff_5 = aie.buffer(%mem_tile_4_1) {address = 327680 : i32, mem_bank = 5 : i32, sym_name = "outOProjAccumOut3_cons_buff_5"} : memref<32x96xbf16> 
    %outOProjAccumOut3_cons_buff_6 = aie.buffer(%mem_tile_4_1) {address = 393216 : i32, mem_bank = 6 : i32, sym_name = "outOProjAccumOut3_cons_buff_6"} : memref<32x96xbf16> 
    %outOProjAccumOut3_cons_buff_7 = aie.buffer(%mem_tile_4_1) {address = 458752 : i32, mem_bank = 7 : i32, sym_name = "outOProjAccumOut3_cons_buff_7"} : memref<32x96xbf16> 
    %outOProjAccumOut3_cons_prod_lock_0 = aie.lock(%mem_tile_4_1, 0) {init = 8 : i32, sym_name = "outOProjAccumOut3_cons_prod_lock_0"}
    %outOProjAccumOut3_cons_cons_lock_0 = aie.lock(%mem_tile_4_1, 1) {init = 0 : i32, sym_name = "outOProjAccumOut3_cons_cons_lock_0"}
    %outOProjAccumOut3_buff_0 = aie.buffer(%tile_3_5) {address = 52480 : i32, sym_name = "outOProjAccumOut3_buff_0"} : memref<32x96xbf16> 
    %outOProjAccumOut3_prod_lock_0 = aie.lock(%tile_3_5, 4) {init = 1 : i32, sym_name = "outOProjAccumOut3_prod_lock_0"}
    %outOProjAccumOut3_cons_lock_0 = aie.lock(%tile_3_5, 5) {init = 0 : i32, sym_name = "outOProjAccumOut3_cons_lock_0"}
    %outOProjAccumIn3_cons_buff_0 = aie.buffer(%tile_3_5) {address = 58624 : i32, sym_name = "outOProjAccumIn3_cons_buff_0"} : memref<32x96xbf16> 
    %outOProjAccumIn3_cons_prod_lock_0 = aie.lock(%tile_3_5, 2) {init = 1 : i32, sym_name = "outOProjAccumIn3_cons_prod_lock_0"}
    %outOProjAccumIn3_cons_cons_lock_0 = aie.lock(%tile_3_5, 3) {init = 0 : i32, sym_name = "outOProjAccumIn3_cons_cons_lock_0"}
    %outOProj3_buff_0 = aie.buffer(%tile_3_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "outOProj3_buff_0"} : memref<32x64xbf16> 
    %outOProj3_buff_1 = aie.buffer(%tile_3_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "outOProj3_buff_1"} : memref<32x64xbf16> 
    %outOProj3_prod_lock_0 = aie.lock(%tile_3_4, 4) {init = 2 : i32, sym_name = "outOProj3_prod_lock_0"}
    %outOProj3_cons_lock_0 = aie.lock(%tile_3_4, 5) {init = 0 : i32, sym_name = "outOProj3_cons_lock_0"}
    %outOProj2_buff_0 = aie.buffer(%tile_2_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "outOProj2_buff_0"} : memref<32x64xbf16> 
    %outOProj2_buff_1 = aie.buffer(%tile_2_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "outOProj2_buff_1"} : memref<32x64xbf16> 
    %outOProj2_prod_lock_0 = aie.lock(%tile_2_4, 4) {init = 2 : i32, sym_name = "outOProj2_prod_lock_0"}
    %outOProj2_cons_lock_0 = aie.lock(%tile_2_4, 5) {init = 0 : i32, sym_name = "outOProj2_cons_lock_0"}
    %outOProj1_buff_0 = aie.buffer(%tile_1_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "outOProj1_buff_0"} : memref<32x64xbf16> 
    %outOProj1_buff_1 = aie.buffer(%tile_1_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "outOProj1_buff_1"} : memref<32x64xbf16> 
    %outOProj1_prod_lock_0 = aie.lock(%tile_1_4, 4) {init = 2 : i32, sym_name = "outOProj1_prod_lock_0"}
    %outOProj1_cons_lock_0 = aie.lock(%tile_1_4, 5) {init = 0 : i32, sym_name = "outOProj1_cons_lock_0"}
    %outOProj0_buff_0 = aie.buffer(%tile_0_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "outOProj0_buff_0"} : memref<32x64xbf16> 
    %outOProj0_buff_1 = aie.buffer(%tile_0_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "outOProj0_buff_1"} : memref<32x64xbf16> 
    %outOProj0_prod_lock_0 = aie.lock(%tile_0_4, 4) {init = 2 : i32, sym_name = "outOProj0_prod_lock_0"}
    %outOProj0_cons_lock_0 = aie.lock(%tile_0_4, 5) {init = 0 : i32, sym_name = "outOProj0_cons_lock_0"}
    %outOGroupPart2_buff_0 = aie.buffer(%tile_2_5) {address = 38912 : i32, mem_bank = 2 : i32, sym_name = "outOGroupPart2_buff_0"} : memref<32x96xbf16> 
    %outOGroupPart2_buff_1 = aie.buffer(%tile_2_5) {address = 55296 : i32, mem_bank = 3 : i32, sym_name = "outOGroupPart2_buff_1"} : memref<32x96xbf16> 
    %outOGroupPart2_prod_lock_0 = aie.lock(%tile_2_5, 2) {init = 2 : i32, sym_name = "outOGroupPart2_prod_lock_0"}
    %outOGroupPart2_cons_lock_0 = aie.lock(%tile_2_5, 3) {init = 0 : i32, sym_name = "outOGroupPart2_cons_lock_0"}
    %outOGroupPart1_buff_0 = aie.buffer(%tile_1_5) {address = 38912 : i32, mem_bank = 2 : i32, sym_name = "outOGroupPart1_buff_0"} : memref<32x96xbf16> 
    %outOGroupPart1_buff_1 = aie.buffer(%tile_1_5) {address = 55296 : i32, mem_bank = 3 : i32, sym_name = "outOGroupPart1_buff_1"} : memref<32x96xbf16> 
    %outOGroupPart1_prod_lock_0 = aie.lock(%tile_1_5, 2) {init = 2 : i32, sym_name = "outOGroupPart1_prod_lock_0"}
    %outOGroupPart1_cons_lock_0 = aie.lock(%tile_1_5, 3) {init = 0 : i32, sym_name = "outOGroupPart1_cons_lock_0"}
    %outOGroupPart0_buff_0 = aie.buffer(%tile_0_5) {address = 38912 : i32, mem_bank = 2 : i32, sym_name = "outOGroupPart0_buff_0"} : memref<32x96xbf16> 
    %outOGroupPart0_buff_1 = aie.buffer(%tile_0_5) {address = 55296 : i32, mem_bank = 3 : i32, sym_name = "outOGroupPart0_buff_1"} : memref<32x96xbf16> 
    %outOGroupPart0_prod_lock_0 = aie.lock(%tile_0_5, 2) {init = 2 : i32, sym_name = "outOGroupPart0_prod_lock_0"}
    %outOGroupPart0_cons_lock_0 = aie.lock(%tile_0_5, 3) {init = 0 : i32, sym_name = "outOGroupPart0_cons_lock_0"}
    %outLNBroadcast_cons_buff_0 = aie.buffer(%tile_6_4) {address = 1792 : i32, mem_bank = 0 : i32, sym_name = "outLNBroadcast_cons_buff_0"} : memref<32x96xbf16> 
    %outLNBroadcast_cons_buff_1 = aie.buffer(%tile_6_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "outLNBroadcast_cons_buff_1"} : memref<32x96xbf16> 
    %outLNBroadcast_cons_prod_lock_0 = aie.lock(%tile_6_4, 2) {init = 2 : i32, sym_name = "outLNBroadcast_cons_prod_lock_0"}
    %outLNBroadcast_cons_cons_lock_0 = aie.lock(%tile_6_4, 3) {init = 0 : i32, sym_name = "outLNBroadcast_cons_cons_lock_0"}
    %outLNBroadcast_buff_0 = aie.buffer(%tile_4_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "outLNBroadcast_buff_0"} : memref<32x96xbf16> 
    %outLNBroadcast_prod_lock_0 = aie.lock(%tile_4_2, 6) {init = 1 : i32, sym_name = "outLNBroadcast_prod_lock_0"}
    %outLNBroadcast_cons_lock_0 = aie.lock(%tile_4_2, 7) {init = 0 : i32, sym_name = "outLNBroadcast_cons_lock_0"}
    %memP3_cons_buff_0 = aie.buffer(%tile_3_4) {address = 11520 : i32, mem_bank = 0 : i32, sym_name = "memP3_cons_buff_0"} : memref<32x64xbf16> 
    %memP3_cons_buff_1 = aie.buffer(%tile_3_4) {address = 24576 : i32, mem_bank = 1 : i32, sym_name = "memP3_cons_buff_1"} : memref<32x64xbf16> 
    %memP3_cons_prod_lock_0 = aie.lock(%tile_3_4, 2) {init = 2 : i32, sym_name = "memP3_cons_prod_lock_0"}
    %memP3_cons_cons_lock_0 = aie.lock(%tile_3_4, 3) {init = 0 : i32, sym_name = "memP3_cons_cons_lock_0"}
    %memP3_buff_0 = aie.buffer(%tile_3_3) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "memP3_buff_0"} : memref<32x64xbf16> 
    %memP3_buff_1 = aie.buffer(%tile_3_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "memP3_buff_1"} : memref<32x64xbf16> 
    %memP3_prod_lock_0 = aie.lock(%tile_3_3, 2) {init = 2 : i32, sym_name = "memP3_prod_lock_0"}
    %memP3_cons_lock_0 = aie.lock(%tile_3_3, 3) {init = 0 : i32, sym_name = "memP3_cons_lock_0"}
    %memP2_cons_buff_0 = aie.buffer(%tile_2_4) {address = 11520 : i32, mem_bank = 0 : i32, sym_name = "memP2_cons_buff_0"} : memref<32x64xbf16> 
    %memP2_cons_buff_1 = aie.buffer(%tile_2_4) {address = 24576 : i32, mem_bank = 1 : i32, sym_name = "memP2_cons_buff_1"} : memref<32x64xbf16> 
    %memP2_cons_prod_lock_0 = aie.lock(%tile_2_4, 2) {init = 2 : i32, sym_name = "memP2_cons_prod_lock_0"}
    %memP2_cons_cons_lock_0 = aie.lock(%tile_2_4, 3) {init = 0 : i32, sym_name = "memP2_cons_cons_lock_0"}
    %memP2_buff_0 = aie.buffer(%tile_2_3) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "memP2_buff_0"} : memref<32x64xbf16> 
    %memP2_buff_1 = aie.buffer(%tile_2_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "memP2_buff_1"} : memref<32x64xbf16> 
    %memP2_prod_lock_0 = aie.lock(%tile_2_3, 2) {init = 2 : i32, sym_name = "memP2_prod_lock_0"}
    %memP2_cons_lock_0 = aie.lock(%tile_2_3, 3) {init = 0 : i32, sym_name = "memP2_cons_lock_0"}
    %memP1_cons_buff_0 = aie.buffer(%tile_1_4) {address = 11520 : i32, mem_bank = 0 : i32, sym_name = "memP1_cons_buff_0"} : memref<32x64xbf16> 
    %memP1_cons_buff_1 = aie.buffer(%tile_1_4) {address = 24576 : i32, mem_bank = 1 : i32, sym_name = "memP1_cons_buff_1"} : memref<32x64xbf16> 
    %memP1_cons_prod_lock_0 = aie.lock(%tile_1_4, 2) {init = 2 : i32, sym_name = "memP1_cons_prod_lock_0"}
    %memP1_cons_cons_lock_0 = aie.lock(%tile_1_4, 3) {init = 0 : i32, sym_name = "memP1_cons_cons_lock_0"}
    %memP1_buff_0 = aie.buffer(%tile_1_3) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "memP1_buff_0"} : memref<32x64xbf16> 
    %memP1_buff_1 = aie.buffer(%tile_1_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "memP1_buff_1"} : memref<32x64xbf16> 
    %memP1_prod_lock_0 = aie.lock(%tile_1_3, 2) {init = 2 : i32, sym_name = "memP1_prod_lock_0"}
    %memP1_cons_lock_0 = aie.lock(%tile_1_3, 3) {init = 0 : i32, sym_name = "memP1_cons_lock_0"}
    %memP0_cons_buff_0 = aie.buffer(%tile_0_4) {address = 11520 : i32, mem_bank = 0 : i32, sym_name = "memP0_cons_buff_0"} : memref<32x64xbf16> 
    %memP0_cons_buff_1 = aie.buffer(%tile_0_4) {address = 24576 : i32, mem_bank = 1 : i32, sym_name = "memP0_cons_buff_1"} : memref<32x64xbf16> 
    %memP0_cons_prod_lock_0 = aie.lock(%tile_0_4, 2) {init = 2 : i32, sym_name = "memP0_cons_prod_lock_0"}
    %memP0_cons_cons_lock_0 = aie.lock(%tile_0_4, 3) {init = 0 : i32, sym_name = "memP0_cons_cons_lock_0"}
    %memP0_buff_0 = aie.buffer(%tile_0_3) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "memP0_buff_0"} : memref<32x64xbf16> 
    %memP0_buff_1 = aie.buffer(%tile_0_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "memP0_buff_1"} : memref<32x64xbf16> 
    %memP0_prod_lock_0 = aie.lock(%tile_0_3, 2) {init = 2 : i32, sym_name = "memP0_prod_lock_0"}
    %memP0_cons_lock_0 = aie.lock(%tile_0_3, 3) {init = 0 : i32, sym_name = "memP0_cons_lock_0"}
    %outLN2_cons_buff_0 = aie.buffer(%mem_tile_7_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "outLN2_cons_buff_0"} : memref<32x96xbf16> 
    %outLN2_cons_buff_1 = aie.buffer(%mem_tile_7_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "outLN2_cons_buff_1"} : memref<32x96xbf16> 
    %outLN2_cons_prod_lock_0 = aie.lock(%mem_tile_7_1, 4) {init = 2 : i32, sym_name = "outLN2_cons_prod_lock_0"}
    %outLN2_cons_cons_lock_0 = aie.lock(%mem_tile_7_1, 5) {init = 0 : i32, sym_name = "outLN2_cons_cons_lock_0"}
    %outLN2_buff_0 = aie.buffer(%tile_7_5) {address = 3840 : i32, mem_bank = 0 : i32, sym_name = "outLN2_buff_0"} : memref<32x96xbf16> 
    %outLN2_buff_1 = aie.buffer(%tile_7_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "outLN2_buff_1"} : memref<32x96xbf16> 
    %outLN2_prod_lock_0 = aie.lock(%tile_7_5, 2) {init = 2 : i32, sym_name = "outLN2_prod_lock_0"}
    %outLN2_cons_lock_0 = aie.lock(%tile_7_5, 3) {init = 0 : i32, sym_name = "outLN2_cons_lock_0"}
    %memLN2_cons_prod_lock_0 = aie.lock(%shim_noc_tile_7_0, 2) {init = 0 : i32, sym_name = "memLN2_cons_prod_lock_0"}
    %memLN2_cons_cons_lock_0 = aie.lock(%shim_noc_tile_7_0, 3) {init = 0 : i32, sym_name = "memLN2_cons_cons_lock_0"}
    %memA3_cons_buff_0 = aie.buffer(%tile_3_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "memA3_cons_buff_0"} : memref<32x64xbf16> 
    %memA3_cons_buff_1 = aie.buffer(%tile_3_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "memA3_cons_buff_1"} : memref<32x64xbf16> 
    %memA3_cons_prod_lock_0 = aie.lock(%tile_3_3, 0) {init = 2 : i32, sym_name = "memA3_cons_prod_lock_0"}
    %memA3_cons_cons_lock_0 = aie.lock(%tile_3_3, 1) {init = 0 : i32, sym_name = "memA3_cons_cons_lock_0"}
    %memA3_buff_0 = aie.buffer(%tile_3_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "memA3_buff_0"} : memref<32x64xbf16> 
    %memA3_buff_1 = aie.buffer(%tile_3_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "memA3_buff_1"} : memref<32x64xbf16> 
    %memA3_prod_lock_0 = aie.lock(%tile_3_2, 4) {init = 2 : i32, sym_name = "memA3_prod_lock_0"}
    %memA3_cons_lock_0 = aie.lock(%tile_3_2, 5) {init = 0 : i32, sym_name = "memA3_cons_lock_0"}
    %memA2_cons_buff_0 = aie.buffer(%tile_2_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "memA2_cons_buff_0"} : memref<32x64xbf16> 
    %memA2_cons_buff_1 = aie.buffer(%tile_2_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "memA2_cons_buff_1"} : memref<32x64xbf16> 
    %memA2_cons_prod_lock_0 = aie.lock(%tile_2_3, 0) {init = 2 : i32, sym_name = "memA2_cons_prod_lock_0"}
    %memA2_cons_cons_lock_0 = aie.lock(%tile_2_3, 1) {init = 0 : i32, sym_name = "memA2_cons_cons_lock_0"}
    %memA2_buff_0 = aie.buffer(%tile_2_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "memA2_buff_0"} : memref<32x64xbf16> 
    %memA2_buff_1 = aie.buffer(%tile_2_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "memA2_buff_1"} : memref<32x64xbf16> 
    %memA2_prod_lock_0 = aie.lock(%tile_2_2, 4) {init = 2 : i32, sym_name = "memA2_prod_lock_0"}
    %memA2_cons_lock_0 = aie.lock(%tile_2_2, 5) {init = 0 : i32, sym_name = "memA2_cons_lock_0"}
    %memA1_cons_buff_0 = aie.buffer(%tile_1_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "memA1_cons_buff_0"} : memref<32x64xbf16> 
    %memA1_cons_buff_1 = aie.buffer(%tile_1_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "memA1_cons_buff_1"} : memref<32x64xbf16> 
    %memA1_cons_prod_lock_0 = aie.lock(%tile_1_3, 0) {init = 2 : i32, sym_name = "memA1_cons_prod_lock_0"}
    %memA1_cons_cons_lock_0 = aie.lock(%tile_1_3, 1) {init = 0 : i32, sym_name = "memA1_cons_cons_lock_0"}
    %memA1_buff_0 = aie.buffer(%tile_1_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "memA1_buff_0"} : memref<32x64xbf16> 
    %memA1_buff_1 = aie.buffer(%tile_1_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "memA1_buff_1"} : memref<32x64xbf16> 
    %memA1_prod_lock_0 = aie.lock(%tile_1_2, 4) {init = 2 : i32, sym_name = "memA1_prod_lock_0"}
    %memA1_cons_lock_0 = aie.lock(%tile_1_2, 5) {init = 0 : i32, sym_name = "memA1_cons_lock_0"}
    %memA0_cons_buff_0 = aie.buffer(%tile_0_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "memA0_cons_buff_0"} : memref<32x64xbf16> 
    %memA0_cons_buff_1 = aie.buffer(%tile_0_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "memA0_cons_buff_1"} : memref<32x64xbf16> 
    %memA0_cons_prod_lock_0 = aie.lock(%tile_0_3, 0) {init = 2 : i32, sym_name = "memA0_cons_prod_lock_0"}
    %memA0_cons_cons_lock_0 = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "memA0_cons_cons_lock_0"}
    %memA0_buff_0 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "memA0_buff_0"} : memref<32x64xbf16> 
    %memA0_buff_1 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "memA0_buff_1"} : memref<32x64xbf16> 
    %memA0_prod_lock_0 = aie.lock(%tile_0_2, 4) {init = 2 : i32, sym_name = "memA0_prod_lock_0"}
    %memA0_cons_lock_0 = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "memA0_cons_lock_0"}
    %ln1Norm_cons_buff_0 = aie.buffer(%tile_4_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "ln1Norm_cons_buff_0"} : memref<32x96xbf16> 
    %ln1Norm_cons_buff_1 = aie.buffer(%tile_4_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "ln1Norm_cons_buff_1"} : memref<32x96xbf16> 
    %ln1Norm_cons_prod_lock_0 = aie.lock(%tile_4_2, 4) {init = 2 : i32, sym_name = "ln1Norm_cons_prod_lock_0"}
    %ln1Norm_cons_cons_lock_0 = aie.lock(%tile_4_2, 5) {init = 0 : i32, sym_name = "ln1Norm_cons_cons_lock_0"}
    %ln1Norm_buff_0 = aie.buffer(%tile_6_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "ln1Norm_buff_0"} : memref<32x96xbf16> 
    %ln1Norm_buff_1 = aie.buffer(%tile_6_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "ln1Norm_buff_1"} : memref<32x96xbf16> 
    %ln1Norm_prod_lock_0 = aie.lock(%tile_6_5, 4) {init = 2 : i32, sym_name = "ln1Norm_prod_lock_0"}
    %ln1Norm_cons_lock_0 = aie.lock(%tile_6_5, 5) {init = 0 : i32, sym_name = "ln1Norm_cons_lock_0"}
    %memV3_cons_buff_0 = aie.buffer(%tile_3_4) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "memV3_cons_buff_0"} : memref<64x64xbf16> 
    %memV3_cons_buff_1 = aie.buffer(%tile_3_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "memV3_cons_buff_1"} : memref<64x64xbf16> 
    %memV3_cons_prod_lock_0 = aie.lock(%tile_3_4, 0) {init = 2 : i32, sym_name = "memV3_cons_prod_lock_0"}
    %memV3_cons_cons_lock_0 = aie.lock(%tile_3_4, 1) {init = 0 : i32, sym_name = "memV3_cons_cons_lock_0"}
    %memV2_cons_buff_0 = aie.buffer(%tile_2_4) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "memV2_cons_buff_0"} : memref<64x64xbf16> 
    %memV2_cons_buff_1 = aie.buffer(%tile_2_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "memV2_cons_buff_1"} : memref<64x64xbf16> 
    %memV2_cons_prod_lock_0 = aie.lock(%tile_2_4, 0) {init = 2 : i32, sym_name = "memV2_cons_prod_lock_0"}
    %memV2_cons_cons_lock_0 = aie.lock(%tile_2_4, 1) {init = 0 : i32, sym_name = "memV2_cons_cons_lock_0"}
    %memV1_cons_buff_0 = aie.buffer(%tile_1_4) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "memV1_cons_buff_0"} : memref<64x64xbf16> 
    %memV1_cons_buff_1 = aie.buffer(%tile_1_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "memV1_cons_buff_1"} : memref<64x64xbf16> 
    %memV1_cons_prod_lock_0 = aie.lock(%tile_1_4, 0) {init = 2 : i32, sym_name = "memV1_cons_prod_lock_0"}
    %memV1_cons_cons_lock_0 = aie.lock(%tile_1_4, 1) {init = 0 : i32, sym_name = "memV1_cons_cons_lock_0"}
    %memV0_cons_buff_0 = aie.buffer(%tile_0_4) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "memV0_cons_buff_0"} : memref<64x64xbf16> 
    %memV0_cons_buff_1 = aie.buffer(%tile_0_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "memV0_cons_buff_1"} : memref<64x64xbf16> 
    %memV0_cons_prod_lock_0 = aie.lock(%tile_0_4, 0) {init = 2 : i32, sym_name = "memV0_cons_prod_lock_0"}
    %memV0_cons_cons_lock_0 = aie.lock(%tile_0_4, 1) {init = 0 : i32, sym_name = "memV0_cons_cons_lock_0"}
    %inV_cons_buff_0 = aie.buffer(%mem_tile_2_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "inV_cons_buff_0"} : memref<64x256xbf16> 
    %inV_cons_buff_1 = aie.buffer(%mem_tile_2_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "inV_cons_buff_1"} : memref<64x256xbf16> 
    %inV_cons_prod_lock_0 = aie.lock(%mem_tile_2_1, 0) {init = 2 : i32, sym_name = "inV_cons_prod_lock_0"}
    %inV_cons_cons_lock_0 = aie.lock(%mem_tile_2_1, 1) {init = 0 : i32, sym_name = "inV_cons_cons_lock_0"}
    %inV_cons_prod_lock_1 = aie.lock(%mem_tile_2_1, 2) {init = 2 : i32, sym_name = "inV_cons_prod_lock_1"}
    %inV_cons_cons_lock_1 = aie.lock(%mem_tile_2_1, 3) {init = 0 : i32, sym_name = "inV_cons_cons_lock_1"}
    %inV_cons_prod_lock_2 = aie.lock(%mem_tile_2_1, 4) {init = 2 : i32, sym_name = "inV_cons_prod_lock_2"}
    %inV_cons_cons_lock_2 = aie.lock(%mem_tile_2_1, 5) {init = 0 : i32, sym_name = "inV_cons_cons_lock_2"}
    %inV_cons_prod_lock_3 = aie.lock(%mem_tile_2_1, 6) {init = 2 : i32, sym_name = "inV_cons_prod_lock_3"}
    %inV_cons_cons_lock_3 = aie.lock(%mem_tile_2_1, 7) {init = 0 : i32, sym_name = "inV_cons_cons_lock_3"}
    %inV_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 0) {init = 0 : i32, sym_name = "inV_prod_lock_0"}
    %inV_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 1) {init = 0 : i32, sym_name = "inV_cons_lock_0"}
    %memR_cons_buff_0 = aie.buffer(%tile_4_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "memR_cons_buff_0"} : memref<32x96xbf16> 
    %memR_cons_buff_1 = aie.buffer(%tile_4_2) {address = 7168 : i32, mem_bank = 0 : i32, sym_name = "memR_cons_buff_1"} : memref<32x96xbf16> 
    %memR_cons_prod_lock_0 = aie.lock(%tile_4_2, 2) {init = 2 : i32, sym_name = "memR_cons_prod_lock_0"}
    %memR_cons_cons_lock_0 = aie.lock(%tile_4_2, 3) {init = 0 : i32, sym_name = "memR_cons_cons_lock_0"}
    %inR_cons_buff_0 = aie.buffer(%mem_tile_7_1) {address = 131072 : i32, mem_bank = 2 : i32, sym_name = "inR_cons_buff_0"} : memref<32x96xbf16> 
    %inR_cons_buff_1 = aie.buffer(%mem_tile_7_1) {address = 196608 : i32, mem_bank = 3 : i32, sym_name = "inR_cons_buff_1"} : memref<32x96xbf16> 
    %inR_cons_prod_lock_0 = aie.lock(%mem_tile_7_1, 2) {init = 2 : i32, sym_name = "inR_cons_prod_lock_0"}
    %inR_cons_cons_lock_0 = aie.lock(%mem_tile_7_1, 3) {init = 0 : i32, sym_name = "inR_cons_cons_lock_0"}
    %inR_prod_lock_0 = aie.lock(%shim_noc_tile_7_0, 0) {init = 0 : i32, sym_name = "inR_prod_lock_0"}
    %inR_cons_lock_0 = aie.lock(%shim_noc_tile_7_0, 1) {init = 0 : i32, sym_name = "inR_cons_lock_0"}
    %memQ3_cons_buff_0 = aie.buffer(%tile_3_2) {address = 11520 : i32, mem_bank = 0 : i32, sym_name = "memQ3_cons_buff_0"} : memref<32x64xbf16> 
    %memQ3_cons_buff_1 = aie.buffer(%tile_3_2) {address = 24576 : i32, mem_bank = 1 : i32, sym_name = "memQ3_cons_buff_1"} : memref<32x64xbf16> 
    %memQ3_cons_prod_lock_0 = aie.lock(%tile_3_2, 2) {init = 2 : i32, sym_name = "memQ3_cons_prod_lock_0"}
    %memQ3_cons_cons_lock_0 = aie.lock(%tile_3_2, 3) {init = 0 : i32, sym_name = "memQ3_cons_cons_lock_0"}
    %memQ2_cons_buff_0 = aie.buffer(%tile_2_2) {address = 11520 : i32, mem_bank = 0 : i32, sym_name = "memQ2_cons_buff_0"} : memref<32x64xbf16> 
    %memQ2_cons_buff_1 = aie.buffer(%tile_2_2) {address = 24576 : i32, mem_bank = 1 : i32, sym_name = "memQ2_cons_buff_1"} : memref<32x64xbf16> 
    %memQ2_cons_prod_lock_0 = aie.lock(%tile_2_2, 2) {init = 2 : i32, sym_name = "memQ2_cons_prod_lock_0"}
    %memQ2_cons_cons_lock_0 = aie.lock(%tile_2_2, 3) {init = 0 : i32, sym_name = "memQ2_cons_cons_lock_0"}
    %memQ1_cons_buff_0 = aie.buffer(%tile_1_2) {address = 11520 : i32, mem_bank = 0 : i32, sym_name = "memQ1_cons_buff_0"} : memref<32x64xbf16> 
    %memQ1_cons_buff_1 = aie.buffer(%tile_1_2) {address = 24576 : i32, mem_bank = 1 : i32, sym_name = "memQ1_cons_buff_1"} : memref<32x64xbf16> 
    %memQ1_cons_prod_lock_0 = aie.lock(%tile_1_2, 2) {init = 2 : i32, sym_name = "memQ1_cons_prod_lock_0"}
    %memQ1_cons_cons_lock_0 = aie.lock(%tile_1_2, 3) {init = 0 : i32, sym_name = "memQ1_cons_cons_lock_0"}
    %memQ0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 11520 : i32, mem_bank = 0 : i32, sym_name = "memQ0_cons_buff_0"} : memref<32x64xbf16> 
    %memQ0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 24576 : i32, mem_bank = 1 : i32, sym_name = "memQ0_cons_buff_1"} : memref<32x64xbf16> 
    %memQ0_cons_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "memQ0_cons_prod_lock_0"}
    %memQ0_cons_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "memQ0_cons_cons_lock_0"}
    %inQ_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "inQ_cons_buff_0"} : memref<32x256xbf16> 
    %inQ_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "inQ_cons_buff_1"} : memref<32x256xbf16> 
    %inQ_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 0) {init = 2 : i32, sym_name = "inQ_cons_prod_lock_0"}
    %inQ_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "inQ_cons_cons_lock_0"}
    %inQ_cons_prod_lock_1 = aie.lock(%mem_tile_0_1, 2) {init = 2 : i32, sym_name = "inQ_cons_prod_lock_1"}
    %inQ_cons_cons_lock_1 = aie.lock(%mem_tile_0_1, 3) {init = 0 : i32, sym_name = "inQ_cons_cons_lock_1"}
    %inQ_cons_prod_lock_2 = aie.lock(%mem_tile_0_1, 4) {init = 2 : i32, sym_name = "inQ_cons_prod_lock_2"}
    %inQ_cons_cons_lock_2 = aie.lock(%mem_tile_0_1, 5) {init = 0 : i32, sym_name = "inQ_cons_cons_lock_2"}
    %inQ_cons_prod_lock_3 = aie.lock(%mem_tile_0_1, 6) {init = 2 : i32, sym_name = "inQ_cons_prod_lock_3"}
    %inQ_cons_cons_lock_3 = aie.lock(%mem_tile_0_1, 7) {init = 0 : i32, sym_name = "inQ_cons_cons_lock_3"}
    %inQ_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "inQ_prod_lock_0"}
    %inQ_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "inQ_cons_lock_0"}
    %memOW3_cons_buff_0 = aie.buffer(%tile_3_5) {address = 3328 : i32, sym_name = "memOW3_cons_buff_0"} : memref<64x96xbf16> 
    %memOW3_cons_buff_1 = aie.buffer(%tile_3_5) {address = 15616 : i32, sym_name = "memOW3_cons_buff_1"} : memref<64x96xbf16> 
    %memOW3_cons_prod_lock_0 = aie.lock(%tile_3_5, 0) {init = 2 : i32, sym_name = "memOW3_cons_prod_lock_0"}
    %memOW3_cons_cons_lock_0 = aie.lock(%tile_3_5, 1) {init = 0 : i32, sym_name = "memOW3_cons_cons_lock_0"}
    %memOW2_cons_buff_0 = aie.buffer(%tile_2_5) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "memOW2_cons_buff_0"} : memref<64x96xbf16> 
    %memOW2_cons_buff_1 = aie.buffer(%tile_2_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "memOW2_cons_buff_1"} : memref<64x96xbf16> 
    %memOW2_cons_prod_lock_0 = aie.lock(%tile_2_5, 0) {init = 2 : i32, sym_name = "memOW2_cons_prod_lock_0"}
    %memOW2_cons_cons_lock_0 = aie.lock(%tile_2_5, 1) {init = 0 : i32, sym_name = "memOW2_cons_cons_lock_0"}
    %memOW1_cons_buff_0 = aie.buffer(%tile_1_5) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "memOW1_cons_buff_0"} : memref<64x96xbf16> 
    %memOW1_cons_buff_1 = aie.buffer(%tile_1_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "memOW1_cons_buff_1"} : memref<64x96xbf16> 
    %memOW1_cons_prod_lock_0 = aie.lock(%tile_1_5, 0) {init = 2 : i32, sym_name = "memOW1_cons_prod_lock_0"}
    %memOW1_cons_cons_lock_0 = aie.lock(%tile_1_5, 1) {init = 0 : i32, sym_name = "memOW1_cons_cons_lock_0"}
    %memOW0_cons_buff_0 = aie.buffer(%tile_0_5) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "memOW0_cons_buff_0"} : memref<64x96xbf16> 
    %memOW0_cons_buff_1 = aie.buffer(%tile_0_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "memOW0_cons_buff_1"} : memref<64x96xbf16> 
    %memOW0_cons_prod_lock_0 = aie.lock(%tile_0_5, 0) {init = 2 : i32, sym_name = "memOW0_cons_prod_lock_0"}
    %memOW0_cons_cons_lock_0 = aie.lock(%tile_0_5, 1) {init = 0 : i32, sym_name = "memOW0_cons_cons_lock_0"}
    %inOW_cons_buff_0 = aie.buffer(%mem_tile_3_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "inOW_cons_buff_0"} : memref<256x96xbf16> 
    %inOW_cons_buff_1 = aie.buffer(%mem_tile_3_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "inOW_cons_buff_1"} : memref<256x96xbf16> 
    %inOW_cons_prod_lock_0 = aie.lock(%mem_tile_3_1, 0) {init = 2 : i32, sym_name = "inOW_cons_prod_lock_0"}
    %inOW_cons_cons_lock_0 = aie.lock(%mem_tile_3_1, 1) {init = 0 : i32, sym_name = "inOW_cons_cons_lock_0"}
    %inOW_cons_prod_lock_1 = aie.lock(%mem_tile_3_1, 2) {init = 2 : i32, sym_name = "inOW_cons_prod_lock_1"}
    %inOW_cons_cons_lock_1 = aie.lock(%mem_tile_3_1, 3) {init = 0 : i32, sym_name = "inOW_cons_cons_lock_1"}
    %inOW_cons_prod_lock_2 = aie.lock(%mem_tile_3_1, 4) {init = 2 : i32, sym_name = "inOW_cons_prod_lock_2"}
    %inOW_cons_cons_lock_2 = aie.lock(%mem_tile_3_1, 5) {init = 0 : i32, sym_name = "inOW_cons_cons_lock_2"}
    %inOW_cons_prod_lock_3 = aie.lock(%mem_tile_3_1, 6) {init = 2 : i32, sym_name = "inOW_cons_prod_lock_3"}
    %inOW_cons_cons_lock_3 = aie.lock(%mem_tile_3_1, 7) {init = 0 : i32, sym_name = "inOW_cons_cons_lock_3"}
    %inOW_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 0) {init = 0 : i32, sym_name = "inOW_prod_lock_0"}
    %inOW_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 1) {init = 0 : i32, sym_name = "inOW_cons_lock_0"}
    %memK3_cons_buff_0 = aie.buffer(%tile_3_2) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "memK3_cons_buff_0"} : memref<64x64xbf16> 
    %memK3_cons_buff_1 = aie.buffer(%tile_3_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "memK3_cons_buff_1"} : memref<64x64xbf16> 
    %memK3_cons_prod_lock_0 = aie.lock(%tile_3_2, 0) {init = 2 : i32, sym_name = "memK3_cons_prod_lock_0"}
    %memK3_cons_cons_lock_0 = aie.lock(%tile_3_2, 1) {init = 0 : i32, sym_name = "memK3_cons_cons_lock_0"}
    %memK2_cons_buff_0 = aie.buffer(%tile_2_2) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "memK2_cons_buff_0"} : memref<64x64xbf16> 
    %memK2_cons_buff_1 = aie.buffer(%tile_2_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "memK2_cons_buff_1"} : memref<64x64xbf16> 
    %memK2_cons_prod_lock_0 = aie.lock(%tile_2_2, 0) {init = 2 : i32, sym_name = "memK2_cons_prod_lock_0"}
    %memK2_cons_cons_lock_0 = aie.lock(%tile_2_2, 1) {init = 0 : i32, sym_name = "memK2_cons_cons_lock_0"}
    %memK1_cons_buff_0 = aie.buffer(%tile_1_2) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "memK1_cons_buff_0"} : memref<64x64xbf16> 
    %memK1_cons_buff_1 = aie.buffer(%tile_1_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "memK1_cons_buff_1"} : memref<64x64xbf16> 
    %memK1_cons_prod_lock_0 = aie.lock(%tile_1_2, 0) {init = 2 : i32, sym_name = "memK1_cons_prod_lock_0"}
    %memK1_cons_cons_lock_0 = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "memK1_cons_cons_lock_0"}
    %memK0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "memK0_cons_buff_0"} : memref<64x64xbf16> 
    %memK0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "memK0_cons_buff_1"} : memref<64x64xbf16> 
    %memK0_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "memK0_cons_prod_lock_0"}
    %memK0_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "memK0_cons_cons_lock_0"}
    %inK_cons_buff_0 = aie.buffer(%mem_tile_1_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "inK_cons_buff_0"} : memref<64x256xbf16> 
    %inK_cons_buff_1 = aie.buffer(%mem_tile_1_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "inK_cons_buff_1"} : memref<64x256xbf16> 
    %inK_cons_prod_lock_0 = aie.lock(%mem_tile_1_1, 0) {init = 2 : i32, sym_name = "inK_cons_prod_lock_0"}
    %inK_cons_cons_lock_0 = aie.lock(%mem_tile_1_1, 1) {init = 0 : i32, sym_name = "inK_cons_cons_lock_0"}
    %inK_cons_prod_lock_1 = aie.lock(%mem_tile_1_1, 2) {init = 2 : i32, sym_name = "inK_cons_prod_lock_1"}
    %inK_cons_cons_lock_1 = aie.lock(%mem_tile_1_1, 3) {init = 0 : i32, sym_name = "inK_cons_cons_lock_1"}
    %inK_cons_prod_lock_2 = aie.lock(%mem_tile_1_1, 4) {init = 2 : i32, sym_name = "inK_cons_prod_lock_2"}
    %inK_cons_cons_lock_2 = aie.lock(%mem_tile_1_1, 5) {init = 0 : i32, sym_name = "inK_cons_cons_lock_2"}
    %inK_cons_prod_lock_3 = aie.lock(%mem_tile_1_1, 6) {init = 2 : i32, sym_name = "inK_cons_prod_lock_3"}
    %inK_cons_cons_lock_3 = aie.lock(%mem_tile_1_1, 7) {init = 0 : i32, sym_name = "inK_cons_cons_lock_3"}
    %inK_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 0) {init = 0 : i32, sym_name = "inK_prod_lock_0"}
    %inK_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 1) {init = 0 : i32, sym_name = "inK_cons_lock_0"}
    %ffnUpOut_buff_0 = aie.buffer(%tile_6_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "ffnUpOut_buff_0"} : memref<32x96xbf16> 
    %ffnUpOut_buff_1 = aie.buffer(%tile_6_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "ffnUpOut_buff_1"} : memref<32x96xbf16> 
    %ffnUpOut_prod_lock_0 = aie.lock(%tile_6_4, 0) {init = 2 : i32, sym_name = "ffnUpOut_prod_lock_0"}
    %ffnUpOut_cons_lock_0 = aie.lock(%tile_6_4, 1) {init = 0 : i32, sym_name = "ffnUpOut_cons_lock_0"}
    %ffnROut_cons_buff_0 = aie.buffer(%mem_tile_7_1) {address = 262144 : i32, mem_bank = 4 : i32, sym_name = "ffnROut_cons_buff_0"} : memref<32x96xbf16> 
    %ffnROut_cons_buff_1 = aie.buffer(%mem_tile_7_1) {address = 327680 : i32, mem_bank = 5 : i32, sym_name = "ffnROut_cons_buff_1"} : memref<32x96xbf16> 
    %ffnROut_cons_buff_2 = aie.buffer(%mem_tile_7_1) {address = 393216 : i32, mem_bank = 6 : i32, sym_name = "ffnROut_cons_buff_2"} : memref<32x96xbf16> 
    %ffnROut_cons_buff_3 = aie.buffer(%mem_tile_7_1) {address = 458752 : i32, mem_bank = 7 : i32, sym_name = "ffnROut_cons_buff_3"} : memref<32x96xbf16> 
    %ffnROut_cons_buff_4 = aie.buffer(%mem_tile_7_1) {address = 6144 : i32, mem_bank = 0 : i32, sym_name = "ffnROut_cons_buff_4"} : memref<32x96xbf16> 
    %ffnROut_cons_buff_5 = aie.buffer(%mem_tile_7_1) {address = 71680 : i32, mem_bank = 1 : i32, sym_name = "ffnROut_cons_buff_5"} : memref<32x96xbf16> 
    %ffnROut_cons_buff_6 = aie.buffer(%mem_tile_7_1) {address = 137216 : i32, mem_bank = 2 : i32, sym_name = "ffnROut_cons_buff_6"} : memref<32x96xbf16> 
    %ffnROut_cons_buff_7 = aie.buffer(%mem_tile_7_1) {address = 202752 : i32, mem_bank = 3 : i32, sym_name = "ffnROut_cons_buff_7"} : memref<32x96xbf16> 
    %ffnROut_cons_prod_lock_0 = aie.lock(%mem_tile_7_1, 0) {init = 8 : i32, sym_name = "ffnROut_cons_prod_lock_0"}
    %ffnROut_cons_cons_lock_0 = aie.lock(%mem_tile_7_1, 1) {init = 0 : i32, sym_name = "ffnROut_cons_cons_lock_0"}
    %ffnROut_buff_0 = aie.buffer(%tile_4_2) {address = 22528 : i32, mem_bank = 1 : i32, sym_name = "ffnROut_buff_0"} : memref<32x96xbf16> 
    %ffnROut_prod_lock_0 = aie.lock(%tile_4_2, 0) {init = 1 : i32, sym_name = "ffnROut_prod_lock_0"}
    %ffnROut_cons_lock_0 = aie.lock(%tile_4_2, 1) {init = 0 : i32, sym_name = "ffnROut_cons_lock_0"}
    %ffnRIn_cons_buff_0 = aie.buffer(%tile_7_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "ffnRIn_cons_buff_0"} : memref<32x96xbf16> 
    %ffnRIn_cons_buff_1 = aie.buffer(%tile_7_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "ffnRIn_cons_buff_1"} : memref<32x96xbf16> 
    %ffnRIn_cons_prod_lock_0 = aie.lock(%tile_7_5, 0) {init = 2 : i32, sym_name = "ffnRIn_cons_prod_lock_0"}
    %ffnRIn_cons_cons_lock_0 = aie.lock(%tile_7_5, 1) {init = 0 : i32, sym_name = "ffnRIn_cons_cons_lock_0"}
    %ffnDownOut_buff_0 = aie.buffer(%tile_7_4) {address = 3840 : i32, mem_bank = 0 : i32, sym_name = "ffnDownOut_buff_0"} : memref<32x96xbf16> 
    %ffnDownOut_buff_1 = aie.buffer(%tile_7_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "ffnDownOut_buff_1"} : memref<32x96xbf16> 
    %ffnDownOut_prod_lock_0 = aie.lock(%tile_7_4, 4) {init = 2 : i32, sym_name = "ffnDownOut_prod_lock_0"}
    %ffnDownOut_cons_lock_0 = aie.lock(%tile_7_4, 5) {init = 0 : i32, sym_name = "ffnDownOut_cons_lock_0"}
    %ffnDownPart_cons_buff_0 = aie.buffer(%mem_tile_6_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "ffnDownPart_cons_buff_0"} : memref<32x96xbf16> 
    %ffnDownPart_cons_buff_1 = aie.buffer(%mem_tile_6_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "ffnDownPart_cons_buff_1"} : memref<32x96xbf16> 
    %ffnDownPart_cons_buff_2 = aie.buffer(%mem_tile_6_1) {address = 131072 : i32, mem_bank = 2 : i32, sym_name = "ffnDownPart_cons_buff_2"} : memref<32x96xbf16> 
    %ffnDownPart_cons_buff_3 = aie.buffer(%mem_tile_6_1) {address = 196608 : i32, mem_bank = 3 : i32, sym_name = "ffnDownPart_cons_buff_3"} : memref<32x96xbf16> 
    %ffnDownPart_cons_buff_4 = aie.buffer(%mem_tile_6_1) {address = 262144 : i32, mem_bank = 4 : i32, sym_name = "ffnDownPart_cons_buff_4"} : memref<32x96xbf16> 
    %ffnDownPart_cons_buff_5 = aie.buffer(%mem_tile_6_1) {address = 327680 : i32, mem_bank = 5 : i32, sym_name = "ffnDownPart_cons_buff_5"} : memref<32x96xbf16> 
    %ffnDownPart_cons_buff_6 = aie.buffer(%mem_tile_6_1) {address = 393216 : i32, mem_bank = 6 : i32, sym_name = "ffnDownPart_cons_buff_6"} : memref<32x96xbf16> 
    %ffnDownPart_cons_buff_7 = aie.buffer(%mem_tile_6_1) {address = 458752 : i32, mem_bank = 7 : i32, sym_name = "ffnDownPart_cons_buff_7"} : memref<32x96xbf16> 
    %ffnDownPart_cons_prod_lock_0 = aie.lock(%mem_tile_6_1, 0) {init = 8 : i32, sym_name = "ffnDownPart_cons_prod_lock_0"}
    %ffnDownPart_cons_cons_lock_0 = aie.lock(%mem_tile_6_1, 1) {init = 0 : i32, sym_name = "ffnDownPart_cons_cons_lock_0"}
    %ffnDownPart_buff_0 = aie.buffer(%tile_7_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "ffnDownPart_buff_0"} : memref<32x96xbf16> 
    %ffnDownPart_prod_lock_0 = aie.lock(%tile_7_4, 2) {init = 1 : i32, sym_name = "ffnDownPart_prod_lock_0"}
    %ffnDownPart_cons_lock_0 = aie.lock(%tile_7_4, 3) {init = 0 : i32, sym_name = "ffnDownPart_cons_lock_0"}
    %ffnDownAccum_cons_buff_0 = aie.buffer(%tile_7_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "ffnDownAccum_cons_buff_0"} : memref<32x96xbf16> 
    %ffnDownAccum_cons_prod_lock_0 = aie.lock(%tile_7_4, 0) {init = 1 : i32, sym_name = "ffnDownAccum_cons_prod_lock_0"}
    %ffnDownAccum_cons_cons_lock_0 = aie.lock(%tile_7_4, 1) {init = 0 : i32, sym_name = "ffnDownAccum_cons_cons_lock_0"}
    %ln1_norm_sum_buffer = aie.buffer(%tile_6_5) {address = 38912 : i32, mem_bank = 2 : i32, sym_name = "ln1_norm_sum_buffer"} : memref<32xf32> 
    %ln1_norm_sumsq_buffer = aie.buffer(%tile_6_5) {address = 55296 : i32, mem_bank = 3 : i32, sym_name = "ln1_norm_sumsq_buffer"} : memref<32xf32> 
    func.func private @fused_layer_norm_1outs(memref<32x96xbf16>, memref<32xf32>, memref<32xf32>, memref<32x96xbf16>, i32)
    func.func private @unpack_stats_bf16_packet_to_f32(memref<32x96xbf16>, memref<32xf32>, memref<32xf32>, i32)
    %static_ln1_weights = aie.buffer(%tile_4_2) {address = 38912 : i32, mem_bank = 2 : i32, sym_name = "static_ln1_weights"} : memref<768xbf16> = dense<1.000000e+00>
    func.func private @ln_mul_add_1outs(memref<32x96xbf16>, memref<32x96xbf16>, memref<768xbf16>, memref<32x96xbf16>, i32)
    func.func private @ffn_zero_bf16_up_proj(memref<32x96xbf16>)
    func.func private @ffn_matmul_init_bf16_bf16_up_proj(memref<32x96xbf16>, memref<96x96xbf16>, memref<32x96xbf16>)
    func.func private @ffn_matmul_bf16_bf16_up_proj(memref<32x96xbf16>, memref<96x96xbf16>, memref<32x96xbf16>)
    func.func private @ffn_gelu_bf16(memref<32x96xbf16>, memref<32x96xbf16>, i32)
    func.func private @ffn_zero_bf16_down_proj(memref<32x96xbf16>)
    func.func private @ffn_matmul_init_bf16_bf16_down_proj(memref<32x96xbf16>, memref<96x96xbf16>, memref<32x96xbf16>)
    func.func private @ffn_matmul_with_acc_bf16_bf16_down_proj(memref<32x96xbf16>, memref<96x96xbf16>, memref<32x96xbf16>, memref<32x96xbf16>)
    %ln2_sum_buffer = aie.buffer(%tile_7_5) {address = 22528 : i32, mem_bank = 1 : i32, sym_name = "ln2_sum_buffer"} : memref<32xf32> 
    %ln2_sumsq_buffer = aie.buffer(%tile_7_5) {address = 38912 : i32, mem_bank = 2 : i32, sym_name = "ln2_sumsq_buffer"} : memref<32xf32> 
    %static_ln2_weights = aie.buffer(%tile_7_5) {address = 9984 : i32, mem_bank = 0 : i32, sym_name = "static_ln2_weights"} : memref<768xbf16> = dense<1.000000e+00>
    func.func private @fused_add_layer_norm_1outs(memref<32x96xbf16>, memref<32x96xbf16>, memref<768xbf16>, memref<32xf32>, memref<32xf32>, memref<32x96xbf16>, i32, i32)
    %ln1Replay_src = aie.buffer(%tile_6_5) {address = 7168 : i32, mem_bank = 0 : i32, sym_name = "ln1Replay_src"} : memref<32x96xbf16> 
    %ln1Replay_dst = aie.buffer(%tile_6_5) {address = 22528 : i32, mem_bank = 1 : i32, sym_name = "ln1Replay_dst"} : memref<32x96xbf16> 
    %ln1Replay_row = aie.buffer(%mem_tile_5_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "ln1Replay_row"} : memref<24576xbf16> 
    %ln1Replay_src_empty = aie.lock(%tile_6_5, 0) {init = 1 : i32, sym_name = "ln1Replay_src_empty"}
    %ln1Replay_src_full = aie.lock(%tile_6_5, 1) {init = 0 : i32, sym_name = "ln1Replay_src_full"}
    %ln1Replay_dst_empty = aie.lock(%tile_6_5, 2) {init = 1 : i32, sym_name = "ln1Replay_dst_empty"}
    %ln1Replay_dst_full = aie.lock(%tile_6_5, 3) {init = 0 : i32, sym_name = "ln1Replay_dst_full"}
    %ln1Replay_row_empty = aie.lock(%mem_tile_5_1, 0) {init = 1 : i32, sym_name = "ln1Replay_row_empty"}
    %ln1Replay_row_full = aie.lock(%mem_tile_5_1, 1) {init = 0 : i32, sym_name = "ln1Replay_row_full"}
    aie.flow(%tile_6_5, DMA : 0, %mem_tile_5_1, DMA : 0)
    aie.flow(%mem_tile_5_1, DMA : 1, %tile_6_5, DMA : 0)
    %_anonymous0 = aie.buffer(%tile_0_2) {address = 36864 : i32, mem_bank = 2 : i32, sym_name = "_anonymous0"} : memref<3xi32> 
    %core_0_2 = aie.core(%tile_0_2) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c3 = arith.constant 3 : index
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous0[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous0[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous0[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb20
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb21
    ^bb2:  // pred: ^bb1
      memref.store %c0_i32, %idx_buffer_qk_0[%c0] : memref<2xi32>
      memref.store %c0_i32, %idx_buffer_qk_0[%c1] : memref<2xi32>
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb19
      %3 = arith.cmpi slt, %2, %c3 : index
      cf.cond_br %3, ^bb4, ^bb20
    ^bb4:  // pred: ^bb3
      aie.use_lock(%memQ0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%memQ0_cons_buff_0 : memref<32x64xbf16>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%memQ0_cons_buff_1 : memref<32x64xbf16>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%memQ0_cons_buff_0 : memref<32x64xbf16>)
    ^bb8(%7: memref<32x64xbf16>):  // 3 preds: ^bb5, ^bb6, ^bb7
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb18
      %9 = arith.cmpi slt, %8, %c8 : index
      cf.cond_br %9, ^bb10, ^bb19
    ^bb10:  // pred: ^bb9
      aie.use_lock(%memK0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%memK0_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%memK0_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%memK0_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%13: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      aie.use_lock(%memA0_prod_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i32
      cf.switch %16 : i32, [
        default: ^bb17,
        0: ^bb15,
        1: ^bb16
      ]
    ^bb15:  // pred: ^bb14
      cf.br ^bb18(%memA0_buff_0 : memref<32x64xbf16>)
    ^bb16:  // pred: ^bb14
      cf.br ^bb18(%memA0_buff_1 : memref<32x64xbf16>)
    ^bb17:  // pred: ^bb14
      cf.br ^bb18(%memA0_buff_0 : memref<32x64xbf16>)
    ^bb18(%17: memref<32x64xbf16>):  // 3 preds: ^bb15, ^bb16, ^bb17
      func.call @zero_bf16(%17) : (memref<32x64xbf16>) -> ()
      func.call @matmul_bf16_bf16_wrapper(%7, %13, %17, %idx_buffer_qk_0) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<2xi32>) -> ()
      aie.use_lock(%memK0_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%memA0_cons_lock_0, Release, 1)
      %23 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous0[%c2] : memref<3xi32>
      %28 = memref.load %idx_buffer_qk_0[%c0] : memref<2xi32>
      memref.store %28, %idx_buffer_qk_0[%c0] : memref<2xi32>
      %29 = arith.addi %8, %c1 : index
      cf.br ^bb9(%29 : index)
    ^bb19:  // pred: ^bb9
      memref.store %c0_i32, %idx_buffer_qk_0[%c0] : memref<2xi32>
      %30 = memref.load %idx_buffer_qk_0[%c1] : memref<2xi32>
      memref.store %30, %idx_buffer_qk_0[%c1] : memref<2xi32>
      aie.use_lock(%memQ0_cons_prod_lock_0, Release, 1)
      %31 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %32 = arith.addi %31, %c1_i32 : i32
      %33 = arith.cmpi sge, %32, %c2_i32 : i32
      %34 = arith.subi %32, %c2_i32 : i32
      %35 = arith.select %33, %34, %32 : i32
      memref.store %35, %_anonymous0[%c0] : memref<3xi32>
      %36 = arith.addi %2, %c1 : index
      cf.br ^bb3(%36 : index)
    ^bb20:  // pred: ^bb3
      %37 = arith.addi %0, %c1 : index
      cf.br ^bb1(%37 : index)
    ^bb21:  // pred: ^bb1
      aie.end
    } {link_with = "ep_12h_512s_96e_4ph_8pa_4g_1pf_d9_m0_f6_n1-1_n21_me_8971c3b5f9ae_kernels.a", stack_size = 3328 : i32}
    %_anonymous1 = aie.buffer(%tile_0_3) {address = 53248 : i32, mem_bank = 3 : i32, sym_name = "_anonymous1"} : memref<3xi32> 
    %core_0_3 = aie.core(%tile_0_3) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c3 = arith.constant 3 : index
      %c32_i32 = arith.constant 32 : i32
      %c8 = arith.constant 8 : index
      %cst = arith.constant 1.806640e-01 : bf16
      %c64_i32 = arith.constant 64 : i32
      %c512_i32 = arith.constant 512 : i32
      %c128_i32 = arith.constant 128 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous1[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous1[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous1[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb20
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb21
    ^bb2:  // pred: ^bb1
      memref.store %c0_i32, %idx_buffer_softmax_0[%c0] : memref<2xi32>
      memref.store %c0_i32, %idx_buffer_softmax_0[%c1] : memref<2xi32>
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb19
      %3 = arith.cmpi slt, %2, %c3 : index
      cf.cond_br %3, ^bb4, ^bb20
    ^bb4:  // pred: ^bb3
      func.call @init_scale_buffer(%scale_buffer_softmax_0, %c32_i32) : (memref<128xbf16>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb18
      %5 = arith.cmpi slt, %4, %c8 : index
      cf.cond_br %5, ^bb6, ^bb19
    ^bb6:  // pred: ^bb5
      aie.use_lock(%memP0_prod_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%memP0_buff_0 : memref<32x64xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%memP0_buff_1 : memref<32x64xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%memP0_buff_0 : memref<32x64xbf16>)
    ^bb10(%9: memref<32x64xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%memA0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%memA0_cons_buff_0 : memref<32x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%memA0_cons_buff_1 : memref<32x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%memA0_cons_buff_0 : memref<32x64xbf16>)
    ^bb14(%13: memref<32x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      aie.use_lock(%scaleOF0_prod_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i32
      cf.switch %16 : i32, [
        default: ^bb17,
        0: ^bb15,
        1: ^bb16
      ]
    ^bb15:  // pred: ^bb14
      cf.br ^bb18(%scaleOF0_buff_0 : memref<128xbf16>)
    ^bb16:  // pred: ^bb14
      cf.br ^bb18(%scaleOF0_buff_1 : memref<128xbf16>)
    ^bb17:  // pred: ^bb14
      cf.br ^bb18(%scaleOF0_buff_0 : memref<128xbf16>)
    ^bb18(%17: memref<128xbf16>):  // 3 preds: ^bb15, ^bb16, ^bb17
      func.call @partial_softmax(%13, %9, %scale_buffer_softmax_0, %idx_buffer_softmax_0, %cst, %c32_i32, %c64_i32, %c512_i32, %c512_i32) : (memref<32x64xbf16>, memref<32x64xbf16>, memref<128xbf16>, memref<2xi32>, bf16, i32, i32, i32, i32) -> ()
      func.call @passThroughLine(%scale_buffer_softmax_0, %17, %c128_i32) : (memref<128xbf16>, memref<128xbf16>, i32) -> ()
      aie.use_lock(%memA0_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous1[%c1] : memref<3xi32>
      aie.use_lock(%memP0_cons_lock_0, Release, 1)
      %23 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous1[%c0] : memref<3xi32>
      aie.use_lock(%scaleOF0_cons_lock_0, Release, 1)
      %28 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %29 = arith.addi %28, %c1_i32 : i32
      %30 = arith.cmpi sge, %29, %c2_i32 : i32
      %31 = arith.subi %29, %c2_i32 : i32
      %32 = arith.select %30, %31, %29 : i32
      memref.store %32, %_anonymous1[%c2] : memref<3xi32>
      %33 = memref.load %idx_buffer_softmax_0[%c0] : memref<2xi32>
      memref.store %33, %idx_buffer_softmax_0[%c0] : memref<2xi32>
      %34 = arith.addi %4, %c1 : index
      cf.br ^bb5(%34 : index)
    ^bb19:  // pred: ^bb5
      memref.store %c0_i32, %idx_buffer_softmax_0[%c0] : memref<2xi32>
      %35 = memref.load %idx_buffer_softmax_0[%c1] : memref<2xi32>
      memref.store %35, %idx_buffer_softmax_0[%c1] : memref<2xi32>
      %36 = arith.addi %2, %c1 : index
      cf.br ^bb3(%36 : index)
    ^bb20:  // pred: ^bb3
      %37 = arith.addi %0, %c1 : index
      cf.br ^bb1(%37 : index)
    ^bb21:  // pred: ^bb1
      aie.end
    } {link_with = "ep_12h_512s_96e_4ph_8pa_4g_1pf_d9_m0_f6_n1-1_n21_me_8971c3b5f9ae_kernels.a", stack_size = 3328 : i32}
    %_anonymous2 = aie.buffer(%tile_0_4) {address = 36864 : i32, mem_bank = 2 : i32, sym_name = "_anonymous2"} : memref<4xi32> 
    %core_0_4 = aie.core(%tile_0_4) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c32_i32 = arith.constant 32 : i32
      %c6 = arith.constant 6 : index
      %c1_i32 = arith.constant 1 : i32
      %c3 = arith.constant 3 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous2[%c0] : memref<4xi32>
      memref.store %c0_i32, %_anonymous2[%c1] : memref<4xi32>
      memref.store %c0_i32, %_anonymous2[%c2] : memref<4xi32>
      memref.store %c0_i32, %_anonymous2[%c3] : memref<4xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      memref.store %c0_i32, %idx_buffer_pv_0[%c0] : memref<2xi32>
      memref.store %c0_i32, %idx_buffer_pv_0[%c1] : memref<2xi32>
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb47
      %3 = arith.cmpi slt, %2, %c3 : index
      cf.cond_br %3, ^bb4, ^bb48
    ^bb4:  // pred: ^bb3
      aie.use_lock(%outOProj0_prod_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous2[%c0] : memref<4xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%outOProj0_buff_0 : memref<32x64xbf16>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%outOProj0_buff_1 : memref<32x64xbf16>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%outOProj0_buff_0 : memref<32x64xbf16>)
    ^bb8(%7: memref<32x64xbf16>):  // 3 preds: ^bb5, ^bb6, ^bb7
      func.call @zero_bf16(%7) : (memref<32x64xbf16>) -> ()
      aie.use_lock(%memP0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous2[%c1] : memref<4xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%memP0_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%memP0_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%memP0_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      aie.use_lock(%memV0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %12 = memref.load %_anonymous2[%c2] : memref<4xi32>
      %13 = arith.index_cast %12 : i32 to index
      %14 = arith.index_cast %13 : index to i32
      cf.switch %14 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%memV0_cons_buff_0 : memref<64x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%memV0_cons_buff_1 : memref<64x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%memV0_cons_buff_0 : memref<64x64xbf16>)
    ^bb16(%15: memref<64x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      aie.use_lock(%scaleOF0_cons_lock_0, AcquireGreaterEqual, 1)
      %16 = memref.load %_anonymous2[%c3] : memref<4xi32>
      %17 = arith.index_cast %16 : i32 to index
      %18 = arith.index_cast %17 : index to i32
      cf.switch %18 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%scaleOF0_buff_0 : memref<128xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%scaleOF0_buff_1 : memref<128xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%scaleOF0_buff_0 : memref<128xbf16>)
    ^bb20(%19: memref<128xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_PV(%11, %15, %7, %19, %c32_i32, %c0_i32, %idx_buffer_pv_0) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<128xbf16>, i32, i32, memref<2xi32>) -> ()
      aie.use_lock(%memP0_cons_prod_lock_0, Release, 1)
      %20 = memref.load %_anonymous2[%c1] : memref<4xi32>
      %21 = arith.addi %20, %c1_i32 : i32
      %22 = arith.cmpi sge, %21, %c2_i32 : i32
      %23 = arith.subi %21, %c2_i32 : i32
      %24 = arith.select %22, %23, %21 : i32
      memref.store %24, %_anonymous2[%c1] : memref<4xi32>
      aie.use_lock(%memV0_cons_prod_lock_0, Release, 1)
      %25 = memref.load %_anonymous2[%c2] : memref<4xi32>
      %26 = arith.addi %25, %c1_i32 : i32
      %27 = arith.cmpi sge, %26, %c2_i32 : i32
      %28 = arith.subi %26, %c2_i32 : i32
      %29 = arith.select %27, %28, %26 : i32
      memref.store %29, %_anonymous2[%c2] : memref<4xi32>
      aie.use_lock(%scaleOF0_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous2[%c3] : memref<4xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous2[%c3] : memref<4xi32>
      %35 = memref.load %idx_buffer_pv_0[%c0] : memref<2xi32>
      memref.store %35, %idx_buffer_pv_0[%c0] : memref<2xi32>
      cf.br ^bb21(%c0 : index)
    ^bb21(%36: index):  // 2 preds: ^bb20, ^bb34
      %37 = arith.cmpi slt, %36, %c6 : index
      cf.cond_br %37, ^bb22, ^bb35
    ^bb22:  // pred: ^bb21
      aie.use_lock(%memP0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %38 = memref.load %_anonymous2[%c1] : memref<4xi32>
      %39 = arith.index_cast %38 : i32 to index
      %40 = arith.index_cast %39 : index to i32
      cf.switch %40 : i32, [
        default: ^bb25,
        0: ^bb23,
        1: ^bb24
      ]
    ^bb23:  // pred: ^bb22
      cf.br ^bb26(%memP0_cons_buff_0 : memref<32x64xbf16>)
    ^bb24:  // pred: ^bb22
      cf.br ^bb26(%memP0_cons_buff_1 : memref<32x64xbf16>)
    ^bb25:  // pred: ^bb22
      cf.br ^bb26(%memP0_cons_buff_0 : memref<32x64xbf16>)
    ^bb26(%41: memref<32x64xbf16>):  // 3 preds: ^bb23, ^bb24, ^bb25
      aie.use_lock(%memV0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %42 = memref.load %_anonymous2[%c2] : memref<4xi32>
      %43 = arith.index_cast %42 : i32 to index
      %44 = arith.index_cast %43 : index to i32
      cf.switch %44 : i32, [
        default: ^bb29,
        0: ^bb27,
        1: ^bb28
      ]
    ^bb27:  // pred: ^bb26
      cf.br ^bb30(%memV0_cons_buff_0 : memref<64x64xbf16>)
    ^bb28:  // pred: ^bb26
      cf.br ^bb30(%memV0_cons_buff_1 : memref<64x64xbf16>)
    ^bb29:  // pred: ^bb26
      cf.br ^bb30(%memV0_cons_buff_0 : memref<64x64xbf16>)
    ^bb30(%45: memref<64x64xbf16>):  // 3 preds: ^bb27, ^bb28, ^bb29
      aie.use_lock(%scaleOF0_cons_lock_0, AcquireGreaterEqual, 1)
      %46 = memref.load %_anonymous2[%c3] : memref<4xi32>
      %47 = arith.index_cast %46 : i32 to index
      %48 = arith.index_cast %47 : index to i32
      cf.switch %48 : i32, [
        default: ^bb33,
        0: ^bb31,
        1: ^bb32
      ]
    ^bb31:  // pred: ^bb30
      cf.br ^bb34(%scaleOF0_buff_0 : memref<128xbf16>)
    ^bb32:  // pred: ^bb30
      cf.br ^bb34(%scaleOF0_buff_1 : memref<128xbf16>)
    ^bb33:  // pred: ^bb30
      cf.br ^bb34(%scaleOF0_buff_0 : memref<128xbf16>)
    ^bb34(%49: memref<128xbf16>):  // 3 preds: ^bb31, ^bb32, ^bb33
      func.call @matmul_PV(%41, %45, %7, %49, %c32_i32, %c1_i32, %idx_buffer_pv_0) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<128xbf16>, i32, i32, memref<2xi32>) -> ()
      aie.use_lock(%memP0_cons_prod_lock_0, Release, 1)
      %50 = memref.load %_anonymous2[%c1] : memref<4xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c2_i32 : i32
      %53 = arith.subi %51, %c2_i32 : i32
      %54 = arith.select %52, %53, %51 : i32
      memref.store %54, %_anonymous2[%c1] : memref<4xi32>
      aie.use_lock(%memV0_cons_prod_lock_0, Release, 1)
      %55 = memref.load %_anonymous2[%c2] : memref<4xi32>
      %56 = arith.addi %55, %c1_i32 : i32
      %57 = arith.cmpi sge, %56, %c2_i32 : i32
      %58 = arith.subi %56, %c2_i32 : i32
      %59 = arith.select %57, %58, %56 : i32
      memref.store %59, %_anonymous2[%c2] : memref<4xi32>
      aie.use_lock(%scaleOF0_prod_lock_0, Release, 1)
      %60 = memref.load %_anonymous2[%c3] : memref<4xi32>
      %61 = arith.addi %60, %c1_i32 : i32
      %62 = arith.cmpi sge, %61, %c2_i32 : i32
      %63 = arith.subi %61, %c2_i32 : i32
      %64 = arith.select %62, %63, %61 : i32
      memref.store %64, %_anonymous2[%c3] : memref<4xi32>
      %65 = memref.load %idx_buffer_pv_0[%c0] : memref<2xi32>
      memref.store %65, %idx_buffer_pv_0[%c0] : memref<2xi32>
      %66 = arith.addi %36, %c1 : index
      cf.br ^bb21(%66 : index)
    ^bb35:  // pred: ^bb21
      aie.use_lock(%memP0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %67 = memref.load %_anonymous2[%c1] : memref<4xi32>
      %68 = arith.index_cast %67 : i32 to index
      %69 = arith.index_cast %68 : index to i32
      cf.switch %69 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%memP0_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%memP0_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%memP0_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%70: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      aie.use_lock(%memV0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %71 = memref.load %_anonymous2[%c2] : memref<4xi32>
      %72 = arith.index_cast %71 : i32 to index
      %73 = arith.index_cast %72 : index to i32
      cf.switch %73 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%memV0_cons_buff_0 : memref<64x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%memV0_cons_buff_1 : memref<64x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%memV0_cons_buff_0 : memref<64x64xbf16>)
    ^bb43(%74: memref<64x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      aie.use_lock(%scaleOF0_cons_lock_0, AcquireGreaterEqual, 1)
      %75 = memref.load %_anonymous2[%c3] : memref<4xi32>
      %76 = arith.index_cast %75 : i32 to index
      %77 = arith.index_cast %76 : index to i32
      cf.switch %77 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%scaleOF0_buff_0 : memref<128xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%scaleOF0_buff_1 : memref<128xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%scaleOF0_buff_0 : memref<128xbf16>)
    ^bb47(%78: memref<128xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_PV(%70, %74, %7, %78, %c32_i32, %c1_i32, %idx_buffer_pv_0) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<128xbf16>, i32, i32, memref<2xi32>) -> ()
      func.call @rescale_O(%7, %78, %c32_i32, %idx_buffer_pv_0) : (memref<32x64xbf16>, memref<128xbf16>, i32, memref<2xi32>) -> ()
      aie.use_lock(%memP0_cons_prod_lock_0, Release, 1)
      %79 = memref.load %_anonymous2[%c1] : memref<4xi32>
      %80 = arith.addi %79, %c1_i32 : i32
      %81 = arith.cmpi sge, %80, %c2_i32 : i32
      %82 = arith.subi %80, %c2_i32 : i32
      %83 = arith.select %81, %82, %80 : i32
      memref.store %83, %_anonymous2[%c1] : memref<4xi32>
      aie.use_lock(%memV0_cons_prod_lock_0, Release, 1)
      %84 = memref.load %_anonymous2[%c2] : memref<4xi32>
      %85 = arith.addi %84, %c1_i32 : i32
      %86 = arith.cmpi sge, %85, %c2_i32 : i32
      %87 = arith.subi %85, %c2_i32 : i32
      %88 = arith.select %86, %87, %85 : i32
      memref.store %88, %_anonymous2[%c2] : memref<4xi32>
      aie.use_lock(%scaleOF0_prod_lock_0, Release, 1)
      %89 = memref.load %_anonymous2[%c3] : memref<4xi32>
      %90 = arith.addi %89, %c1_i32 : i32
      %91 = arith.cmpi sge, %90, %c2_i32 : i32
      %92 = arith.subi %90, %c2_i32 : i32
      %93 = arith.select %91, %92, %90 : i32
      memref.store %93, %_anonymous2[%c3] : memref<4xi32>
      %94 = memref.load %idx_buffer_pv_0[%c0] : memref<2xi32>
      memref.store %94, %idx_buffer_pv_0[%c0] : memref<2xi32>
      memref.store %c0_i32, %idx_buffer_pv_0[%c0] : memref<2xi32>
      %95 = memref.load %idx_buffer_pv_0[%c1] : memref<2xi32>
      memref.store %95, %idx_buffer_pv_0[%c1] : memref<2xi32>
      aie.use_lock(%outOProj0_cons_lock_0, Release, 1)
      %96 = memref.load %_anonymous2[%c0] : memref<4xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous2[%c0] : memref<4xi32>
      %101 = arith.addi %2, %c1 : index
      cf.br ^bb3(%101 : index)
    ^bb48:  // pred: ^bb3
      %102 = arith.addi %0, %c1 : index
      cf.br ^bb1(%102 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_with = "ep_12h_512s_96e_4ph_8pa_4g_1pf_d9_m0_f6_n1-1_n21_me_8971c3b5f9ae_kernels.a", stack_size = 3328 : i32}
    %_anonymous3 = aie.buffer(%tile_0_5) {address = 45056 : i32, mem_bank = 2 : i32, sym_name = "_anonymous3"} : memref<3xi32> 
    %core_0_5 = aie.core(%tile_0_5) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c3 = arith.constant 3 : index
      %c8 = arith.constant 8 : index
      %c3072_i32 = arith.constant 3072 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous3[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous3[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous3[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb20
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb21
    ^bb2:  // pred: ^bb1
      func.call @zero_bf16_o_proj(%o_proj_zero_scratch_0) : (memref<32x96xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb19
      %3 = arith.cmpi slt, %2, %c3 : index
      cf.cond_br %3, ^bb4, ^bb20
    ^bb4:  // pred: ^bb3
      aie.use_lock(%outOProj0_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%outOProj0_buff_0 : memref<32x64xbf16>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%outOProj0_buff_1 : memref<32x64xbf16>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%outOProj0_buff_0 : memref<32x64xbf16>)
    ^bb8(%7: memref<32x64xbf16>):  // 3 preds: ^bb5, ^bb6, ^bb7
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb18
      %9 = arith.cmpi slt, %8, %c8 : index
      cf.cond_br %9, ^bb10, ^bb19
    ^bb10:  // pred: ^bb9
      aie.use_lock(%memOW0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%memOW0_cons_buff_0 : memref<64x96xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%memOW0_cons_buff_1 : memref<64x96xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%memOW0_cons_buff_0 : memref<64x96xbf16>)
    ^bb14(%13: memref<64x96xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @matmul_with_acc_bf16_bf16_o_proj(%7, %13, %o_proj_zero_scratch_0, %o_proj_partial_scratch_0) : (memref<32x64xbf16>, memref<64x96xbf16>, memref<32x96xbf16>, memref<32x96xbf16>) -> ()
      aie.use_lock(%outOGroupPart0_prod_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i32
      cf.switch %16 : i32, [
        default: ^bb17,
        0: ^bb15,
        1: ^bb16
      ]
    ^bb15:  // pred: ^bb14
      cf.br ^bb18(%outOGroupPart0_buff_0 : memref<32x96xbf16>)
    ^bb16:  // pred: ^bb14
      cf.br ^bb18(%outOGroupPart0_buff_1 : memref<32x96xbf16>)
    ^bb17:  // pred: ^bb14
      cf.br ^bb18(%outOGroupPart0_buff_0 : memref<32x96xbf16>)
    ^bb18(%17: memref<32x96xbf16>):  // 3 preds: ^bb15, ^bb16, ^bb17
      func.call @passThroughLine_o_proj(%o_proj_partial_scratch_0, %17, %c3072_i32) : (memref<32x96xbf16>, memref<32x96xbf16>, i32) -> ()
      aie.use_lock(%outOGroupPart0_cons_lock_0, Release, 1)
      %18 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%memOW0_cons_prod_lock_0, Release, 1)
      %23 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous3[%c1] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb9(%28 : index)
    ^bb19:  // pred: ^bb9
      aie.use_lock(%outOProj0_prod_lock_0, Release, 1)
      %29 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c2_i32 : i32
      %32 = arith.subi %30, %c2_i32 : i32
      %33 = arith.select %31, %32, %30 : i32
      memref.store %33, %_anonymous3[%c0] : memref<3xi32>
      %34 = arith.addi %2, %c1 : index
      cf.br ^bb3(%34 : index)
    ^bb20:  // pred: ^bb3
      %35 = arith.addi %0, %c1 : index
      cf.br ^bb1(%35 : index)
    ^bb21:  // pred: ^bb1
      aie.end
    } {link_with = "ep_12h_512s_96e_4ph_8pa_4g_1pf_d9_m0_f6_n1-1_n21_me_8971c3b5f9ae_kernels.a", stack_size = 3328 : i32}
    %_anonymous4 = aie.buffer(%tile_1_2) {address = 36864 : i32, mem_bank = 2 : i32, sym_name = "_anonymous4"} : memref<3xi32> 
    %core_1_2 = aie.core(%tile_1_2) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c3 = arith.constant 3 : index
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous4[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous4[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous4[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb20
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb21
    ^bb2:  // pred: ^bb1
      memref.store %c0_i32, %idx_buffer_qk_1[%c0] : memref<2xi32>
      memref.store %c0_i32, %idx_buffer_qk_1[%c1] : memref<2xi32>
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb19
      %3 = arith.cmpi slt, %2, %c3 : index
      cf.cond_br %3, ^bb4, ^bb20
    ^bb4:  // pred: ^bb3
      aie.use_lock(%memQ1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%memQ1_cons_buff_0 : memref<32x64xbf16>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%memQ1_cons_buff_1 : memref<32x64xbf16>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%memQ1_cons_buff_0 : memref<32x64xbf16>)
    ^bb8(%7: memref<32x64xbf16>):  // 3 preds: ^bb5, ^bb6, ^bb7
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb18
      %9 = arith.cmpi slt, %8, %c8 : index
      cf.cond_br %9, ^bb10, ^bb19
    ^bb10:  // pred: ^bb9
      aie.use_lock(%memK1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%memK1_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%memK1_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%memK1_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%13: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      aie.use_lock(%memA1_prod_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i32
      cf.switch %16 : i32, [
        default: ^bb17,
        0: ^bb15,
        1: ^bb16
      ]
    ^bb15:  // pred: ^bb14
      cf.br ^bb18(%memA1_buff_0 : memref<32x64xbf16>)
    ^bb16:  // pred: ^bb14
      cf.br ^bb18(%memA1_buff_1 : memref<32x64xbf16>)
    ^bb17:  // pred: ^bb14
      cf.br ^bb18(%memA1_buff_0 : memref<32x64xbf16>)
    ^bb18(%17: memref<32x64xbf16>):  // 3 preds: ^bb15, ^bb16, ^bb17
      func.call @zero_bf16(%17) : (memref<32x64xbf16>) -> ()
      func.call @matmul_bf16_bf16_wrapper(%7, %13, %17, %idx_buffer_qk_1) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<2xi32>) -> ()
      aie.use_lock(%memK1_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous4[%c1] : memref<3xi32>
      aie.use_lock(%memA1_cons_lock_0, Release, 1)
      %23 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous4[%c2] : memref<3xi32>
      %28 = memref.load %idx_buffer_qk_1[%c0] : memref<2xi32>
      memref.store %28, %idx_buffer_qk_1[%c0] : memref<2xi32>
      %29 = arith.addi %8, %c1 : index
      cf.br ^bb9(%29 : index)
    ^bb19:  // pred: ^bb9
      memref.store %c0_i32, %idx_buffer_qk_1[%c0] : memref<2xi32>
      %30 = memref.load %idx_buffer_qk_1[%c1] : memref<2xi32>
      memref.store %30, %idx_buffer_qk_1[%c1] : memref<2xi32>
      aie.use_lock(%memQ1_cons_prod_lock_0, Release, 1)
      %31 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %32 = arith.addi %31, %c1_i32 : i32
      %33 = arith.cmpi sge, %32, %c2_i32 : i32
      %34 = arith.subi %32, %c2_i32 : i32
      %35 = arith.select %33, %34, %32 : i32
      memref.store %35, %_anonymous4[%c0] : memref<3xi32>
      %36 = arith.addi %2, %c1 : index
      cf.br ^bb3(%36 : index)
    ^bb20:  // pred: ^bb3
      %37 = arith.addi %0, %c1 : index
      cf.br ^bb1(%37 : index)
    ^bb21:  // pred: ^bb1
      aie.end
    } {link_with = "ep_12h_512s_96e_4ph_8pa_4g_1pf_d9_m0_f6_n1-1_n21_me_8971c3b5f9ae_kernels.a", stack_size = 3328 : i32}
    %_anonymous5 = aie.buffer(%tile_1_3) {address = 53248 : i32, mem_bank = 3 : i32, sym_name = "_anonymous5"} : memref<3xi32> 
    %core_1_3 = aie.core(%tile_1_3) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c3 = arith.constant 3 : index
      %c32_i32 = arith.constant 32 : i32
      %c8 = arith.constant 8 : index
      %cst = arith.constant 1.806640e-01 : bf16
      %c64_i32 = arith.constant 64 : i32
      %c512_i32 = arith.constant 512 : i32
      %c128_i32 = arith.constant 128 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous5[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous5[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous5[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb20
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb21
    ^bb2:  // pred: ^bb1
      memref.store %c0_i32, %idx_buffer_softmax_1[%c0] : memref<2xi32>
      memref.store %c0_i32, %idx_buffer_softmax_1[%c1] : memref<2xi32>
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb19
      %3 = arith.cmpi slt, %2, %c3 : index
      cf.cond_br %3, ^bb4, ^bb20
    ^bb4:  // pred: ^bb3
      func.call @init_scale_buffer(%scale_buffer_softmax_1, %c32_i32) : (memref<128xbf16>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb18
      %5 = arith.cmpi slt, %4, %c8 : index
      cf.cond_br %5, ^bb6, ^bb19
    ^bb6:  // pred: ^bb5
      aie.use_lock(%memP1_prod_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%memP1_buff_0 : memref<32x64xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%memP1_buff_1 : memref<32x64xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%memP1_buff_0 : memref<32x64xbf16>)
    ^bb10(%9: memref<32x64xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%memA1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%memA1_cons_buff_0 : memref<32x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%memA1_cons_buff_1 : memref<32x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%memA1_cons_buff_0 : memref<32x64xbf16>)
    ^bb14(%13: memref<32x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      aie.use_lock(%scaleOF1_prod_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i32
      cf.switch %16 : i32, [
        default: ^bb17,
        0: ^bb15,
        1: ^bb16
      ]
    ^bb15:  // pred: ^bb14
      cf.br ^bb18(%scaleOF1_buff_0 : memref<128xbf16>)
    ^bb16:  // pred: ^bb14
      cf.br ^bb18(%scaleOF1_buff_1 : memref<128xbf16>)
    ^bb17:  // pred: ^bb14
      cf.br ^bb18(%scaleOF1_buff_0 : memref<128xbf16>)
    ^bb18(%17: memref<128xbf16>):  // 3 preds: ^bb15, ^bb16, ^bb17
      func.call @partial_softmax(%13, %9, %scale_buffer_softmax_1, %idx_buffer_softmax_1, %cst, %c32_i32, %c64_i32, %c512_i32, %c512_i32) : (memref<32x64xbf16>, memref<32x64xbf16>, memref<128xbf16>, memref<2xi32>, bf16, i32, i32, i32, i32) -> ()
      func.call @passThroughLine(%scale_buffer_softmax_1, %17, %c128_i32) : (memref<128xbf16>, memref<128xbf16>, i32) -> ()
      aie.use_lock(%memA1_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous5[%c1] : memref<3xi32>
      aie.use_lock(%memP1_cons_lock_0, Release, 1)
      %23 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous5[%c0] : memref<3xi32>
      aie.use_lock(%scaleOF1_cons_lock_0, Release, 1)
      %28 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %29 = arith.addi %28, %c1_i32 : i32
      %30 = arith.cmpi sge, %29, %c2_i32 : i32
      %31 = arith.subi %29, %c2_i32 : i32
      %32 = arith.select %30, %31, %29 : i32
      memref.store %32, %_anonymous5[%c2] : memref<3xi32>
      %33 = memref.load %idx_buffer_softmax_1[%c0] : memref<2xi32>
      memref.store %33, %idx_buffer_softmax_1[%c0] : memref<2xi32>
      %34 = arith.addi %4, %c1 : index
      cf.br ^bb5(%34 : index)
    ^bb19:  // pred: ^bb5
      memref.store %c0_i32, %idx_buffer_softmax_1[%c0] : memref<2xi32>
      %35 = memref.load %idx_buffer_softmax_1[%c1] : memref<2xi32>
      memref.store %35, %idx_buffer_softmax_1[%c1] : memref<2xi32>
      %36 = arith.addi %2, %c1 : index
      cf.br ^bb3(%36 : index)
    ^bb20:  // pred: ^bb3
      %37 = arith.addi %0, %c1 : index
      cf.br ^bb1(%37 : index)
    ^bb21:  // pred: ^bb1
      aie.end
    } {link_with = "ep_12h_512s_96e_4ph_8pa_4g_1pf_d9_m0_f6_n1-1_n21_me_8971c3b5f9ae_kernels.a", stack_size = 3328 : i32}
    %_anonymous6 = aie.buffer(%tile_1_4) {address = 36864 : i32, mem_bank = 2 : i32, sym_name = "_anonymous6"} : memref<4xi32> 
    %core_1_4 = aie.core(%tile_1_4) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c32_i32 = arith.constant 32 : i32
      %c6 = arith.constant 6 : index
      %c1_i32 = arith.constant 1 : i32
      %c3 = arith.constant 3 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous6[%c0] : memref<4xi32>
      memref.store %c0_i32, %_anonymous6[%c1] : memref<4xi32>
      memref.store %c0_i32, %_anonymous6[%c2] : memref<4xi32>
      memref.store %c0_i32, %_anonymous6[%c3] : memref<4xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      memref.store %c0_i32, %idx_buffer_pv_1[%c0] : memref<2xi32>
      memref.store %c0_i32, %idx_buffer_pv_1[%c1] : memref<2xi32>
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb47
      %3 = arith.cmpi slt, %2, %c3 : index
      cf.cond_br %3, ^bb4, ^bb48
    ^bb4:  // pred: ^bb3
      aie.use_lock(%outOProj1_prod_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous6[%c0] : memref<4xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%outOProj1_buff_0 : memref<32x64xbf16>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%outOProj1_buff_1 : memref<32x64xbf16>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%outOProj1_buff_0 : memref<32x64xbf16>)
    ^bb8(%7: memref<32x64xbf16>):  // 3 preds: ^bb5, ^bb6, ^bb7
      func.call @zero_bf16(%7) : (memref<32x64xbf16>) -> ()
      aie.use_lock(%memP1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous6[%c1] : memref<4xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%memP1_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%memP1_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%memP1_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      aie.use_lock(%memV1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %12 = memref.load %_anonymous6[%c2] : memref<4xi32>
      %13 = arith.index_cast %12 : i32 to index
      %14 = arith.index_cast %13 : index to i32
      cf.switch %14 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%memV1_cons_buff_0 : memref<64x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%memV1_cons_buff_1 : memref<64x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%memV1_cons_buff_0 : memref<64x64xbf16>)
    ^bb16(%15: memref<64x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      aie.use_lock(%scaleOF1_cons_lock_0, AcquireGreaterEqual, 1)
      %16 = memref.load %_anonymous6[%c3] : memref<4xi32>
      %17 = arith.index_cast %16 : i32 to index
      %18 = arith.index_cast %17 : index to i32
      cf.switch %18 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%scaleOF1_buff_0 : memref<128xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%scaleOF1_buff_1 : memref<128xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%scaleOF1_buff_0 : memref<128xbf16>)
    ^bb20(%19: memref<128xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_PV(%11, %15, %7, %19, %c32_i32, %c0_i32, %idx_buffer_pv_1) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<128xbf16>, i32, i32, memref<2xi32>) -> ()
      aie.use_lock(%memP1_cons_prod_lock_0, Release, 1)
      %20 = memref.load %_anonymous6[%c1] : memref<4xi32>
      %21 = arith.addi %20, %c1_i32 : i32
      %22 = arith.cmpi sge, %21, %c2_i32 : i32
      %23 = arith.subi %21, %c2_i32 : i32
      %24 = arith.select %22, %23, %21 : i32
      memref.store %24, %_anonymous6[%c1] : memref<4xi32>
      aie.use_lock(%memV1_cons_prod_lock_0, Release, 1)
      %25 = memref.load %_anonymous6[%c2] : memref<4xi32>
      %26 = arith.addi %25, %c1_i32 : i32
      %27 = arith.cmpi sge, %26, %c2_i32 : i32
      %28 = arith.subi %26, %c2_i32 : i32
      %29 = arith.select %27, %28, %26 : i32
      memref.store %29, %_anonymous6[%c2] : memref<4xi32>
      aie.use_lock(%scaleOF1_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous6[%c3] : memref<4xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous6[%c3] : memref<4xi32>
      %35 = memref.load %idx_buffer_pv_1[%c0] : memref<2xi32>
      memref.store %35, %idx_buffer_pv_1[%c0] : memref<2xi32>
      cf.br ^bb21(%c0 : index)
    ^bb21(%36: index):  // 2 preds: ^bb20, ^bb34
      %37 = arith.cmpi slt, %36, %c6 : index
      cf.cond_br %37, ^bb22, ^bb35
    ^bb22:  // pred: ^bb21
      aie.use_lock(%memP1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %38 = memref.load %_anonymous6[%c1] : memref<4xi32>
      %39 = arith.index_cast %38 : i32 to index
      %40 = arith.index_cast %39 : index to i32
      cf.switch %40 : i32, [
        default: ^bb25,
        0: ^bb23,
        1: ^bb24
      ]
    ^bb23:  // pred: ^bb22
      cf.br ^bb26(%memP1_cons_buff_0 : memref<32x64xbf16>)
    ^bb24:  // pred: ^bb22
      cf.br ^bb26(%memP1_cons_buff_1 : memref<32x64xbf16>)
    ^bb25:  // pred: ^bb22
      cf.br ^bb26(%memP1_cons_buff_0 : memref<32x64xbf16>)
    ^bb26(%41: memref<32x64xbf16>):  // 3 preds: ^bb23, ^bb24, ^bb25
      aie.use_lock(%memV1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %42 = memref.load %_anonymous6[%c2] : memref<4xi32>
      %43 = arith.index_cast %42 : i32 to index
      %44 = arith.index_cast %43 : index to i32
      cf.switch %44 : i32, [
        default: ^bb29,
        0: ^bb27,
        1: ^bb28
      ]
    ^bb27:  // pred: ^bb26
      cf.br ^bb30(%memV1_cons_buff_0 : memref<64x64xbf16>)
    ^bb28:  // pred: ^bb26
      cf.br ^bb30(%memV1_cons_buff_1 : memref<64x64xbf16>)
    ^bb29:  // pred: ^bb26
      cf.br ^bb30(%memV1_cons_buff_0 : memref<64x64xbf16>)
    ^bb30(%45: memref<64x64xbf16>):  // 3 preds: ^bb27, ^bb28, ^bb29
      aie.use_lock(%scaleOF1_cons_lock_0, AcquireGreaterEqual, 1)
      %46 = memref.load %_anonymous6[%c3] : memref<4xi32>
      %47 = arith.index_cast %46 : i32 to index
      %48 = arith.index_cast %47 : index to i32
      cf.switch %48 : i32, [
        default: ^bb33,
        0: ^bb31,
        1: ^bb32
      ]
    ^bb31:  // pred: ^bb30
      cf.br ^bb34(%scaleOF1_buff_0 : memref<128xbf16>)
    ^bb32:  // pred: ^bb30
      cf.br ^bb34(%scaleOF1_buff_1 : memref<128xbf16>)
    ^bb33:  // pred: ^bb30
      cf.br ^bb34(%scaleOF1_buff_0 : memref<128xbf16>)
    ^bb34(%49: memref<128xbf16>):  // 3 preds: ^bb31, ^bb32, ^bb33
      func.call @matmul_PV(%41, %45, %7, %49, %c32_i32, %c1_i32, %idx_buffer_pv_1) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<128xbf16>, i32, i32, memref<2xi32>) -> ()
      aie.use_lock(%memP1_cons_prod_lock_0, Release, 1)
      %50 = memref.load %_anonymous6[%c1] : memref<4xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c2_i32 : i32
      %53 = arith.subi %51, %c2_i32 : i32
      %54 = arith.select %52, %53, %51 : i32
      memref.store %54, %_anonymous6[%c1] : memref<4xi32>
      aie.use_lock(%memV1_cons_prod_lock_0, Release, 1)
      %55 = memref.load %_anonymous6[%c2] : memref<4xi32>
      %56 = arith.addi %55, %c1_i32 : i32
      %57 = arith.cmpi sge, %56, %c2_i32 : i32
      %58 = arith.subi %56, %c2_i32 : i32
      %59 = arith.select %57, %58, %56 : i32
      memref.store %59, %_anonymous6[%c2] : memref<4xi32>
      aie.use_lock(%scaleOF1_prod_lock_0, Release, 1)
      %60 = memref.load %_anonymous6[%c3] : memref<4xi32>
      %61 = arith.addi %60, %c1_i32 : i32
      %62 = arith.cmpi sge, %61, %c2_i32 : i32
      %63 = arith.subi %61, %c2_i32 : i32
      %64 = arith.select %62, %63, %61 : i32
      memref.store %64, %_anonymous6[%c3] : memref<4xi32>
      %65 = memref.load %idx_buffer_pv_1[%c0] : memref<2xi32>
      memref.store %65, %idx_buffer_pv_1[%c0] : memref<2xi32>
      %66 = arith.addi %36, %c1 : index
      cf.br ^bb21(%66 : index)
    ^bb35:  // pred: ^bb21
      aie.use_lock(%memP1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %67 = memref.load %_anonymous6[%c1] : memref<4xi32>
      %68 = arith.index_cast %67 : i32 to index
      %69 = arith.index_cast %68 : index to i32
      cf.switch %69 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%memP1_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%memP1_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%memP1_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%70: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      aie.use_lock(%memV1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %71 = memref.load %_anonymous6[%c2] : memref<4xi32>
      %72 = arith.index_cast %71 : i32 to index
      %73 = arith.index_cast %72 : index to i32
      cf.switch %73 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%memV1_cons_buff_0 : memref<64x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%memV1_cons_buff_1 : memref<64x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%memV1_cons_buff_0 : memref<64x64xbf16>)
    ^bb43(%74: memref<64x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      aie.use_lock(%scaleOF1_cons_lock_0, AcquireGreaterEqual, 1)
      %75 = memref.load %_anonymous6[%c3] : memref<4xi32>
      %76 = arith.index_cast %75 : i32 to index
      %77 = arith.index_cast %76 : index to i32
      cf.switch %77 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%scaleOF1_buff_0 : memref<128xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%scaleOF1_buff_1 : memref<128xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%scaleOF1_buff_0 : memref<128xbf16>)
    ^bb47(%78: memref<128xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_PV(%70, %74, %7, %78, %c32_i32, %c1_i32, %idx_buffer_pv_1) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<128xbf16>, i32, i32, memref<2xi32>) -> ()
      func.call @rescale_O(%7, %78, %c32_i32, %idx_buffer_pv_1) : (memref<32x64xbf16>, memref<128xbf16>, i32, memref<2xi32>) -> ()
      aie.use_lock(%memP1_cons_prod_lock_0, Release, 1)
      %79 = memref.load %_anonymous6[%c1] : memref<4xi32>
      %80 = arith.addi %79, %c1_i32 : i32
      %81 = arith.cmpi sge, %80, %c2_i32 : i32
      %82 = arith.subi %80, %c2_i32 : i32
      %83 = arith.select %81, %82, %80 : i32
      memref.store %83, %_anonymous6[%c1] : memref<4xi32>
      aie.use_lock(%memV1_cons_prod_lock_0, Release, 1)
      %84 = memref.load %_anonymous6[%c2] : memref<4xi32>
      %85 = arith.addi %84, %c1_i32 : i32
      %86 = arith.cmpi sge, %85, %c2_i32 : i32
      %87 = arith.subi %85, %c2_i32 : i32
      %88 = arith.select %86, %87, %85 : i32
      memref.store %88, %_anonymous6[%c2] : memref<4xi32>
      aie.use_lock(%scaleOF1_prod_lock_0, Release, 1)
      %89 = memref.load %_anonymous6[%c3] : memref<4xi32>
      %90 = arith.addi %89, %c1_i32 : i32
      %91 = arith.cmpi sge, %90, %c2_i32 : i32
      %92 = arith.subi %90, %c2_i32 : i32
      %93 = arith.select %91, %92, %90 : i32
      memref.store %93, %_anonymous6[%c3] : memref<4xi32>
      %94 = memref.load %idx_buffer_pv_1[%c0] : memref<2xi32>
      memref.store %94, %idx_buffer_pv_1[%c0] : memref<2xi32>
      memref.store %c0_i32, %idx_buffer_pv_1[%c0] : memref<2xi32>
      %95 = memref.load %idx_buffer_pv_1[%c1] : memref<2xi32>
      memref.store %95, %idx_buffer_pv_1[%c1] : memref<2xi32>
      aie.use_lock(%outOProj1_cons_lock_0, Release, 1)
      %96 = memref.load %_anonymous6[%c0] : memref<4xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous6[%c0] : memref<4xi32>
      %101 = arith.addi %2, %c1 : index
      cf.br ^bb3(%101 : index)
    ^bb48:  // pred: ^bb3
      %102 = arith.addi %0, %c1 : index
      cf.br ^bb1(%102 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_with = "ep_12h_512s_96e_4ph_8pa_4g_1pf_d9_m0_f6_n1-1_n21_me_8971c3b5f9ae_kernels.a", stack_size = 3328 : i32}
    %_anonymous7 = aie.buffer(%tile_1_5) {address = 45056 : i32, mem_bank = 2 : i32, sym_name = "_anonymous7"} : memref<4xi32> 
    %core_1_5 = aie.core(%tile_1_5) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c8 = arith.constant 8 : index
      %c3072_i32 = arith.constant 3072 : i32
      %c3 = arith.constant 3 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous7[%c0] : memref<4xi32>
      memref.store %c0_i32, %_anonymous7[%c1] : memref<4xi32>
      memref.store %c0_i32, %_anonymous7[%c2] : memref<4xi32>
      memref.store %c0_i32, %_anonymous7[%c3] : memref<4xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb24
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb25
    ^bb2:  // pred: ^bb1
      func.call @zero_bf16_o_proj(%o_proj_zero_scratch_1) : (memref<32x96xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb23
      %3 = arith.cmpi slt, %2, %c3 : index
      cf.cond_br %3, ^bb4, ^bb24
    ^bb4:  // pred: ^bb3
      aie.use_lock(%outOProj1_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous7[%c0] : memref<4xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%outOProj1_buff_0 : memref<32x64xbf16>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%outOProj1_buff_1 : memref<32x64xbf16>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%outOProj1_buff_0 : memref<32x64xbf16>)
    ^bb8(%7: memref<32x64xbf16>):  // 3 preds: ^bb5, ^bb6, ^bb7
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb22
      %9 = arith.cmpi slt, %8, %c8 : index
      cf.cond_br %9, ^bb10, ^bb23
    ^bb10:  // pred: ^bb9
      aie.use_lock(%memOW1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous7[%c1] : memref<4xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%memOW1_cons_buff_0 : memref<64x96xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%memOW1_cons_buff_1 : memref<64x96xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%memOW1_cons_buff_0 : memref<64x96xbf16>)
    ^bb14(%13: memref<64x96xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @matmul_with_acc_bf16_bf16_o_proj(%7, %13, %o_proj_zero_scratch_1, %o_proj_partial_scratch_1) : (memref<32x64xbf16>, memref<64x96xbf16>, memref<32x96xbf16>, memref<32x96xbf16>) -> ()
      aie.use_lock(%outOGroupPart0_cons_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous7[%c2] : memref<4xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i32
      cf.switch %16 : i32, [
        default: ^bb17,
        0: ^bb15,
        1: ^bb16
      ]
    ^bb15:  // pred: ^bb14
      cf.br ^bb18(%outOGroupPart0_buff_0 : memref<32x96xbf16>)
    ^bb16:  // pred: ^bb14
      cf.br ^bb18(%outOGroupPart0_buff_1 : memref<32x96xbf16>)
    ^bb17:  // pred: ^bb14
      cf.br ^bb18(%outOGroupPart0_buff_0 : memref<32x96xbf16>)
    ^bb18(%17: memref<32x96xbf16>):  // 3 preds: ^bb15, ^bb16, ^bb17
      func.call @eltwise_add_bf16_vector_o_proj(%17, %o_proj_partial_scratch_1, %o_proj_partial_scratch_1, %c3072_i32) : (memref<32x96xbf16>, memref<32x96xbf16>, memref<32x96xbf16>, i32) -> ()
      aie.use_lock(%outOGroupPart0_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous7[%c2] : memref<4xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous7[%c2] : memref<4xi32>
      aie.use_lock(%outOGroupPart1_prod_lock_0, AcquireGreaterEqual, 1)
      %23 = memref.load %_anonymous7[%c3] : memref<4xi32>
      %24 = arith.index_cast %23 : i32 to index
      %25 = arith.index_cast %24 : index to i32
      cf.switch %25 : i32, [
        default: ^bb21,
        0: ^bb19,
        1: ^bb20
      ]
    ^bb19:  // pred: ^bb18
      cf.br ^bb22(%outOGroupPart1_buff_0 : memref<32x96xbf16>)
    ^bb20:  // pred: ^bb18
      cf.br ^bb22(%outOGroupPart1_buff_1 : memref<32x96xbf16>)
    ^bb21:  // pred: ^bb18
      cf.br ^bb22(%outOGroupPart1_buff_0 : memref<32x96xbf16>)
    ^bb22(%26: memref<32x96xbf16>):  // 3 preds: ^bb19, ^bb20, ^bb21
      func.call @passThroughLine_o_proj(%o_proj_partial_scratch_1, %26, %c3072_i32) : (memref<32x96xbf16>, memref<32x96xbf16>, i32) -> ()
      aie.use_lock(%outOGroupPart1_cons_lock_0, Release, 1)
      %27 = memref.load %_anonymous7[%c3] : memref<4xi32>
      %28 = arith.addi %27, %c1_i32 : i32
      %29 = arith.cmpi sge, %28, %c2_i32 : i32
      %30 = arith.subi %28, %c2_i32 : i32
      %31 = arith.select %29, %30, %28 : i32
      memref.store %31, %_anonymous7[%c3] : memref<4xi32>
      aie.use_lock(%memOW1_cons_prod_lock_0, Release, 1)
      %32 = memref.load %_anonymous7[%c1] : memref<4xi32>
      %33 = arith.addi %32, %c1_i32 : i32
      %34 = arith.cmpi sge, %33, %c2_i32 : i32
      %35 = arith.subi %33, %c2_i32 : i32
      %36 = arith.select %34, %35, %33 : i32
      memref.store %36, %_anonymous7[%c1] : memref<4xi32>
      %37 = arith.addi %8, %c1 : index
      cf.br ^bb9(%37 : index)
    ^bb23:  // pred: ^bb9
      aie.use_lock(%outOProj1_prod_lock_0, Release, 1)
      %38 = memref.load %_anonymous7[%c0] : memref<4xi32>
      %39 = arith.addi %38, %c1_i32 : i32
      %40 = arith.cmpi sge, %39, %c2_i32 : i32
      %41 = arith.subi %39, %c2_i32 : i32
      %42 = arith.select %40, %41, %39 : i32
      memref.store %42, %_anonymous7[%c0] : memref<4xi32>
      %43 = arith.addi %2, %c1 : index
      cf.br ^bb3(%43 : index)
    ^bb24:  // pred: ^bb3
      %44 = arith.addi %0, %c1 : index
      cf.br ^bb1(%44 : index)
    ^bb25:  // pred: ^bb1
      aie.end
    } {link_with = "ep_12h_512s_96e_4ph_8pa_4g_1pf_d9_m0_f6_n1-1_n21_me_8971c3b5f9ae_kernels.a", stack_size = 3328 : i32}
    %_anonymous8 = aie.buffer(%tile_2_2) {address = 36864 : i32, mem_bank = 2 : i32, sym_name = "_anonymous8"} : memref<3xi32> 
    %core_2_2 = aie.core(%tile_2_2) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c3 = arith.constant 3 : index
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous8[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous8[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous8[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb20
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb21
    ^bb2:  // pred: ^bb1
      memref.store %c0_i32, %idx_buffer_qk_2[%c0] : memref<2xi32>
      memref.store %c0_i32, %idx_buffer_qk_2[%c1] : memref<2xi32>
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb19
      %3 = arith.cmpi slt, %2, %c3 : index
      cf.cond_br %3, ^bb4, ^bb20
    ^bb4:  // pred: ^bb3
      aie.use_lock(%memQ2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous8[%c0] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%memQ2_cons_buff_0 : memref<32x64xbf16>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%memQ2_cons_buff_1 : memref<32x64xbf16>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%memQ2_cons_buff_0 : memref<32x64xbf16>)
    ^bb8(%7: memref<32x64xbf16>):  // 3 preds: ^bb5, ^bb6, ^bb7
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb18
      %9 = arith.cmpi slt, %8, %c8 : index
      cf.cond_br %9, ^bb10, ^bb19
    ^bb10:  // pred: ^bb9
      aie.use_lock(%memK2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous8[%c1] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%memK2_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%memK2_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%memK2_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%13: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      aie.use_lock(%memA2_prod_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous8[%c2] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i32
      cf.switch %16 : i32, [
        default: ^bb17,
        0: ^bb15,
        1: ^bb16
      ]
    ^bb15:  // pred: ^bb14
      cf.br ^bb18(%memA2_buff_0 : memref<32x64xbf16>)
    ^bb16:  // pred: ^bb14
      cf.br ^bb18(%memA2_buff_1 : memref<32x64xbf16>)
    ^bb17:  // pred: ^bb14
      cf.br ^bb18(%memA2_buff_0 : memref<32x64xbf16>)
    ^bb18(%17: memref<32x64xbf16>):  // 3 preds: ^bb15, ^bb16, ^bb17
      func.call @zero_bf16(%17) : (memref<32x64xbf16>) -> ()
      func.call @matmul_bf16_bf16_wrapper(%7, %13, %17, %idx_buffer_qk_2) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<2xi32>) -> ()
      aie.use_lock(%memK2_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous8[%c1] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous8[%c1] : memref<3xi32>
      aie.use_lock(%memA2_cons_lock_0, Release, 1)
      %23 = memref.load %_anonymous8[%c2] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous8[%c2] : memref<3xi32>
      %28 = memref.load %idx_buffer_qk_2[%c0] : memref<2xi32>
      memref.store %28, %idx_buffer_qk_2[%c0] : memref<2xi32>
      %29 = arith.addi %8, %c1 : index
      cf.br ^bb9(%29 : index)
    ^bb19:  // pred: ^bb9
      memref.store %c0_i32, %idx_buffer_qk_2[%c0] : memref<2xi32>
      %30 = memref.load %idx_buffer_qk_2[%c1] : memref<2xi32>
      memref.store %30, %idx_buffer_qk_2[%c1] : memref<2xi32>
      aie.use_lock(%memQ2_cons_prod_lock_0, Release, 1)
      %31 = memref.load %_anonymous8[%c0] : memref<3xi32>
      %32 = arith.addi %31, %c1_i32 : i32
      %33 = arith.cmpi sge, %32, %c2_i32 : i32
      %34 = arith.subi %32, %c2_i32 : i32
      %35 = arith.select %33, %34, %32 : i32
      memref.store %35, %_anonymous8[%c0] : memref<3xi32>
      %36 = arith.addi %2, %c1 : index
      cf.br ^bb3(%36 : index)
    ^bb20:  // pred: ^bb3
      %37 = arith.addi %0, %c1 : index
      cf.br ^bb1(%37 : index)
    ^bb21:  // pred: ^bb1
      aie.end
    } {link_with = "ep_12h_512s_96e_4ph_8pa_4g_1pf_d9_m0_f6_n1-1_n21_me_8971c3b5f9ae_kernels.a", stack_size = 3328 : i32}
    %_anonymous9 = aie.buffer(%tile_2_3) {address = 53248 : i32, mem_bank = 3 : i32, sym_name = "_anonymous9"} : memref<3xi32> 
    %core_2_3 = aie.core(%tile_2_3) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c3 = arith.constant 3 : index
      %c32_i32 = arith.constant 32 : i32
      %c8 = arith.constant 8 : index
      %cst = arith.constant 1.806640e-01 : bf16
      %c64_i32 = arith.constant 64 : i32
      %c512_i32 = arith.constant 512 : i32
      %c128_i32 = arith.constant 128 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous9[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous9[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous9[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb20
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb21
    ^bb2:  // pred: ^bb1
      memref.store %c0_i32, %idx_buffer_softmax_2[%c0] : memref<2xi32>
      memref.store %c0_i32, %idx_buffer_softmax_2[%c1] : memref<2xi32>
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb19
      %3 = arith.cmpi slt, %2, %c3 : index
      cf.cond_br %3, ^bb4, ^bb20
    ^bb4:  // pred: ^bb3
      func.call @init_scale_buffer(%scale_buffer_softmax_2, %c32_i32) : (memref<128xbf16>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb18
      %5 = arith.cmpi slt, %4, %c8 : index
      cf.cond_br %5, ^bb6, ^bb19
    ^bb6:  // pred: ^bb5
      aie.use_lock(%memP2_prod_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous9[%c0] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%memP2_buff_0 : memref<32x64xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%memP2_buff_1 : memref<32x64xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%memP2_buff_0 : memref<32x64xbf16>)
    ^bb10(%9: memref<32x64xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%memA2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous9[%c1] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%memA2_cons_buff_0 : memref<32x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%memA2_cons_buff_1 : memref<32x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%memA2_cons_buff_0 : memref<32x64xbf16>)
    ^bb14(%13: memref<32x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      aie.use_lock(%scaleOF2_prod_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous9[%c2] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i32
      cf.switch %16 : i32, [
        default: ^bb17,
        0: ^bb15,
        1: ^bb16
      ]
    ^bb15:  // pred: ^bb14
      cf.br ^bb18(%scaleOF2_buff_0 : memref<128xbf16>)
    ^bb16:  // pred: ^bb14
      cf.br ^bb18(%scaleOF2_buff_1 : memref<128xbf16>)
    ^bb17:  // pred: ^bb14
      cf.br ^bb18(%scaleOF2_buff_0 : memref<128xbf16>)
    ^bb18(%17: memref<128xbf16>):  // 3 preds: ^bb15, ^bb16, ^bb17
      func.call @partial_softmax(%13, %9, %scale_buffer_softmax_2, %idx_buffer_softmax_2, %cst, %c32_i32, %c64_i32, %c512_i32, %c512_i32) : (memref<32x64xbf16>, memref<32x64xbf16>, memref<128xbf16>, memref<2xi32>, bf16, i32, i32, i32, i32) -> ()
      func.call @passThroughLine(%scale_buffer_softmax_2, %17, %c128_i32) : (memref<128xbf16>, memref<128xbf16>, i32) -> ()
      aie.use_lock(%memA2_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous9[%c1] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous9[%c1] : memref<3xi32>
      aie.use_lock(%memP2_cons_lock_0, Release, 1)
      %23 = memref.load %_anonymous9[%c0] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous9[%c0] : memref<3xi32>
      aie.use_lock(%scaleOF2_cons_lock_0, Release, 1)
      %28 = memref.load %_anonymous9[%c2] : memref<3xi32>
      %29 = arith.addi %28, %c1_i32 : i32
      %30 = arith.cmpi sge, %29, %c2_i32 : i32
      %31 = arith.subi %29, %c2_i32 : i32
      %32 = arith.select %30, %31, %29 : i32
      memref.store %32, %_anonymous9[%c2] : memref<3xi32>
      %33 = memref.load %idx_buffer_softmax_2[%c0] : memref<2xi32>
      memref.store %33, %idx_buffer_softmax_2[%c0] : memref<2xi32>
      %34 = arith.addi %4, %c1 : index
      cf.br ^bb5(%34 : index)
    ^bb19:  // pred: ^bb5
      memref.store %c0_i32, %idx_buffer_softmax_2[%c0] : memref<2xi32>
      %35 = memref.load %idx_buffer_softmax_2[%c1] : memref<2xi32>
      memref.store %35, %idx_buffer_softmax_2[%c1] : memref<2xi32>
      %36 = arith.addi %2, %c1 : index
      cf.br ^bb3(%36 : index)
    ^bb20:  // pred: ^bb3
      %37 = arith.addi %0, %c1 : index
      cf.br ^bb1(%37 : index)
    ^bb21:  // pred: ^bb1
      aie.end
    } {link_with = "ep_12h_512s_96e_4ph_8pa_4g_1pf_d9_m0_f6_n1-1_n21_me_8971c3b5f9ae_kernels.a", stack_size = 3328 : i32}
    %_anonymous10 = aie.buffer(%tile_2_4) {address = 36864 : i32, mem_bank = 2 : i32, sym_name = "_anonymous10"} : memref<4xi32> 
    %core_2_4 = aie.core(%tile_2_4) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c32_i32 = arith.constant 32 : i32
      %c6 = arith.constant 6 : index
      %c1_i32 = arith.constant 1 : i32
      %c3 = arith.constant 3 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous10[%c0] : memref<4xi32>
      memref.store %c0_i32, %_anonymous10[%c1] : memref<4xi32>
      memref.store %c0_i32, %_anonymous10[%c2] : memref<4xi32>
      memref.store %c0_i32, %_anonymous10[%c3] : memref<4xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      memref.store %c0_i32, %idx_buffer_pv_2[%c0] : memref<2xi32>
      memref.store %c0_i32, %idx_buffer_pv_2[%c1] : memref<2xi32>
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb47
      %3 = arith.cmpi slt, %2, %c3 : index
      cf.cond_br %3, ^bb4, ^bb48
    ^bb4:  // pred: ^bb3
      aie.use_lock(%outOProj2_prod_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous10[%c0] : memref<4xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%outOProj2_buff_0 : memref<32x64xbf16>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%outOProj2_buff_1 : memref<32x64xbf16>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%outOProj2_buff_0 : memref<32x64xbf16>)
    ^bb8(%7: memref<32x64xbf16>):  // 3 preds: ^bb5, ^bb6, ^bb7
      func.call @zero_bf16(%7) : (memref<32x64xbf16>) -> ()
      aie.use_lock(%memP2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous10[%c1] : memref<4xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%memP2_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%memP2_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%memP2_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      aie.use_lock(%memV2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %12 = memref.load %_anonymous10[%c2] : memref<4xi32>
      %13 = arith.index_cast %12 : i32 to index
      %14 = arith.index_cast %13 : index to i32
      cf.switch %14 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%memV2_cons_buff_0 : memref<64x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%memV2_cons_buff_1 : memref<64x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%memV2_cons_buff_0 : memref<64x64xbf16>)
    ^bb16(%15: memref<64x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      aie.use_lock(%scaleOF2_cons_lock_0, AcquireGreaterEqual, 1)
      %16 = memref.load %_anonymous10[%c3] : memref<4xi32>
      %17 = arith.index_cast %16 : i32 to index
      %18 = arith.index_cast %17 : index to i32
      cf.switch %18 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%scaleOF2_buff_0 : memref<128xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%scaleOF2_buff_1 : memref<128xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%scaleOF2_buff_0 : memref<128xbf16>)
    ^bb20(%19: memref<128xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_PV(%11, %15, %7, %19, %c32_i32, %c0_i32, %idx_buffer_pv_2) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<128xbf16>, i32, i32, memref<2xi32>) -> ()
      aie.use_lock(%memP2_cons_prod_lock_0, Release, 1)
      %20 = memref.load %_anonymous10[%c1] : memref<4xi32>
      %21 = arith.addi %20, %c1_i32 : i32
      %22 = arith.cmpi sge, %21, %c2_i32 : i32
      %23 = arith.subi %21, %c2_i32 : i32
      %24 = arith.select %22, %23, %21 : i32
      memref.store %24, %_anonymous10[%c1] : memref<4xi32>
      aie.use_lock(%memV2_cons_prod_lock_0, Release, 1)
      %25 = memref.load %_anonymous10[%c2] : memref<4xi32>
      %26 = arith.addi %25, %c1_i32 : i32
      %27 = arith.cmpi sge, %26, %c2_i32 : i32
      %28 = arith.subi %26, %c2_i32 : i32
      %29 = arith.select %27, %28, %26 : i32
      memref.store %29, %_anonymous10[%c2] : memref<4xi32>
      aie.use_lock(%scaleOF2_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous10[%c3] : memref<4xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous10[%c3] : memref<4xi32>
      %35 = memref.load %idx_buffer_pv_2[%c0] : memref<2xi32>
      memref.store %35, %idx_buffer_pv_2[%c0] : memref<2xi32>
      cf.br ^bb21(%c0 : index)
    ^bb21(%36: index):  // 2 preds: ^bb20, ^bb34
      %37 = arith.cmpi slt, %36, %c6 : index
      cf.cond_br %37, ^bb22, ^bb35
    ^bb22:  // pred: ^bb21
      aie.use_lock(%memP2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %38 = memref.load %_anonymous10[%c1] : memref<4xi32>
      %39 = arith.index_cast %38 : i32 to index
      %40 = arith.index_cast %39 : index to i32
      cf.switch %40 : i32, [
        default: ^bb25,
        0: ^bb23,
        1: ^bb24
      ]
    ^bb23:  // pred: ^bb22
      cf.br ^bb26(%memP2_cons_buff_0 : memref<32x64xbf16>)
    ^bb24:  // pred: ^bb22
      cf.br ^bb26(%memP2_cons_buff_1 : memref<32x64xbf16>)
    ^bb25:  // pred: ^bb22
      cf.br ^bb26(%memP2_cons_buff_0 : memref<32x64xbf16>)
    ^bb26(%41: memref<32x64xbf16>):  // 3 preds: ^bb23, ^bb24, ^bb25
      aie.use_lock(%memV2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %42 = memref.load %_anonymous10[%c2] : memref<4xi32>
      %43 = arith.index_cast %42 : i32 to index
      %44 = arith.index_cast %43 : index to i32
      cf.switch %44 : i32, [
        default: ^bb29,
        0: ^bb27,
        1: ^bb28
      ]
    ^bb27:  // pred: ^bb26
      cf.br ^bb30(%memV2_cons_buff_0 : memref<64x64xbf16>)
    ^bb28:  // pred: ^bb26
      cf.br ^bb30(%memV2_cons_buff_1 : memref<64x64xbf16>)
    ^bb29:  // pred: ^bb26
      cf.br ^bb30(%memV2_cons_buff_0 : memref<64x64xbf16>)
    ^bb30(%45: memref<64x64xbf16>):  // 3 preds: ^bb27, ^bb28, ^bb29
      aie.use_lock(%scaleOF2_cons_lock_0, AcquireGreaterEqual, 1)
      %46 = memref.load %_anonymous10[%c3] : memref<4xi32>
      %47 = arith.index_cast %46 : i32 to index
      %48 = arith.index_cast %47 : index to i32
      cf.switch %48 : i32, [
        default: ^bb33,
        0: ^bb31,
        1: ^bb32
      ]
    ^bb31:  // pred: ^bb30
      cf.br ^bb34(%scaleOF2_buff_0 : memref<128xbf16>)
    ^bb32:  // pred: ^bb30
      cf.br ^bb34(%scaleOF2_buff_1 : memref<128xbf16>)
    ^bb33:  // pred: ^bb30
      cf.br ^bb34(%scaleOF2_buff_0 : memref<128xbf16>)
    ^bb34(%49: memref<128xbf16>):  // 3 preds: ^bb31, ^bb32, ^bb33
      func.call @matmul_PV(%41, %45, %7, %49, %c32_i32, %c1_i32, %idx_buffer_pv_2) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<128xbf16>, i32, i32, memref<2xi32>) -> ()
      aie.use_lock(%memP2_cons_prod_lock_0, Release, 1)
      %50 = memref.load %_anonymous10[%c1] : memref<4xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c2_i32 : i32
      %53 = arith.subi %51, %c2_i32 : i32
      %54 = arith.select %52, %53, %51 : i32
      memref.store %54, %_anonymous10[%c1] : memref<4xi32>
      aie.use_lock(%memV2_cons_prod_lock_0, Release, 1)
      %55 = memref.load %_anonymous10[%c2] : memref<4xi32>
      %56 = arith.addi %55, %c1_i32 : i32
      %57 = arith.cmpi sge, %56, %c2_i32 : i32
      %58 = arith.subi %56, %c2_i32 : i32
      %59 = arith.select %57, %58, %56 : i32
      memref.store %59, %_anonymous10[%c2] : memref<4xi32>
      aie.use_lock(%scaleOF2_prod_lock_0, Release, 1)
      %60 = memref.load %_anonymous10[%c3] : memref<4xi32>
      %61 = arith.addi %60, %c1_i32 : i32
      %62 = arith.cmpi sge, %61, %c2_i32 : i32
      %63 = arith.subi %61, %c2_i32 : i32
      %64 = arith.select %62, %63, %61 : i32
      memref.store %64, %_anonymous10[%c3] : memref<4xi32>
      %65 = memref.load %idx_buffer_pv_2[%c0] : memref<2xi32>
      memref.store %65, %idx_buffer_pv_2[%c0] : memref<2xi32>
      %66 = arith.addi %36, %c1 : index
      cf.br ^bb21(%66 : index)
    ^bb35:  // pred: ^bb21
      aie.use_lock(%memP2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %67 = memref.load %_anonymous10[%c1] : memref<4xi32>
      %68 = arith.index_cast %67 : i32 to index
      %69 = arith.index_cast %68 : index to i32
      cf.switch %69 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%memP2_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%memP2_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%memP2_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%70: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      aie.use_lock(%memV2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %71 = memref.load %_anonymous10[%c2] : memref<4xi32>
      %72 = arith.index_cast %71 : i32 to index
      %73 = arith.index_cast %72 : index to i32
      cf.switch %73 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%memV2_cons_buff_0 : memref<64x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%memV2_cons_buff_1 : memref<64x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%memV2_cons_buff_0 : memref<64x64xbf16>)
    ^bb43(%74: memref<64x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      aie.use_lock(%scaleOF2_cons_lock_0, AcquireGreaterEqual, 1)
      %75 = memref.load %_anonymous10[%c3] : memref<4xi32>
      %76 = arith.index_cast %75 : i32 to index
      %77 = arith.index_cast %76 : index to i32
      cf.switch %77 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%scaleOF2_buff_0 : memref<128xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%scaleOF2_buff_1 : memref<128xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%scaleOF2_buff_0 : memref<128xbf16>)
    ^bb47(%78: memref<128xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_PV(%70, %74, %7, %78, %c32_i32, %c1_i32, %idx_buffer_pv_2) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<128xbf16>, i32, i32, memref<2xi32>) -> ()
      func.call @rescale_O(%7, %78, %c32_i32, %idx_buffer_pv_2) : (memref<32x64xbf16>, memref<128xbf16>, i32, memref<2xi32>) -> ()
      aie.use_lock(%memP2_cons_prod_lock_0, Release, 1)
      %79 = memref.load %_anonymous10[%c1] : memref<4xi32>
      %80 = arith.addi %79, %c1_i32 : i32
      %81 = arith.cmpi sge, %80, %c2_i32 : i32
      %82 = arith.subi %80, %c2_i32 : i32
      %83 = arith.select %81, %82, %80 : i32
      memref.store %83, %_anonymous10[%c1] : memref<4xi32>
      aie.use_lock(%memV2_cons_prod_lock_0, Release, 1)
      %84 = memref.load %_anonymous10[%c2] : memref<4xi32>
      %85 = arith.addi %84, %c1_i32 : i32
      %86 = arith.cmpi sge, %85, %c2_i32 : i32
      %87 = arith.subi %85, %c2_i32 : i32
      %88 = arith.select %86, %87, %85 : i32
      memref.store %88, %_anonymous10[%c2] : memref<4xi32>
      aie.use_lock(%scaleOF2_prod_lock_0, Release, 1)
      %89 = memref.load %_anonymous10[%c3] : memref<4xi32>
      %90 = arith.addi %89, %c1_i32 : i32
      %91 = arith.cmpi sge, %90, %c2_i32 : i32
      %92 = arith.subi %90, %c2_i32 : i32
      %93 = arith.select %91, %92, %90 : i32
      memref.store %93, %_anonymous10[%c3] : memref<4xi32>
      %94 = memref.load %idx_buffer_pv_2[%c0] : memref<2xi32>
      memref.store %94, %idx_buffer_pv_2[%c0] : memref<2xi32>
      memref.store %c0_i32, %idx_buffer_pv_2[%c0] : memref<2xi32>
      %95 = memref.load %idx_buffer_pv_2[%c1] : memref<2xi32>
      memref.store %95, %idx_buffer_pv_2[%c1] : memref<2xi32>
      aie.use_lock(%outOProj2_cons_lock_0, Release, 1)
      %96 = memref.load %_anonymous10[%c0] : memref<4xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous10[%c0] : memref<4xi32>
      %101 = arith.addi %2, %c1 : index
      cf.br ^bb3(%101 : index)
    ^bb48:  // pred: ^bb3
      %102 = arith.addi %0, %c1 : index
      cf.br ^bb1(%102 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_with = "ep_12h_512s_96e_4ph_8pa_4g_1pf_d9_m0_f6_n1-1_n21_me_8971c3b5f9ae_kernels.a", stack_size = 3328 : i32}
    %_anonymous11 = aie.buffer(%tile_2_5) {address = 45056 : i32, mem_bank = 2 : i32, sym_name = "_anonymous11"} : memref<4xi32> 
    %core_2_5 = aie.core(%tile_2_5) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c8 = arith.constant 8 : index
      %c3072_i32 = arith.constant 3072 : i32
      %c3 = arith.constant 3 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous11[%c0] : memref<4xi32>
      memref.store %c0_i32, %_anonymous11[%c1] : memref<4xi32>
      memref.store %c0_i32, %_anonymous11[%c2] : memref<4xi32>
      memref.store %c0_i32, %_anonymous11[%c3] : memref<4xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb24
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb25
    ^bb2:  // pred: ^bb1
      func.call @zero_bf16_o_proj(%o_proj_zero_scratch_2) : (memref<32x96xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb23
      %3 = arith.cmpi slt, %2, %c3 : index
      cf.cond_br %3, ^bb4, ^bb24
    ^bb4:  // pred: ^bb3
      aie.use_lock(%outOProj2_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous11[%c0] : memref<4xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%outOProj2_buff_0 : memref<32x64xbf16>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%outOProj2_buff_1 : memref<32x64xbf16>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%outOProj2_buff_0 : memref<32x64xbf16>)
    ^bb8(%7: memref<32x64xbf16>):  // 3 preds: ^bb5, ^bb6, ^bb7
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb22
      %9 = arith.cmpi slt, %8, %c8 : index
      cf.cond_br %9, ^bb10, ^bb23
    ^bb10:  // pred: ^bb9
      aie.use_lock(%memOW2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous11[%c1] : memref<4xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%memOW2_cons_buff_0 : memref<64x96xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%memOW2_cons_buff_1 : memref<64x96xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%memOW2_cons_buff_0 : memref<64x96xbf16>)
    ^bb14(%13: memref<64x96xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @matmul_with_acc_bf16_bf16_o_proj(%7, %13, %o_proj_zero_scratch_2, %o_proj_partial_scratch_2) : (memref<32x64xbf16>, memref<64x96xbf16>, memref<32x96xbf16>, memref<32x96xbf16>) -> ()
      aie.use_lock(%outOGroupPart1_cons_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous11[%c2] : memref<4xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i32
      cf.switch %16 : i32, [
        default: ^bb17,
        0: ^bb15,
        1: ^bb16
      ]
    ^bb15:  // pred: ^bb14
      cf.br ^bb18(%outOGroupPart1_buff_0 : memref<32x96xbf16>)
    ^bb16:  // pred: ^bb14
      cf.br ^bb18(%outOGroupPart1_buff_1 : memref<32x96xbf16>)
    ^bb17:  // pred: ^bb14
      cf.br ^bb18(%outOGroupPart1_buff_0 : memref<32x96xbf16>)
    ^bb18(%17: memref<32x96xbf16>):  // 3 preds: ^bb15, ^bb16, ^bb17
      func.call @eltwise_add_bf16_vector_o_proj(%17, %o_proj_partial_scratch_2, %o_proj_partial_scratch_2, %c3072_i32) : (memref<32x96xbf16>, memref<32x96xbf16>, memref<32x96xbf16>, i32) -> ()
      aie.use_lock(%outOGroupPart1_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous11[%c2] : memref<4xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous11[%c2] : memref<4xi32>
      aie.use_lock(%outOGroupPart2_prod_lock_0, AcquireGreaterEqual, 1)
      %23 = memref.load %_anonymous11[%c3] : memref<4xi32>
      %24 = arith.index_cast %23 : i32 to index
      %25 = arith.index_cast %24 : index to i32
      cf.switch %25 : i32, [
        default: ^bb21,
        0: ^bb19,
        1: ^bb20
      ]
    ^bb19:  // pred: ^bb18
      cf.br ^bb22(%outOGroupPart2_buff_0 : memref<32x96xbf16>)
    ^bb20:  // pred: ^bb18
      cf.br ^bb22(%outOGroupPart2_buff_1 : memref<32x96xbf16>)
    ^bb21:  // pred: ^bb18
      cf.br ^bb22(%outOGroupPart2_buff_0 : memref<32x96xbf16>)
    ^bb22(%26: memref<32x96xbf16>):  // 3 preds: ^bb19, ^bb20, ^bb21
      func.call @passThroughLine_o_proj(%o_proj_partial_scratch_2, %26, %c3072_i32) : (memref<32x96xbf16>, memref<32x96xbf16>, i32) -> ()
      aie.use_lock(%outOGroupPart2_cons_lock_0, Release, 1)
      %27 = memref.load %_anonymous11[%c3] : memref<4xi32>
      %28 = arith.addi %27, %c1_i32 : i32
      %29 = arith.cmpi sge, %28, %c2_i32 : i32
      %30 = arith.subi %28, %c2_i32 : i32
      %31 = arith.select %29, %30, %28 : i32
      memref.store %31, %_anonymous11[%c3] : memref<4xi32>
      aie.use_lock(%memOW2_cons_prod_lock_0, Release, 1)
      %32 = memref.load %_anonymous11[%c1] : memref<4xi32>
      %33 = arith.addi %32, %c1_i32 : i32
      %34 = arith.cmpi sge, %33, %c2_i32 : i32
      %35 = arith.subi %33, %c2_i32 : i32
      %36 = arith.select %34, %35, %33 : i32
      memref.store %36, %_anonymous11[%c1] : memref<4xi32>
      %37 = arith.addi %8, %c1 : index
      cf.br ^bb9(%37 : index)
    ^bb23:  // pred: ^bb9
      aie.use_lock(%outOProj2_prod_lock_0, Release, 1)
      %38 = memref.load %_anonymous11[%c0] : memref<4xi32>
      %39 = arith.addi %38, %c1_i32 : i32
      %40 = arith.cmpi sge, %39, %c2_i32 : i32
      %41 = arith.subi %39, %c2_i32 : i32
      %42 = arith.select %40, %41, %39 : i32
      memref.store %42, %_anonymous11[%c0] : memref<4xi32>
      %43 = arith.addi %2, %c1 : index
      cf.br ^bb3(%43 : index)
    ^bb24:  // pred: ^bb3
      %44 = arith.addi %0, %c1 : index
      cf.br ^bb1(%44 : index)
    ^bb25:  // pred: ^bb1
      aie.end
    } {link_with = "ep_12h_512s_96e_4ph_8pa_4g_1pf_d9_m0_f6_n1-1_n21_me_8971c3b5f9ae_kernels.a", stack_size = 3328 : i32}
    %_anonymous12 = aie.buffer(%tile_3_2) {address = 36864 : i32, mem_bank = 2 : i32, sym_name = "_anonymous12"} : memref<3xi32> 
    %core_3_2 = aie.core(%tile_3_2) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c3 = arith.constant 3 : index
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous12[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous12[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous12[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb20
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb21
    ^bb2:  // pred: ^bb1
      memref.store %c0_i32, %idx_buffer_qk_3[%c0] : memref<2xi32>
      memref.store %c0_i32, %idx_buffer_qk_3[%c1] : memref<2xi32>
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb19
      %3 = arith.cmpi slt, %2, %c3 : index
      cf.cond_br %3, ^bb4, ^bb20
    ^bb4:  // pred: ^bb3
      aie.use_lock(%memQ3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous12[%c0] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%memQ3_cons_buff_0 : memref<32x64xbf16>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%memQ3_cons_buff_1 : memref<32x64xbf16>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%memQ3_cons_buff_0 : memref<32x64xbf16>)
    ^bb8(%7: memref<32x64xbf16>):  // 3 preds: ^bb5, ^bb6, ^bb7
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb18
      %9 = arith.cmpi slt, %8, %c8 : index
      cf.cond_br %9, ^bb10, ^bb19
    ^bb10:  // pred: ^bb9
      aie.use_lock(%memK3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous12[%c1] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%memK3_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%memK3_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%memK3_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%13: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      aie.use_lock(%memA3_prod_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous12[%c2] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i32
      cf.switch %16 : i32, [
        default: ^bb17,
        0: ^bb15,
        1: ^bb16
      ]
    ^bb15:  // pred: ^bb14
      cf.br ^bb18(%memA3_buff_0 : memref<32x64xbf16>)
    ^bb16:  // pred: ^bb14
      cf.br ^bb18(%memA3_buff_1 : memref<32x64xbf16>)
    ^bb17:  // pred: ^bb14
      cf.br ^bb18(%memA3_buff_0 : memref<32x64xbf16>)
    ^bb18(%17: memref<32x64xbf16>):  // 3 preds: ^bb15, ^bb16, ^bb17
      func.call @zero_bf16(%17) : (memref<32x64xbf16>) -> ()
      func.call @matmul_bf16_bf16_wrapper(%7, %13, %17, %idx_buffer_qk_3) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<2xi32>) -> ()
      aie.use_lock(%memK3_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous12[%c1] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous12[%c1] : memref<3xi32>
      aie.use_lock(%memA3_cons_lock_0, Release, 1)
      %23 = memref.load %_anonymous12[%c2] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous12[%c2] : memref<3xi32>
      %28 = memref.load %idx_buffer_qk_3[%c0] : memref<2xi32>
      memref.store %28, %idx_buffer_qk_3[%c0] : memref<2xi32>
      %29 = arith.addi %8, %c1 : index
      cf.br ^bb9(%29 : index)
    ^bb19:  // pred: ^bb9
      memref.store %c0_i32, %idx_buffer_qk_3[%c0] : memref<2xi32>
      %30 = memref.load %idx_buffer_qk_3[%c1] : memref<2xi32>
      memref.store %30, %idx_buffer_qk_3[%c1] : memref<2xi32>
      aie.use_lock(%memQ3_cons_prod_lock_0, Release, 1)
      %31 = memref.load %_anonymous12[%c0] : memref<3xi32>
      %32 = arith.addi %31, %c1_i32 : i32
      %33 = arith.cmpi sge, %32, %c2_i32 : i32
      %34 = arith.subi %32, %c2_i32 : i32
      %35 = arith.select %33, %34, %32 : i32
      memref.store %35, %_anonymous12[%c0] : memref<3xi32>
      %36 = arith.addi %2, %c1 : index
      cf.br ^bb3(%36 : index)
    ^bb20:  // pred: ^bb3
      %37 = arith.addi %0, %c1 : index
      cf.br ^bb1(%37 : index)
    ^bb21:  // pred: ^bb1
      aie.end
    } {link_with = "ep_12h_512s_96e_4ph_8pa_4g_1pf_d9_m0_f6_n1-1_n21_me_8971c3b5f9ae_kernels.a", stack_size = 3328 : i32}
    %_anonymous13 = aie.buffer(%tile_3_3) {address = 53248 : i32, mem_bank = 3 : i32, sym_name = "_anonymous13"} : memref<3xi32> 
    %core_3_3 = aie.core(%tile_3_3) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c3 = arith.constant 3 : index
      %c32_i32 = arith.constant 32 : i32
      %c8 = arith.constant 8 : index
      %cst = arith.constant 1.806640e-01 : bf16
      %c64_i32 = arith.constant 64 : i32
      %c512_i32 = arith.constant 512 : i32
      %c128_i32 = arith.constant 128 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous13[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous13[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous13[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb20
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb21
    ^bb2:  // pred: ^bb1
      memref.store %c0_i32, %idx_buffer_softmax_3[%c0] : memref<2xi32>
      memref.store %c0_i32, %idx_buffer_softmax_3[%c1] : memref<2xi32>
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb19
      %3 = arith.cmpi slt, %2, %c3 : index
      cf.cond_br %3, ^bb4, ^bb20
    ^bb4:  // pred: ^bb3
      func.call @init_scale_buffer(%scale_buffer_softmax_3, %c32_i32) : (memref<128xbf16>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb18
      %5 = arith.cmpi slt, %4, %c8 : index
      cf.cond_br %5, ^bb6, ^bb19
    ^bb6:  // pred: ^bb5
      aie.use_lock(%memP3_prod_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous13[%c0] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%memP3_buff_0 : memref<32x64xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%memP3_buff_1 : memref<32x64xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%memP3_buff_0 : memref<32x64xbf16>)
    ^bb10(%9: memref<32x64xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%memA3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous13[%c1] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%memA3_cons_buff_0 : memref<32x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%memA3_cons_buff_1 : memref<32x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%memA3_cons_buff_0 : memref<32x64xbf16>)
    ^bb14(%13: memref<32x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      aie.use_lock(%scaleOF3_prod_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous13[%c2] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i32
      cf.switch %16 : i32, [
        default: ^bb17,
        0: ^bb15,
        1: ^bb16
      ]
    ^bb15:  // pred: ^bb14
      cf.br ^bb18(%scaleOF3_buff_0 : memref<128xbf16>)
    ^bb16:  // pred: ^bb14
      cf.br ^bb18(%scaleOF3_buff_1 : memref<128xbf16>)
    ^bb17:  // pred: ^bb14
      cf.br ^bb18(%scaleOF3_buff_0 : memref<128xbf16>)
    ^bb18(%17: memref<128xbf16>):  // 3 preds: ^bb15, ^bb16, ^bb17
      func.call @partial_softmax(%13, %9, %scale_buffer_softmax_3, %idx_buffer_softmax_3, %cst, %c32_i32, %c64_i32, %c512_i32, %c512_i32) : (memref<32x64xbf16>, memref<32x64xbf16>, memref<128xbf16>, memref<2xi32>, bf16, i32, i32, i32, i32) -> ()
      func.call @passThroughLine(%scale_buffer_softmax_3, %17, %c128_i32) : (memref<128xbf16>, memref<128xbf16>, i32) -> ()
      aie.use_lock(%memA3_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous13[%c1] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous13[%c1] : memref<3xi32>
      aie.use_lock(%memP3_cons_lock_0, Release, 1)
      %23 = memref.load %_anonymous13[%c0] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous13[%c0] : memref<3xi32>
      aie.use_lock(%scaleOF3_cons_lock_0, Release, 1)
      %28 = memref.load %_anonymous13[%c2] : memref<3xi32>
      %29 = arith.addi %28, %c1_i32 : i32
      %30 = arith.cmpi sge, %29, %c2_i32 : i32
      %31 = arith.subi %29, %c2_i32 : i32
      %32 = arith.select %30, %31, %29 : i32
      memref.store %32, %_anonymous13[%c2] : memref<3xi32>
      %33 = memref.load %idx_buffer_softmax_3[%c0] : memref<2xi32>
      memref.store %33, %idx_buffer_softmax_3[%c0] : memref<2xi32>
      %34 = arith.addi %4, %c1 : index
      cf.br ^bb5(%34 : index)
    ^bb19:  // pred: ^bb5
      memref.store %c0_i32, %idx_buffer_softmax_3[%c0] : memref<2xi32>
      %35 = memref.load %idx_buffer_softmax_3[%c1] : memref<2xi32>
      memref.store %35, %idx_buffer_softmax_3[%c1] : memref<2xi32>
      %36 = arith.addi %2, %c1 : index
      cf.br ^bb3(%36 : index)
    ^bb20:  // pred: ^bb3
      %37 = arith.addi %0, %c1 : index
      cf.br ^bb1(%37 : index)
    ^bb21:  // pred: ^bb1
      aie.end
    } {link_with = "ep_12h_512s_96e_4ph_8pa_4g_1pf_d9_m0_f6_n1-1_n21_me_8971c3b5f9ae_kernels.a", stack_size = 3328 : i32}
    %_anonymous14 = aie.buffer(%tile_3_4) {address = 36864 : i32, mem_bank = 2 : i32, sym_name = "_anonymous14"} : memref<4xi32> 
    %core_3_4 = aie.core(%tile_3_4) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c32_i32 = arith.constant 32 : i32
      %c6 = arith.constant 6 : index
      %c1_i32 = arith.constant 1 : i32
      %c3 = arith.constant 3 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous14[%c0] : memref<4xi32>
      memref.store %c0_i32, %_anonymous14[%c1] : memref<4xi32>
      memref.store %c0_i32, %_anonymous14[%c2] : memref<4xi32>
      memref.store %c0_i32, %_anonymous14[%c3] : memref<4xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      memref.store %c0_i32, %idx_buffer_pv_3[%c0] : memref<2xi32>
      memref.store %c0_i32, %idx_buffer_pv_3[%c1] : memref<2xi32>
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb47
      %3 = arith.cmpi slt, %2, %c3 : index
      cf.cond_br %3, ^bb4, ^bb48
    ^bb4:  // pred: ^bb3
      aie.use_lock(%outOProj3_prod_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous14[%c0] : memref<4xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%outOProj3_buff_0 : memref<32x64xbf16>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%outOProj3_buff_1 : memref<32x64xbf16>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%outOProj3_buff_0 : memref<32x64xbf16>)
    ^bb8(%7: memref<32x64xbf16>):  // 3 preds: ^bb5, ^bb6, ^bb7
      func.call @zero_bf16(%7) : (memref<32x64xbf16>) -> ()
      aie.use_lock(%memP3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous14[%c1] : memref<4xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%memP3_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%memP3_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%memP3_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      aie.use_lock(%memV3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %12 = memref.load %_anonymous14[%c2] : memref<4xi32>
      %13 = arith.index_cast %12 : i32 to index
      %14 = arith.index_cast %13 : index to i32
      cf.switch %14 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%memV3_cons_buff_0 : memref<64x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%memV3_cons_buff_1 : memref<64x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%memV3_cons_buff_0 : memref<64x64xbf16>)
    ^bb16(%15: memref<64x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      aie.use_lock(%scaleOF3_cons_lock_0, AcquireGreaterEqual, 1)
      %16 = memref.load %_anonymous14[%c3] : memref<4xi32>
      %17 = arith.index_cast %16 : i32 to index
      %18 = arith.index_cast %17 : index to i32
      cf.switch %18 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%scaleOF3_buff_0 : memref<128xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%scaleOF3_buff_1 : memref<128xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%scaleOF3_buff_0 : memref<128xbf16>)
    ^bb20(%19: memref<128xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_PV(%11, %15, %7, %19, %c32_i32, %c0_i32, %idx_buffer_pv_3) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<128xbf16>, i32, i32, memref<2xi32>) -> ()
      aie.use_lock(%memP3_cons_prod_lock_0, Release, 1)
      %20 = memref.load %_anonymous14[%c1] : memref<4xi32>
      %21 = arith.addi %20, %c1_i32 : i32
      %22 = arith.cmpi sge, %21, %c2_i32 : i32
      %23 = arith.subi %21, %c2_i32 : i32
      %24 = arith.select %22, %23, %21 : i32
      memref.store %24, %_anonymous14[%c1] : memref<4xi32>
      aie.use_lock(%memV3_cons_prod_lock_0, Release, 1)
      %25 = memref.load %_anonymous14[%c2] : memref<4xi32>
      %26 = arith.addi %25, %c1_i32 : i32
      %27 = arith.cmpi sge, %26, %c2_i32 : i32
      %28 = arith.subi %26, %c2_i32 : i32
      %29 = arith.select %27, %28, %26 : i32
      memref.store %29, %_anonymous14[%c2] : memref<4xi32>
      aie.use_lock(%scaleOF3_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous14[%c3] : memref<4xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous14[%c3] : memref<4xi32>
      %35 = memref.load %idx_buffer_pv_3[%c0] : memref<2xi32>
      memref.store %35, %idx_buffer_pv_3[%c0] : memref<2xi32>
      cf.br ^bb21(%c0 : index)
    ^bb21(%36: index):  // 2 preds: ^bb20, ^bb34
      %37 = arith.cmpi slt, %36, %c6 : index
      cf.cond_br %37, ^bb22, ^bb35
    ^bb22:  // pred: ^bb21
      aie.use_lock(%memP3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %38 = memref.load %_anonymous14[%c1] : memref<4xi32>
      %39 = arith.index_cast %38 : i32 to index
      %40 = arith.index_cast %39 : index to i32
      cf.switch %40 : i32, [
        default: ^bb25,
        0: ^bb23,
        1: ^bb24
      ]
    ^bb23:  // pred: ^bb22
      cf.br ^bb26(%memP3_cons_buff_0 : memref<32x64xbf16>)
    ^bb24:  // pred: ^bb22
      cf.br ^bb26(%memP3_cons_buff_1 : memref<32x64xbf16>)
    ^bb25:  // pred: ^bb22
      cf.br ^bb26(%memP3_cons_buff_0 : memref<32x64xbf16>)
    ^bb26(%41: memref<32x64xbf16>):  // 3 preds: ^bb23, ^bb24, ^bb25
      aie.use_lock(%memV3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %42 = memref.load %_anonymous14[%c2] : memref<4xi32>
      %43 = arith.index_cast %42 : i32 to index
      %44 = arith.index_cast %43 : index to i32
      cf.switch %44 : i32, [
        default: ^bb29,
        0: ^bb27,
        1: ^bb28
      ]
    ^bb27:  // pred: ^bb26
      cf.br ^bb30(%memV3_cons_buff_0 : memref<64x64xbf16>)
    ^bb28:  // pred: ^bb26
      cf.br ^bb30(%memV3_cons_buff_1 : memref<64x64xbf16>)
    ^bb29:  // pred: ^bb26
      cf.br ^bb30(%memV3_cons_buff_0 : memref<64x64xbf16>)
    ^bb30(%45: memref<64x64xbf16>):  // 3 preds: ^bb27, ^bb28, ^bb29
      aie.use_lock(%scaleOF3_cons_lock_0, AcquireGreaterEqual, 1)
      %46 = memref.load %_anonymous14[%c3] : memref<4xi32>
      %47 = arith.index_cast %46 : i32 to index
      %48 = arith.index_cast %47 : index to i32
      cf.switch %48 : i32, [
        default: ^bb33,
        0: ^bb31,
        1: ^bb32
      ]
    ^bb31:  // pred: ^bb30
      cf.br ^bb34(%scaleOF3_buff_0 : memref<128xbf16>)
    ^bb32:  // pred: ^bb30
      cf.br ^bb34(%scaleOF3_buff_1 : memref<128xbf16>)
    ^bb33:  // pred: ^bb30
      cf.br ^bb34(%scaleOF3_buff_0 : memref<128xbf16>)
    ^bb34(%49: memref<128xbf16>):  // 3 preds: ^bb31, ^bb32, ^bb33
      func.call @matmul_PV(%41, %45, %7, %49, %c32_i32, %c1_i32, %idx_buffer_pv_3) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<128xbf16>, i32, i32, memref<2xi32>) -> ()
      aie.use_lock(%memP3_cons_prod_lock_0, Release, 1)
      %50 = memref.load %_anonymous14[%c1] : memref<4xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c2_i32 : i32
      %53 = arith.subi %51, %c2_i32 : i32
      %54 = arith.select %52, %53, %51 : i32
      memref.store %54, %_anonymous14[%c1] : memref<4xi32>
      aie.use_lock(%memV3_cons_prod_lock_0, Release, 1)
      %55 = memref.load %_anonymous14[%c2] : memref<4xi32>
      %56 = arith.addi %55, %c1_i32 : i32
      %57 = arith.cmpi sge, %56, %c2_i32 : i32
      %58 = arith.subi %56, %c2_i32 : i32
      %59 = arith.select %57, %58, %56 : i32
      memref.store %59, %_anonymous14[%c2] : memref<4xi32>
      aie.use_lock(%scaleOF3_prod_lock_0, Release, 1)
      %60 = memref.load %_anonymous14[%c3] : memref<4xi32>
      %61 = arith.addi %60, %c1_i32 : i32
      %62 = arith.cmpi sge, %61, %c2_i32 : i32
      %63 = arith.subi %61, %c2_i32 : i32
      %64 = arith.select %62, %63, %61 : i32
      memref.store %64, %_anonymous14[%c3] : memref<4xi32>
      %65 = memref.load %idx_buffer_pv_3[%c0] : memref<2xi32>
      memref.store %65, %idx_buffer_pv_3[%c0] : memref<2xi32>
      %66 = arith.addi %36, %c1 : index
      cf.br ^bb21(%66 : index)
    ^bb35:  // pred: ^bb21
      aie.use_lock(%memP3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %67 = memref.load %_anonymous14[%c1] : memref<4xi32>
      %68 = arith.index_cast %67 : i32 to index
      %69 = arith.index_cast %68 : index to i32
      cf.switch %69 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%memP3_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%memP3_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%memP3_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%70: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      aie.use_lock(%memV3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %71 = memref.load %_anonymous14[%c2] : memref<4xi32>
      %72 = arith.index_cast %71 : i32 to index
      %73 = arith.index_cast %72 : index to i32
      cf.switch %73 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%memV3_cons_buff_0 : memref<64x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%memV3_cons_buff_1 : memref<64x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%memV3_cons_buff_0 : memref<64x64xbf16>)
    ^bb43(%74: memref<64x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      aie.use_lock(%scaleOF3_cons_lock_0, AcquireGreaterEqual, 1)
      %75 = memref.load %_anonymous14[%c3] : memref<4xi32>
      %76 = arith.index_cast %75 : i32 to index
      %77 = arith.index_cast %76 : index to i32
      cf.switch %77 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%scaleOF3_buff_0 : memref<128xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%scaleOF3_buff_1 : memref<128xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%scaleOF3_buff_0 : memref<128xbf16>)
    ^bb47(%78: memref<128xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_PV(%70, %74, %7, %78, %c32_i32, %c1_i32, %idx_buffer_pv_3) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<128xbf16>, i32, i32, memref<2xi32>) -> ()
      func.call @rescale_O(%7, %78, %c32_i32, %idx_buffer_pv_3) : (memref<32x64xbf16>, memref<128xbf16>, i32, memref<2xi32>) -> ()
      aie.use_lock(%memP3_cons_prod_lock_0, Release, 1)
      %79 = memref.load %_anonymous14[%c1] : memref<4xi32>
      %80 = arith.addi %79, %c1_i32 : i32
      %81 = arith.cmpi sge, %80, %c2_i32 : i32
      %82 = arith.subi %80, %c2_i32 : i32
      %83 = arith.select %81, %82, %80 : i32
      memref.store %83, %_anonymous14[%c1] : memref<4xi32>
      aie.use_lock(%memV3_cons_prod_lock_0, Release, 1)
      %84 = memref.load %_anonymous14[%c2] : memref<4xi32>
      %85 = arith.addi %84, %c1_i32 : i32
      %86 = arith.cmpi sge, %85, %c2_i32 : i32
      %87 = arith.subi %85, %c2_i32 : i32
      %88 = arith.select %86, %87, %85 : i32
      memref.store %88, %_anonymous14[%c2] : memref<4xi32>
      aie.use_lock(%scaleOF3_prod_lock_0, Release, 1)
      %89 = memref.load %_anonymous14[%c3] : memref<4xi32>
      %90 = arith.addi %89, %c1_i32 : i32
      %91 = arith.cmpi sge, %90, %c2_i32 : i32
      %92 = arith.subi %90, %c2_i32 : i32
      %93 = arith.select %91, %92, %90 : i32
      memref.store %93, %_anonymous14[%c3] : memref<4xi32>
      %94 = memref.load %idx_buffer_pv_3[%c0] : memref<2xi32>
      memref.store %94, %idx_buffer_pv_3[%c0] : memref<2xi32>
      memref.store %c0_i32, %idx_buffer_pv_3[%c0] : memref<2xi32>
      %95 = memref.load %idx_buffer_pv_3[%c1] : memref<2xi32>
      memref.store %95, %idx_buffer_pv_3[%c1] : memref<2xi32>
      aie.use_lock(%outOProj3_cons_lock_0, Release, 1)
      %96 = memref.load %_anonymous14[%c0] : memref<4xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous14[%c0] : memref<4xi32>
      %101 = arith.addi %2, %c1 : index
      cf.br ^bb3(%101 : index)
    ^bb48:  // pred: ^bb3
      %102 = arith.addi %0, %c1 : index
      cf.br ^bb1(%102 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_with = "ep_12h_512s_96e_4ph_8pa_4g_1pf_d9_m0_f6_n1-1_n21_me_8971c3b5f9ae_kernels.a", stack_size = 3328 : i32}
    %_anonymous15 = aie.buffer(%tile_3_5) {address = 65024 : i32, sym_name = "_anonymous15"} : memref<6xi32> 
    %core_3_5 = aie.core(%tile_3_5) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c8 = arith.constant 8 : index
      %c3072_i32 = arith.constant 3072 : i32
      %c5 = arith.constant 5 : index
      %c4 = arith.constant 4 : index
      %c3 = arith.constant 3 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous15[%c0] : memref<6xi32>
      memref.store %c0_i32, %_anonymous15[%c1] : memref<6xi32>
      memref.store %c0_i32, %_anonymous15[%c2] : memref<6xi32>
      memref.store %c0_i32, %_anonymous15[%c3] : memref<6xi32>
      memref.store %c0_i32, %_anonymous15[%c4] : memref<6xi32>
      memref.store %c0_i32, %_anonymous15[%c5] : memref<6xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb30
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb31
    ^bb2:  // pred: ^bb1
      func.call @zero_bf16_o_proj(%o_proj_zero_scratch_3) : (memref<32x96xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c8 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%outOProjAccumOut3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_bf16_o_proj(%outOProjAccumOut3_buff_0) : (memref<32x96xbf16>) -> ()
      aie.use_lock(%outOProjAccumOut3_cons_lock_0, Release, 1)
      %4 = memref.load %_anonymous15[%c0] : memref<6xi32>
      %5 = arith.addi %4, %c1_i32 : i32
      %6 = arith.cmpi sge, %5, %c1_i32 : i32
      %7 = arith.select %6, %4, %5 : i32
      memref.store %7, %_anonymous15[%c0] : memref<6xi32>
      %8 = arith.addi %2, %c1 : index
      cf.br ^bb3(%8 : index)
    ^bb5:  // pred: ^bb3
      cf.br ^bb6(%c0 : index)
    ^bb6(%9: index):  // 2 preds: ^bb5, ^bb22
      %10 = arith.cmpi slt, %9, %c3 : index
      cf.cond_br %10, ^bb7, ^bb23
    ^bb7:  // pred: ^bb6
      aie.use_lock(%outOProj3_cons_lock_0, AcquireGreaterEqual, 1)
      %11 = memref.load %_anonymous15[%c1] : memref<6xi32>
      %12 = arith.index_cast %11 : i32 to index
      %13 = arith.index_cast %12 : index to i32
      cf.switch %13 : i32, [
        default: ^bb10,
        0: ^bb8,
        1: ^bb9
      ]
    ^bb8:  // pred: ^bb7
      cf.br ^bb11(%outOProj3_buff_0 : memref<32x64xbf16>)
    ^bb9:  // pred: ^bb7
      cf.br ^bb11(%outOProj3_buff_1 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb7
      cf.br ^bb11(%outOProj3_buff_0 : memref<32x64xbf16>)
    ^bb11(%14: memref<32x64xbf16>):  // 3 preds: ^bb8, ^bb9, ^bb10
      cf.br ^bb12(%c0 : index)
    ^bb12(%15: index):  // 2 preds: ^bb11, ^bb21
      %16 = arith.cmpi slt, %15, %c8 : index
      cf.cond_br %16, ^bb13, ^bb22
    ^bb13:  // pred: ^bb12
      aie.use_lock(%memOW3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %17 = memref.load %_anonymous15[%c2] : memref<6xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i32
      cf.switch %19 : i32, [
        default: ^bb16,
        0: ^bb14,
        1: ^bb15
      ]
    ^bb14:  // pred: ^bb13
      cf.br ^bb17(%memOW3_cons_buff_0 : memref<64x96xbf16>)
    ^bb15:  // pred: ^bb13
      cf.br ^bb17(%memOW3_cons_buff_1 : memref<64x96xbf16>)
    ^bb16:  // pred: ^bb13
      cf.br ^bb17(%memOW3_cons_buff_0 : memref<64x96xbf16>)
    ^bb17(%20: memref<64x96xbf16>):  // 3 preds: ^bb14, ^bb15, ^bb16
      func.call @matmul_with_acc_bf16_bf16_o_proj(%14, %20, %o_proj_zero_scratch_3, %o_proj_partial_scratch_3) : (memref<32x64xbf16>, memref<64x96xbf16>, memref<32x96xbf16>, memref<32x96xbf16>) -> ()
      aie.use_lock(%outOGroupPart2_cons_lock_0, AcquireGreaterEqual, 1)
      %21 = memref.load %_anonymous15[%c3] : memref<6xi32>
      %22 = arith.index_cast %21 : i32 to index
      %23 = arith.index_cast %22 : index to i32
      cf.switch %23 : i32, [
        default: ^bb20,
        0: ^bb18,
        1: ^bb19
      ]
    ^bb18:  // pred: ^bb17
      cf.br ^bb21(%outOGroupPart2_buff_0 : memref<32x96xbf16>)
    ^bb19:  // pred: ^bb17
      cf.br ^bb21(%outOGroupPart2_buff_1 : memref<32x96xbf16>)
    ^bb20:  // pred: ^bb17
      cf.br ^bb21(%outOGroupPart2_buff_0 : memref<32x96xbf16>)
    ^bb21(%24: memref<32x96xbf16>):  // 3 preds: ^bb18, ^bb19, ^bb20
      func.call @eltwise_add_bf16_vector_o_proj(%24, %o_proj_partial_scratch_3, %o_proj_partial_scratch_3, %c3072_i32) : (memref<32x96xbf16>, memref<32x96xbf16>, memref<32x96xbf16>, i32) -> ()
      aie.use_lock(%outOGroupPart2_prod_lock_0, Release, 1)
      %25 = memref.load %_anonymous15[%c3] : memref<6xi32>
      %26 = arith.addi %25, %c1_i32 : i32
      %27 = arith.cmpi sge, %26, %c2_i32 : i32
      %28 = arith.subi %26, %c2_i32 : i32
      %29 = arith.select %27, %28, %26 : i32
      memref.store %29, %_anonymous15[%c3] : memref<6xi32>
      aie.use_lock(%outOProjAccumIn3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%outOProjAccumOut3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @eltwise_add_bf16_vector_o_proj(%outOProjAccumIn3_cons_buff_0, %o_proj_partial_scratch_3, %outOProjAccumOut3_buff_0, %c3072_i32) : (memref<32x96xbf16>, memref<32x96xbf16>, memref<32x96xbf16>, i32) -> ()
      aie.use_lock(%outOProjAccumOut3_cons_lock_0, Release, 1)
      %30 = memref.load %_anonymous15[%c0] : memref<6xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c1_i32 : i32
      %33 = arith.select %32, %30, %31 : i32
      memref.store %33, %_anonymous15[%c0] : memref<6xi32>
      aie.use_lock(%outOProjAccumIn3_cons_prod_lock_0, Release, 1)
      %34 = memref.load %_anonymous15[%c4] : memref<6xi32>
      %35 = arith.addi %34, %c1_i32 : i32
      %36 = arith.cmpi sge, %35, %c1_i32 : i32
      %37 = arith.select %36, %34, %35 : i32
      memref.store %37, %_anonymous15[%c4] : memref<6xi32>
      aie.use_lock(%memOW3_cons_prod_lock_0, Release, 1)
      %38 = memref.load %_anonymous15[%c2] : memref<6xi32>
      %39 = arith.addi %38, %c1_i32 : i32
      %40 = arith.cmpi sge, %39, %c2_i32 : i32
      %41 = arith.subi %39, %c2_i32 : i32
      %42 = arith.select %40, %41, %39 : i32
      memref.store %42, %_anonymous15[%c2] : memref<6xi32>
      %43 = arith.addi %15, %c1 : index
      cf.br ^bb12(%43 : index)
    ^bb22:  // pred: ^bb12
      aie.use_lock(%outOProj3_prod_lock_0, Release, 1)
      %44 = memref.load %_anonymous15[%c1] : memref<6xi32>
      %45 = arith.addi %44, %c1_i32 : i32
      %46 = arith.cmpi sge, %45, %c2_i32 : i32
      %47 = arith.subi %45, %c2_i32 : i32
      %48 = arith.select %46, %47, %45 : i32
      memref.store %48, %_anonymous15[%c1] : memref<6xi32>
      %49 = arith.addi %9, %c1 : index
      cf.br ^bb6(%49 : index)
    ^bb23:  // pred: ^bb6
      cf.br ^bb24(%c0 : index)
    ^bb24(%50: index):  // 2 preds: ^bb23, ^bb29
      %51 = arith.cmpi slt, %50, %c8 : index
      cf.cond_br %51, ^bb25, ^bb30
    ^bb25:  // pred: ^bb24
      aie.use_lock(%outOProjAccumIn3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%outOProjInput_prod_lock_0, AcquireGreaterEqual, 1)
      %52 = memref.load %_anonymous15[%c5] : memref<6xi32>
      %53 = arith.index_cast %52 : i32 to index
      %54 = arith.index_cast %53 : index to i32
      cf.switch %54 : i32, [
        default: ^bb28,
        0: ^bb26,
        1: ^bb27
      ]
    ^bb26:  // pred: ^bb25
      cf.br ^bb29(%outOProjInput_buff_0 : memref<32x96xbf16>)
    ^bb27:  // pred: ^bb25
      cf.br ^bb29(%outOProjInput_buff_1 : memref<32x96xbf16>)
    ^bb28:  // pred: ^bb25
      cf.br ^bb29(%outOProjInput_buff_0 : memref<32x96xbf16>)
    ^bb29(%55: memref<32x96xbf16>):  // 3 preds: ^bb26, ^bb27, ^bb28
      func.call @passThroughLine_o_proj(%outOProjAccumIn3_cons_buff_0, %55, %c3072_i32) : (memref<32x96xbf16>, memref<32x96xbf16>, i32) -> ()
      aie.use_lock(%outOProjInput_cons_lock_0, Release, 1)
      %56 = memref.load %_anonymous15[%c5] : memref<6xi32>
      %57 = arith.addi %56, %c1_i32 : i32
      %58 = arith.cmpi sge, %57, %c2_i32 : i32
      %59 = arith.subi %57, %c2_i32 : i32
      %60 = arith.select %58, %59, %57 : i32
      memref.store %60, %_anonymous15[%c5] : memref<6xi32>
      aie.use_lock(%outOProjAccumIn3_cons_prod_lock_0, Release, 1)
      %61 = memref.load %_anonymous15[%c4] : memref<6xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c1_i32 : i32
      %64 = arith.select %63, %61, %62 : i32
      memref.store %64, %_anonymous15[%c4] : memref<6xi32>
      %65 = arith.addi %50, %c1 : index
      cf.br ^bb24(%65 : index)
    ^bb30:  // pred: ^bb24
      %66 = arith.addi %0, %c1 : index
      cf.br ^bb1(%66 : index)
    ^bb31:  // pred: ^bb1
      aie.end
    } {link_with = "ep_12h_512s_96e_4ph_8pa_4g_1pf_d9_m0_f6_n1-1_n21_me_8971c3b5f9ae_kernels.a", stack_size = 3328 : i32}
    %_anonymous16 = aie.buffer(%tile_6_5) {address = 13312 : i32, mem_bank = 0 : i32, sym_name = "_anonymous16"} : memref<2xi32> 
    %core_6_5 = aie.core(%tile_6_5) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c8 = arith.constant 8 : index
      %c3072_i32 = arith.constant 3072 : i32
      %c30 = arith.constant 30 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous16[%c0] : memref<2xi32>
      memref.store %c0_i32, %_anonymous16[%c1] : memref<2xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb33
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb34
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb8
      %3 = arith.cmpi slt, %2, %c8 : index
      cf.cond_br %3, ^bb4, ^bb9
    ^bb4:  // pred: ^bb3
      aie.use_lock(%outOProjInput_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous16[%c0] : memref<2xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%outOProjInput_cons_buff_0 : memref<32x96xbf16>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%outOProjInput_cons_buff_1 : memref<32x96xbf16>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%outOProjInput_cons_buff_0 : memref<32x96xbf16>)
    ^bb8(%7: memref<32x96xbf16>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%ln1Replay_src_empty, AcquireGreaterEqual, 1)
      func.call @passThroughLine_o_proj(%7, %ln1Replay_src, %c3072_i32) : (memref<32x96xbf16>, memref<32x96xbf16>, i32) -> ()
      aie.use_lock(%ln1Replay_src_full, Release, 1)
      aie.use_lock(%outOProjInput_cons_prod_lock_0, Release, 1)
      %8 = memref.load %_anonymous16[%c0] : memref<2xi32>
      %9 = arith.addi %8, %c1_i32 : i32
      %10 = arith.cmpi sge, %9, %c2_i32 : i32
      %11 = arith.subi %9, %c2_i32 : i32
      %12 = arith.select %10, %11, %9 : i32
      memref.store %12, %_anonymous16[%c0] : memref<2xi32>
      %13 = arith.addi %2, %c1 : index
      cf.br ^bb3(%13 : index)
    ^bb9:  // pred: ^bb3
      cf.br ^bb10(%c0 : index)
    ^bb10(%14: index):  // 2 preds: ^bb9, ^bb15
      %15 = arith.cmpi slt, %14, %c8 : index
      cf.cond_br %15, ^bb11, ^bb16
    ^bb11:  // pred: ^bb10
      aie.use_lock(%ln1Replay_dst_full, AcquireGreaterEqual, 1)
      aie.use_lock(%ln1Replay_src_empty, AcquireGreaterEqual, 1)
      func.call @passThroughLine_o_proj(%ln1Replay_dst, %ln1Replay_src, %c3072_i32) : (memref<32x96xbf16>, memref<32x96xbf16>, i32) -> ()
      aie.use_lock(%ln1Norm_prod_lock_0, AcquireGreaterEqual, 1)
      %16 = memref.load %_anonymous16[%c1] : memref<2xi32>
      %17 = arith.index_cast %16 : i32 to index
      %18 = arith.index_cast %17 : index to i32
      cf.switch %18 : i32, [
        default: ^bb14,
        0: ^bb12,
        1: ^bb13
      ]
    ^bb12:  // pred: ^bb11
      cf.br ^bb15(%ln1Norm_buff_0 : memref<32x96xbf16>)
    ^bb13:  // pred: ^bb11
      cf.br ^bb15(%ln1Norm_buff_1 : memref<32x96xbf16>)
    ^bb14:  // pred: ^bb11
      cf.br ^bb15(%ln1Norm_buff_0 : memref<32x96xbf16>)
    ^bb15(%19: memref<32x96xbf16>):  // 3 preds: ^bb12, ^bb13, ^bb14
      func.call @passThroughLine_o_proj(%ln1Replay_src, %19, %c3072_i32) : (memref<32x96xbf16>, memref<32x96xbf16>, i32) -> ()
      aie.use_lock(%ln1Norm_cons_lock_0, Release, 1)
      %20 = memref.load %_anonymous16[%c1] : memref<2xi32>
      %21 = arith.addi %20, %c1_i32 : i32
      %22 = arith.cmpi sge, %21, %c2_i32 : i32
      %23 = arith.subi %21, %c2_i32 : i32
      %24 = arith.select %22, %23, %21 : i32
      memref.store %24, %_anonymous16[%c1] : memref<2xi32>
      aie.use_lock(%ln1Replay_src_full, Release, 1)
      aie.use_lock(%ln1Replay_dst_empty, Release, 1)
      %25 = arith.addi %14, %c1 : index
      cf.br ^bb10(%25 : index)
    ^bb16:  // pred: ^bb10
      cf.br ^bb17(%c0 : index)
    ^bb17(%26: index):  // 2 preds: ^bb16, ^bb25
      %27 = arith.cmpi slt, %26, %c30 : index
      cf.cond_br %27, ^bb18, ^bb26
    ^bb18:  // pred: ^bb17
      cf.br ^bb19(%c0 : index)
    ^bb19(%28: index):  // 2 preds: ^bb18, ^bb24
      %29 = arith.cmpi slt, %28, %c8 : index
      cf.cond_br %29, ^bb20, ^bb25
    ^bb20:  // pred: ^bb19
      aie.use_lock(%ln1Replay_dst_full, AcquireGreaterEqual, 1)
      aie.use_lock(%ln1Norm_prod_lock_0, AcquireGreaterEqual, 1)
      %30 = memref.load %_anonymous16[%c1] : memref<2xi32>
      %31 = arith.index_cast %30 : i32 to index
      %32 = arith.index_cast %31 : index to i32
      cf.switch %32 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%ln1Norm_buff_0 : memref<32x96xbf16>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%ln1Norm_buff_1 : memref<32x96xbf16>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%ln1Norm_buff_0 : memref<32x96xbf16>)
    ^bb24(%33: memref<32x96xbf16>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @passThroughLine_o_proj(%ln1Replay_dst, %33, %c3072_i32) : (memref<32x96xbf16>, memref<32x96xbf16>, i32) -> ()
      aie.use_lock(%ln1Norm_cons_lock_0, Release, 1)
      %34 = memref.load %_anonymous16[%c1] : memref<2xi32>
      %35 = arith.addi %34, %c1_i32 : i32
      %36 = arith.cmpi sge, %35, %c2_i32 : i32
      %37 = arith.subi %35, %c2_i32 : i32
      %38 = arith.select %36, %37, %35 : i32
      memref.store %38, %_anonymous16[%c1] : memref<2xi32>
      aie.use_lock(%ln1Replay_src_empty, AcquireGreaterEqual, 1)
      func.call @passThroughLine_o_proj(%ln1Replay_dst, %ln1Replay_src, %c3072_i32) : (memref<32x96xbf16>, memref<32x96xbf16>, i32) -> ()
      aie.use_lock(%ln1Replay_src_full, Release, 1)
      aie.use_lock(%ln1Replay_dst_empty, Release, 1)
      %39 = arith.addi %28, %c1 : index
      cf.br ^bb19(%39 : index)
    ^bb25:  // pred: ^bb19
      %40 = arith.addi %26, %c1 : index
      cf.br ^bb17(%40 : index)
    ^bb26:  // pred: ^bb17
      cf.br ^bb27(%c0 : index)
    ^bb27(%41: index):  // 2 preds: ^bb26, ^bb32
      %42 = arith.cmpi slt, %41, %c8 : index
      cf.cond_br %42, ^bb28, ^bb33
    ^bb28:  // pred: ^bb27
      aie.use_lock(%ln1Replay_dst_full, AcquireGreaterEqual, 1)
      aie.use_lock(%ln1Norm_prod_lock_0, AcquireGreaterEqual, 1)
      %43 = memref.load %_anonymous16[%c1] : memref<2xi32>
      %44 = arith.index_cast %43 : i32 to index
      %45 = arith.index_cast %44 : index to i32
      cf.switch %45 : i32, [
        default: ^bb31,
        0: ^bb29,
        1: ^bb30
      ]
    ^bb29:  // pred: ^bb28
      cf.br ^bb32(%ln1Norm_buff_0 : memref<32x96xbf16>)
    ^bb30:  // pred: ^bb28
      cf.br ^bb32(%ln1Norm_buff_1 : memref<32x96xbf16>)
    ^bb31:  // pred: ^bb28
      cf.br ^bb32(%ln1Norm_buff_0 : memref<32x96xbf16>)
    ^bb32(%46: memref<32x96xbf16>):  // 3 preds: ^bb29, ^bb30, ^bb31
      func.call @passThroughLine_o_proj(%ln1Replay_dst, %46, %c3072_i32) : (memref<32x96xbf16>, memref<32x96xbf16>, i32) -> ()
      aie.use_lock(%ln1Norm_cons_lock_0, Release, 1)
      %47 = memref.load %_anonymous16[%c1] : memref<2xi32>
      %48 = arith.addi %47, %c1_i32 : i32
      %49 = arith.cmpi sge, %48, %c2_i32 : i32
      %50 = arith.subi %48, %c2_i32 : i32
      %51 = arith.select %49, %50, %48 : i32
      memref.store %51, %_anonymous16[%c1] : memref<2xi32>
      aie.use_lock(%ln1Replay_dst_empty, Release, 1)
      %52 = arith.addi %41, %c1 : index
      cf.br ^bb27(%52 : index)
    ^bb33:  // pred: ^bb27
      %53 = arith.addi %0, %c1 : index
      cf.br ^bb1(%53 : index)
    ^bb34:  // pred: ^bb1
      aie.end
    } {link_with = "ep_12h_512s_96e_4ph_8pa_4g_1pf_d9_m0_f6_n1-1_n21_me_8971c3b5f9ae_kernels.a"}
    %_anonymous17 = aie.buffer(%tile_4_2) {address = 55296 : i32, mem_bank = 3 : i32, sym_name = "_anonymous17"} : memref<4xi32> 
    %core_4_2 = aie.core(%tile_4_2) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c8 = arith.constant 8 : index
      %c3072_i32 = arith.constant 3072 : i32
      %c31 = arith.constant 31 : index
      %c3 = arith.constant 3 : index
      %c1_i32 = arith.constant 1 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous17[%c0] : memref<4xi32>
      memref.store %c0_i32, %_anonymous17[%c1] : memref<4xi32>
      memref.store %c0_i32, %_anonymous17[%c2] : memref<4xi32>
      memref.store %c0_i32, %_anonymous17[%c3] : memref<4xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb27
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb28
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb12
      %3 = arith.cmpi slt, %2, %c8 : index
      cf.cond_br %3, ^bb4, ^bb13
    ^bb4:  // pred: ^bb3
      %4 = index.casts %2 : index to i32
      aie.use_lock(%ln1Norm_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %5 = memref.load %_anonymous17[%c0] : memref<4xi32>
      %6 = arith.index_cast %5 : i32 to index
      %7 = arith.index_cast %6 : index to i32
      cf.switch %7 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%ln1Norm_cons_buff_0 : memref<32x96xbf16>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%ln1Norm_cons_buff_1 : memref<32x96xbf16>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%ln1Norm_cons_buff_0 : memref<32x96xbf16>)
    ^bb8(%8: memref<32x96xbf16>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%memR_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %9 = memref.load %_anonymous17[%c1] : memref<4xi32>
      %10 = arith.index_cast %9 : i32 to index
      %11 = arith.index_cast %10 : index to i32
      cf.switch %11 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%memR_cons_buff_0 : memref<32x96xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%memR_cons_buff_1 : memref<32x96xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%memR_cons_buff_0 : memref<32x96xbf16>)
    ^bb12(%12: memref<32x96xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      aie.use_lock(%outLNBroadcast_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @ln_mul_add_1outs(%8, %12, %static_ln1_weights, %outLNBroadcast_buff_0, %4) : (memref<32x96xbf16>, memref<32x96xbf16>, memref<768xbf16>, memref<32x96xbf16>, i32) -> ()
      aie.use_lock(%ffnROut_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @passThroughLine_o_proj(%outLNBroadcast_buff_0, %ffnROut_buff_0, %c3072_i32) : (memref<32x96xbf16>, memref<32x96xbf16>, i32) -> ()
      aie.use_lock(%ffnROut_cons_lock_0, Release, 1)
      %13 = memref.load %_anonymous17[%c3] : memref<4xi32>
      %14 = arith.addi %13, %c1_i32 : i32
      %15 = arith.cmpi sge, %14, %c1_i32 : i32
      %16 = arith.select %15, %13, %14 : i32
      memref.store %16, %_anonymous17[%c3] : memref<4xi32>
      aie.use_lock(%outLNBroadcast_cons_lock_0, Release, 1)
      %17 = memref.load %_anonymous17[%c2] : memref<4xi32>
      %18 = arith.addi %17, %c1_i32 : i32
      %19 = arith.cmpi sge, %18, %c1_i32 : i32
      %20 = arith.select %19, %17, %18 : i32
      memref.store %20, %_anonymous17[%c2] : memref<4xi32>
      aie.use_lock(%ln1Norm_cons_prod_lock_0, Release, 1)
      %21 = memref.load %_anonymous17[%c0] : memref<4xi32>
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.subi %22, %c2_i32 : i32
      %25 = arith.select %23, %24, %22 : i32
      memref.store %25, %_anonymous17[%c0] : memref<4xi32>
      aie.use_lock(%memR_cons_prod_lock_0, Release, 1)
      %26 = memref.load %_anonymous17[%c1] : memref<4xi32>
      %27 = arith.addi %26, %c1_i32 : i32
      %28 = arith.cmpi sge, %27, %c2_i32 : i32
      %29 = arith.subi %27, %c2_i32 : i32
      %30 = arith.select %28, %29, %27 : i32
      memref.store %30, %_anonymous17[%c1] : memref<4xi32>
      %31 = arith.addi %2, %c1 : index
      cf.br ^bb3(%31 : index)
    ^bb13:  // pred: ^bb3
      cf.br ^bb14(%c0 : index)
    ^bb14(%32: index):  // 2 preds: ^bb13, ^bb26
      %33 = arith.cmpi slt, %32, %c31 : index
      cf.cond_br %33, ^bb15, ^bb27
    ^bb15:  // pred: ^bb14
      cf.br ^bb16(%c0 : index)
    ^bb16(%34: index):  // 2 preds: ^bb15, ^bb25
      %35 = arith.cmpi slt, %34, %c8 : index
      cf.cond_br %35, ^bb17, ^bb26
    ^bb17:  // pred: ^bb16
      %36 = index.casts %34 : index to i32
      aie.use_lock(%ln1Norm_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %37 = memref.load %_anonymous17[%c0] : memref<4xi32>
      %38 = arith.index_cast %37 : i32 to index
      %39 = arith.index_cast %38 : index to i32
      cf.switch %39 : i32, [
        default: ^bb20,
        0: ^bb18,
        1: ^bb19
      ]
    ^bb18:  // pred: ^bb17
      cf.br ^bb21(%ln1Norm_cons_buff_0 : memref<32x96xbf16>)
    ^bb19:  // pred: ^bb17
      cf.br ^bb21(%ln1Norm_cons_buff_1 : memref<32x96xbf16>)
    ^bb20:  // pred: ^bb17
      cf.br ^bb21(%ln1Norm_cons_buff_0 : memref<32x96xbf16>)
    ^bb21(%40: memref<32x96xbf16>):  // 3 preds: ^bb18, ^bb19, ^bb20
      aie.use_lock(%memR_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %41 = memref.load %_anonymous17[%c1] : memref<4xi32>
      %42 = arith.index_cast %41 : i32 to index
      %43 = arith.index_cast %42 : index to i32
      cf.switch %43 : i32, [
        default: ^bb24,
        0: ^bb22,
        1: ^bb23
      ]
    ^bb22:  // pred: ^bb21
      cf.br ^bb25(%memR_cons_buff_0 : memref<32x96xbf16>)
    ^bb23:  // pred: ^bb21
      cf.br ^bb25(%memR_cons_buff_1 : memref<32x96xbf16>)
    ^bb24:  // pred: ^bb21
      cf.br ^bb25(%memR_cons_buff_0 : memref<32x96xbf16>)
    ^bb25(%44: memref<32x96xbf16>):  // 3 preds: ^bb22, ^bb23, ^bb24
      aie.use_lock(%outLNBroadcast_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @ln_mul_add_1outs(%40, %44, %static_ln1_weights, %outLNBroadcast_buff_0, %36) : (memref<32x96xbf16>, memref<32x96xbf16>, memref<768xbf16>, memref<32x96xbf16>, i32) -> ()
      aie.use_lock(%outLNBroadcast_cons_lock_0, Release, 1)
      %45 = memref.load %_anonymous17[%c2] : memref<4xi32>
      %46 = arith.addi %45, %c1_i32 : i32
      %47 = arith.cmpi sge, %46, %c1_i32 : i32
      %48 = arith.select %47, %45, %46 : i32
      memref.store %48, %_anonymous17[%c2] : memref<4xi32>
      aie.use_lock(%ln1Norm_cons_prod_lock_0, Release, 1)
      %49 = memref.load %_anonymous17[%c0] : memref<4xi32>
      %50 = arith.addi %49, %c1_i32 : i32
      %51 = arith.cmpi sge, %50, %c2_i32 : i32
      %52 = arith.subi %50, %c2_i32 : i32
      %53 = arith.select %51, %52, %50 : i32
      memref.store %53, %_anonymous17[%c0] : memref<4xi32>
      aie.use_lock(%memR_cons_prod_lock_0, Release, 1)
      %54 = memref.load %_anonymous17[%c1] : memref<4xi32>
      %55 = arith.addi %54, %c1_i32 : i32
      %56 = arith.cmpi sge, %55, %c2_i32 : i32
      %57 = arith.subi %55, %c2_i32 : i32
      %58 = arith.select %56, %57, %55 : i32
      memref.store %58, %_anonymous17[%c1] : memref<4xi32>
      %59 = arith.addi %34, %c1 : index
      cf.br ^bb16(%59 : index)
    ^bb26:  // pred: ^bb16
      %60 = arith.addi %32, %c1 : index
      cf.br ^bb14(%60 : index)
    ^bb27:  // pred: ^bb14
      %61 = arith.addi %0, %c1 : index
      cf.br ^bb1(%61 : index)
    ^bb28:  // pred: ^bb1
      aie.end
    } {link_with = "ep_12h_512s_96e_4ph_8pa_4g_1pf_d9_m0_f6_n1-1_n21_me_8971c3b5f9ae_kernels.a"}
    %_anonymous18 = aie.buffer(%tile_6_4) {address = 7936 : i32, mem_bank = 0 : i32, sym_name = "_anonymous18"} : memref<2xi32> 
    %core_6_4 = aie.core(%tile_6_4) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c32 = arith.constant 32 : index
      %c32_i32 = arith.constant 32 : i32
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous18[%c0] : memref<2xi32>
      memref.store %c0_i32, %_anonymous18[%c1] : memref<2xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb18
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb19
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb17
      %3 = arith.cmpi slt, %2, %c32 : index
      cf.cond_br %3, ^bb4, ^bb18
    ^bb4:  // pred: ^bb3
      %4 = index.casts %2 : index to i32
      %5 = arith.cmpi slt, %4, %c32_i32 : i32
      cf.cond_br %5, ^bb5, ^bb13
    ^bb5:  // pred: ^bb4
      aie.use_lock(%ffnUpOut_prod_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous18[%c0] : memref<2xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb8,
        0: ^bb6,
        1: ^bb7
      ]
    ^bb6:  // pred: ^bb5
      cf.br ^bb9(%ffnUpOut_buff_0 : memref<32x96xbf16>)
    ^bb7:  // pred: ^bb5
      cf.br ^bb9(%ffnUpOut_buff_1 : memref<32x96xbf16>)
    ^bb8:  // pred: ^bb5
      cf.br ^bb9(%ffnUpOut_buff_0 : memref<32x96xbf16>)
    ^bb9(%9: memref<32x96xbf16>):  // 3 preds: ^bb6, ^bb7, ^bb8
      func.call @ffn_zero_bf16_up_proj(%9) : (memref<32x96xbf16>) -> ()
      cf.br ^bb10(%c0 : index)
    ^bb10(%10: index):  // 2 preds: ^bb9, ^bb11
      %11 = arith.cmpi slt, %10, %c8 : index
      cf.cond_br %11, ^bb11, ^bb12
    ^bb11:  // pred: ^bb10
      aie.use_lock(%outLNBroadcast_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%outLNBroadcast_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous18[%c1] : memref<2xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous18[%c1] : memref<2xi32>
      %17 = arith.addi %10, %c1 : index
      cf.br ^bb10(%17 : index)
    ^bb12:  // pred: ^bb10
      aie.use_lock(%ffnUpOut_cons_lock_0, Release, 1)
      %18 = memref.load %_anonymous18[%c0] : memref<2xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous18[%c0] : memref<2xi32>
      cf.br ^bb17
    ^bb13:  // pred: ^bb4
      cf.br ^bb14(%c0 : index)
    ^bb14(%23: index):  // 2 preds: ^bb13, ^bb15
      %24 = arith.cmpi slt, %23, %c8 : index
      cf.cond_br %24, ^bb15, ^bb16
    ^bb15:  // pred: ^bb14
      aie.use_lock(%outLNBroadcast_cons_prod_lock_0, Release, 1)
      %25 = memref.load %_anonymous18[%c1] : memref<2xi32>
      %26 = arith.addi %25, %c1_i32 : i32
      %27 = arith.cmpi sge, %26, %c2_i32 : i32
      %28 = arith.subi %26, %c2_i32 : i32
      %29 = arith.select %27, %28, %26 : i32
      memref.store %29, %_anonymous18[%c1] : memref<2xi32>
      %30 = arith.addi %23, %c1 : index
      cf.br ^bb14(%30 : index)
    ^bb16:  // pred: ^bb14
      cf.br ^bb17
    ^bb17:  // 2 preds: ^bb12, ^bb16
      %31 = arith.addi %2, %c1 : index
      cf.br ^bb3(%31 : index)
    ^bb18:  // pred: ^bb3
      %32 = arith.addi %0, %c1 : index
      cf.br ^bb1(%32 : index)
    ^bb19:  // pred: ^bb1
      aie.end
    } {link_with = "ep_12h_512s_96e_4ph_8pa_4g_1pf_d9_m0_f6_n1-1_n21_me_8971c3b5f9ae_kernels.a", stack_size = 1792 : i32}
    %_anonymous19 = aie.buffer(%tile_7_4) {address = 9984 : i32, mem_bank = 0 : i32, sym_name = "_anonymous19"} : memref<4xi32> 
    %core_7_4 = aie.core(%tile_7_4) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c8 = arith.constant 8 : index
      %c31 = arith.constant 31 : index
      %c3 = arith.constant 3 : index
      %c2 = arith.constant 2 : index
      %c1_i32 = arith.constant 1 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous19[%c0] : memref<4xi32>
      memref.store %c0_i32, %_anonymous19[%c1] : memref<4xi32>
      memref.store %c0_i32, %_anonymous19[%c2] : memref<4xi32>
      memref.store %c0_i32, %_anonymous19[%c3] : memref<4xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb18
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb19
    ^bb2:  // pred: ^bb1
      aie.use_lock(%ffnUpOut_cons_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c8 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%ffnDownPart_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @ffn_zero_bf16_down_proj(%ffnDownPart_buff_0) : (memref<32x96xbf16>) -> ()
      aie.use_lock(%ffnDownPart_cons_lock_0, Release, 1)
      %4 = memref.load %_anonymous19[%c1] : memref<4xi32>
      %5 = arith.addi %4, %c1_i32 : i32
      %6 = arith.cmpi sge, %5, %c1_i32 : i32
      %7 = arith.select %6, %4, %5 : i32
      memref.store %7, %_anonymous19[%c1] : memref<4xi32>
      %8 = arith.addi %2, %c1 : index
      cf.br ^bb3(%8 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%ffnUpOut_prod_lock_0, Release, 1)
      %9 = memref.load %_anonymous19[%c0] : memref<4xi32>
      %10 = arith.addi %9, %c1_i32 : i32
      %11 = arith.cmpi sge, %10, %c2_i32 : i32
      %12 = arith.subi %10, %c2_i32 : i32
      %13 = arith.select %11, %12, %10 : i32
      memref.store %13, %_anonymous19[%c0] : memref<4xi32>
      cf.br ^bb6(%c0 : index)
    ^bb6(%14: index):  // 2 preds: ^bb5, ^bb10
      %15 = arith.cmpi slt, %14, %c31 : index
      cf.cond_br %15, ^bb7, ^bb11
    ^bb7:  // pred: ^bb6
      aie.use_lock(%ffnUpOut_cons_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb8(%c0 : index)
    ^bb8(%16: index):  // 2 preds: ^bb7, ^bb9
      %17 = arith.cmpi slt, %16, %c8 : index
      cf.cond_br %17, ^bb9, ^bb10
    ^bb9:  // pred: ^bb8
      aie.use_lock(%ffnDownAccum_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%ffnDownPart_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @ffn_zero_bf16_down_proj(%ffnDownPart_buff_0) : (memref<32x96xbf16>) -> ()
      aie.use_lock(%ffnDownPart_cons_lock_0, Release, 1)
      %18 = memref.load %_anonymous19[%c1] : memref<4xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c1_i32 : i32
      %21 = arith.select %20, %18, %19 : i32
      memref.store %21, %_anonymous19[%c1] : memref<4xi32>
      aie.use_lock(%ffnDownAccum_cons_prod_lock_0, Release, 1)
      %22 = memref.load %_anonymous19[%c2] : memref<4xi32>
      %23 = arith.addi %22, %c1_i32 : i32
      %24 = arith.cmpi sge, %23, %c1_i32 : i32
      %25 = arith.select %24, %22, %23 : i32
      memref.store %25, %_anonymous19[%c2] : memref<4xi32>
      %26 = arith.addi %16, %c1 : index
      cf.br ^bb8(%26 : index)
    ^bb10:  // pred: ^bb8
      aie.use_lock(%ffnUpOut_prod_lock_0, Release, 1)
      %27 = memref.load %_anonymous19[%c0] : memref<4xi32>
      %28 = arith.addi %27, %c1_i32 : i32
      %29 = arith.cmpi sge, %28, %c2_i32 : i32
      %30 = arith.subi %28, %c2_i32 : i32
      %31 = arith.select %29, %30, %28 : i32
      memref.store %31, %_anonymous19[%c0] : memref<4xi32>
      %32 = arith.addi %14, %c1 : index
      cf.br ^bb6(%32 : index)
    ^bb11:  // pred: ^bb6
      cf.br ^bb12(%c0 : index)
    ^bb12(%33: index):  // 2 preds: ^bb11, ^bb17
      %34 = arith.cmpi slt, %33, %c8 : index
      cf.cond_br %34, ^bb13, ^bb18
    ^bb13:  // pred: ^bb12
      aie.use_lock(%ffnDownOut_prod_lock_0, AcquireGreaterEqual, 1)
      %35 = memref.load %_anonymous19[%c3] : memref<4xi32>
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.index_cast %36 : index to i32
      cf.switch %37 : i32, [
        default: ^bb16,
        0: ^bb14,
        1: ^bb15
      ]
    ^bb14:  // pred: ^bb13
      cf.br ^bb17(%ffnDownOut_buff_0 : memref<32x96xbf16>)
    ^bb15:  // pred: ^bb13
      cf.br ^bb17(%ffnDownOut_buff_1 : memref<32x96xbf16>)
    ^bb16:  // pred: ^bb13
      cf.br ^bb17(%ffnDownOut_buff_0 : memref<32x96xbf16>)
    ^bb17(%38: memref<32x96xbf16>):  // 3 preds: ^bb14, ^bb15, ^bb16
      aie.use_lock(%ffnDownAccum_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @ffn_zero_bf16_down_proj(%38) : (memref<32x96xbf16>) -> ()
      aie.use_lock(%ffnDownAccum_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous19[%c2] : memref<4xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c1_i32 : i32
      %42 = arith.select %41, %39, %40 : i32
      memref.store %42, %_anonymous19[%c2] : memref<4xi32>
      aie.use_lock(%ffnDownOut_cons_lock_0, Release, 1)
      %43 = memref.load %_anonymous19[%c3] : memref<4xi32>
      %44 = arith.addi %43, %c1_i32 : i32
      %45 = arith.cmpi sge, %44, %c2_i32 : i32
      %46 = arith.subi %44, %c2_i32 : i32
      %47 = arith.select %45, %46, %44 : i32
      memref.store %47, %_anonymous19[%c3] : memref<4xi32>
      %48 = arith.addi %33, %c1 : index
      cf.br ^bb12(%48 : index)
    ^bb18:  // pred: ^bb12
      %49 = arith.addi %0, %c1 : index
      cf.br ^bb1(%49 : index)
    ^bb19:  // pred: ^bb1
      aie.end
    } {link_with = "ep_12h_512s_96e_4ph_8pa_4g_1pf_d9_m0_f6_n1-1_n21_me_8971c3b5f9ae_kernels.a", stack_size = 3840 : i32}
    %_anonymous20 = aie.buffer(%tile_7_5) {address = 55296 : i32, mem_bank = 3 : i32, sym_name = "_anonymous20"} : memref<3xi32> 
    %core_7_5 = aie.core(%tile_7_5) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c8 = arith.constant 8 : index
      %c3072_i32 = arith.constant 3072 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous20[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous20[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous20[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb13
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb14
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb12
      %3 = arith.cmpi slt, %2, %c8 : index
      cf.cond_br %3, ^bb4, ^bb13
    ^bb4:  // pred: ^bb3
      aie.use_lock(%ffnDownOut_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%ffnRIn_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous20[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%ffnRIn_cons_buff_0 : memref<32x96xbf16>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%ffnRIn_cons_buff_1 : memref<32x96xbf16>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%ffnRIn_cons_buff_0 : memref<32x96xbf16>)
    ^bb8(%7: memref<32x96xbf16>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%outLN2_prod_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous20[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%outLN2_buff_0 : memref<32x96xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%outLN2_buff_1 : memref<32x96xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%outLN2_buff_0 : memref<32x96xbf16>)
    ^bb12(%11: memref<32x96xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @passThroughLine_o_proj(%7, %11, %c3072_i32) : (memref<32x96xbf16>, memref<32x96xbf16>, i32) -> ()
      aie.use_lock(%outLN2_cons_lock_0, Release, 1)
      %12 = memref.load %_anonymous20[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous20[%c2] : memref<3xi32>
      aie.use_lock(%ffnDownOut_prod_lock_0, Release, 1)
      %17 = memref.load %_anonymous20[%c0] : memref<3xi32>
      %18 = arith.addi %17, %c1_i32 : i32
      %19 = arith.cmpi sge, %18, %c2_i32 : i32
      %20 = arith.subi %18, %c2_i32 : i32
      %21 = arith.select %19, %20, %18 : i32
      memref.store %21, %_anonymous20[%c0] : memref<3xi32>
      aie.use_lock(%ffnRIn_cons_prod_lock_0, Release, 1)
      %22 = memref.load %_anonymous20[%c1] : memref<3xi32>
      %23 = arith.addi %22, %c1_i32 : i32
      %24 = arith.cmpi sge, %23, %c2_i32 : i32
      %25 = arith.subi %23, %c2_i32 : i32
      %26 = arith.select %24, %25, %23 : i32
      memref.store %26, %_anonymous20[%c1] : memref<3xi32>
      %27 = arith.addi %2, %c1 : index
      cf.br ^bb3(%27 : index)
    ^bb13:  // pred: ^bb3
      %28 = arith.addi %0, %c1 : index
      cf.br ^bb1(%28 : index)
    ^bb14:  // pred: ^bb1
      aie.end
    } {link_with = "ep_12h_512s_96e_4ph_8pa_4g_1pf_d9_m0_f6_n1-1_n21_me_8971c3b5f9ae_kernels.a", stack_size = 3840 : i32}
    aie.runtime_sequence(%arg0: memref<768x768xbf16>, %arg1: memref<1536x768xbf16>, %arg2: memref<1024x768xbf16>, %arg3: memref<2359296xbf16>, %arg4: memref<2359296xbf16>) {
      %0 = aiex.dma_configure_task_for @inQ_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 0, 24576, [<size = 1, stride = 0>, <size = 12, stride = 64>, <size = 32, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      aiex.dma_free_task(%0)
      %1 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393216, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786432, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 0, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%3)
      aiex.dma_await_task(%1)
      aiex.dma_await_task(%2)
      aiex.dma_await_task(%3)
      %4 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393472, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786688, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 196608, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%6)
      aiex.dma_await_task(%4)
      aiex.dma_await_task(%5)
      aiex.dma_await_task(%6)
      %7 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393728, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786944, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 393216, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%9)
      aiex.dma_await_task(%7)
      aiex.dma_await_task(%8)
      aiex.dma_await_task(%9)
      %10 = aiex.dma_configure_task_for @inR_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 393216, 24576, [<size = 22, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 21 : i32}
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @inR_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 393216, 24576, [<size = 10, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 9 : i32}
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @memLN2_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 0, 24576, [<size = 1, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%12)
      aiex.dma_await_task(%12)
      aiex.dma_await_task(%10)
      aiex.dma_await_task(%11)
      %13 = aiex.dma_configure_task_for @inQ_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 24576, 24576, [<size = 1, stride = 0>, <size = 12, stride = 64>, <size = 32, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%13)
      aiex.dma_free_task(%13)
      %14 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393216, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786432, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 0, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%16)
      aiex.dma_await_task(%14)
      aiex.dma_await_task(%15)
      aiex.dma_await_task(%16)
      %17 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393472, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786688, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 196608, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%19)
      aiex.dma_await_task(%17)
      aiex.dma_await_task(%18)
      aiex.dma_await_task(%19)
      %20 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393728, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786944, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 393216, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%22)
      aiex.dma_await_task(%20)
      aiex.dma_await_task(%21)
      aiex.dma_await_task(%22)
      %23 = aiex.dma_configure_task_for @inR_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 417792, 24576, [<size = 22, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 21 : i32}
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @inR_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 417792, 24576, [<size = 10, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 9 : i32}
      aiex.dma_start_task(%24)
      %25 = aiex.dma_configure_task_for @memLN2_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 24576, 24576, [<size = 1, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%25)
      aiex.dma_await_task(%25)
      aiex.dma_await_task(%23)
      aiex.dma_await_task(%24)
      %26 = aiex.dma_configure_task_for @inQ_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 49152, 24576, [<size = 1, stride = 0>, <size = 12, stride = 64>, <size = 32, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%26)
      aiex.dma_free_task(%26)
      %27 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393216, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%27)
      %28 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786432, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%28)
      %29 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 0, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%29)
      aiex.dma_await_task(%27)
      aiex.dma_await_task(%28)
      aiex.dma_await_task(%29)
      %30 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393472, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%30)
      %31 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786688, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%31)
      %32 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 196608, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%32)
      aiex.dma_await_task(%30)
      aiex.dma_await_task(%31)
      aiex.dma_await_task(%32)
      %33 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393728, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%33)
      %34 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786944, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%34)
      %35 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 393216, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%35)
      aiex.dma_await_task(%33)
      aiex.dma_await_task(%34)
      aiex.dma_await_task(%35)
      %36 = aiex.dma_configure_task_for @inR_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 442368, 24576, [<size = 22, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 21 : i32}
      aiex.dma_start_task(%36)
      %37 = aiex.dma_configure_task_for @inR_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 442368, 24576, [<size = 10, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 9 : i32}
      aiex.dma_start_task(%37)
      %38 = aiex.dma_configure_task_for @memLN2_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 49152, 24576, [<size = 1, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%38)
      aiex.dma_await_task(%38)
      aiex.dma_await_task(%36)
      aiex.dma_await_task(%37)
      %39 = aiex.dma_configure_task_for @inQ_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 73728, 24576, [<size = 1, stride = 0>, <size = 12, stride = 64>, <size = 32, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%39)
      aiex.dma_free_task(%39)
      %40 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393216, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%40)
      %41 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786432, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%41)
      %42 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 0, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%42)
      aiex.dma_await_task(%40)
      aiex.dma_await_task(%41)
      aiex.dma_await_task(%42)
      %43 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393472, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%43)
      %44 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786688, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%44)
      %45 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 196608, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%45)
      aiex.dma_await_task(%43)
      aiex.dma_await_task(%44)
      aiex.dma_await_task(%45)
      %46 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393728, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%46)
      %47 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786944, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%47)
      %48 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 393216, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%48)
      aiex.dma_await_task(%46)
      aiex.dma_await_task(%47)
      aiex.dma_await_task(%48)
      %49 = aiex.dma_configure_task_for @inR_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 466944, 24576, [<size = 22, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 21 : i32}
      aiex.dma_start_task(%49)
      %50 = aiex.dma_configure_task_for @inR_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 466944, 24576, [<size = 10, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 9 : i32}
      aiex.dma_start_task(%50)
      %51 = aiex.dma_configure_task_for @memLN2_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 73728, 24576, [<size = 1, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%51)
      aiex.dma_await_task(%51)
      aiex.dma_await_task(%49)
      aiex.dma_await_task(%50)
      %52 = aiex.dma_configure_task_for @inQ_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 98304, 24576, [<size = 1, stride = 0>, <size = 12, stride = 64>, <size = 32, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%52)
      aiex.dma_free_task(%52)
      %53 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393216, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%53)
      %54 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786432, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%54)
      %55 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 0, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%55)
      aiex.dma_await_task(%53)
      aiex.dma_await_task(%54)
      aiex.dma_await_task(%55)
      %56 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393472, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%56)
      %57 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786688, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%57)
      %58 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 196608, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%58)
      aiex.dma_await_task(%56)
      aiex.dma_await_task(%57)
      aiex.dma_await_task(%58)
      %59 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393728, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%59)
      %60 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786944, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%60)
      %61 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 393216, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%61)
      aiex.dma_await_task(%59)
      aiex.dma_await_task(%60)
      aiex.dma_await_task(%61)
      %62 = aiex.dma_configure_task_for @inR_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 491520, 24576, [<size = 22, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 21 : i32}
      aiex.dma_start_task(%62)
      %63 = aiex.dma_configure_task_for @inR_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 491520, 24576, [<size = 10, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 9 : i32}
      aiex.dma_start_task(%63)
      %64 = aiex.dma_configure_task_for @memLN2_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 98304, 24576, [<size = 1, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%64)
      aiex.dma_await_task(%64)
      aiex.dma_await_task(%62)
      aiex.dma_await_task(%63)
      %65 = aiex.dma_configure_task_for @inQ_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 122880, 24576, [<size = 1, stride = 0>, <size = 12, stride = 64>, <size = 32, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%65)
      aiex.dma_free_task(%65)
      %66 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393216, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%66)
      %67 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786432, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%67)
      %68 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 0, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%68)
      aiex.dma_await_task(%66)
      aiex.dma_await_task(%67)
      aiex.dma_await_task(%68)
      %69 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393472, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%69)
      %70 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786688, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%70)
      %71 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 196608, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%71)
      aiex.dma_await_task(%69)
      aiex.dma_await_task(%70)
      aiex.dma_await_task(%71)
      %72 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393728, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%72)
      %73 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786944, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%73)
      %74 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 393216, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%74)
      aiex.dma_await_task(%72)
      aiex.dma_await_task(%73)
      aiex.dma_await_task(%74)
      %75 = aiex.dma_configure_task_for @inR_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 516096, 24576, [<size = 22, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 21 : i32}
      aiex.dma_start_task(%75)
      %76 = aiex.dma_configure_task_for @inR_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 516096, 24576, [<size = 10, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 9 : i32}
      aiex.dma_start_task(%76)
      %77 = aiex.dma_configure_task_for @memLN2_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 122880, 24576, [<size = 1, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%77)
      aiex.dma_await_task(%77)
      aiex.dma_await_task(%75)
      aiex.dma_await_task(%76)
      %78 = aiex.dma_configure_task_for @inQ_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 147456, 24576, [<size = 1, stride = 0>, <size = 12, stride = 64>, <size = 32, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%78)
      aiex.dma_free_task(%78)
      %79 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393216, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%79)
      %80 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786432, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%80)
      %81 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 0, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%81)
      aiex.dma_await_task(%79)
      aiex.dma_await_task(%80)
      aiex.dma_await_task(%81)
      %82 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393472, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%82)
      %83 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786688, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%83)
      %84 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 196608, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%84)
      aiex.dma_await_task(%82)
      aiex.dma_await_task(%83)
      aiex.dma_await_task(%84)
      %85 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393728, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%85)
      %86 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786944, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%86)
      %87 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 393216, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%87)
      aiex.dma_await_task(%85)
      aiex.dma_await_task(%86)
      aiex.dma_await_task(%87)
      %88 = aiex.dma_configure_task_for @inR_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 540672, 24576, [<size = 22, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 21 : i32}
      aiex.dma_start_task(%88)
      %89 = aiex.dma_configure_task_for @inR_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 540672, 24576, [<size = 10, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 9 : i32}
      aiex.dma_start_task(%89)
      %90 = aiex.dma_configure_task_for @memLN2_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 147456, 24576, [<size = 1, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%90)
      aiex.dma_await_task(%90)
      aiex.dma_await_task(%88)
      aiex.dma_await_task(%89)
      %91 = aiex.dma_configure_task_for @inQ_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 172032, 24576, [<size = 1, stride = 0>, <size = 12, stride = 64>, <size = 32, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%91)
      aiex.dma_free_task(%91)
      %92 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393216, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%92)
      %93 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786432, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%93)
      %94 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 0, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%94)
      aiex.dma_await_task(%92)
      aiex.dma_await_task(%93)
      aiex.dma_await_task(%94)
      %95 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393472, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%95)
      %96 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786688, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%96)
      %97 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 196608, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%97)
      aiex.dma_await_task(%95)
      aiex.dma_await_task(%96)
      aiex.dma_await_task(%97)
      %98 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393728, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%98)
      %99 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786944, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%99)
      %100 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 393216, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%100)
      aiex.dma_await_task(%98)
      aiex.dma_await_task(%99)
      aiex.dma_await_task(%100)
      %101 = aiex.dma_configure_task_for @inR_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 565248, 24576, [<size = 22, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 21 : i32}
      aiex.dma_start_task(%101)
      %102 = aiex.dma_configure_task_for @inR_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 565248, 24576, [<size = 10, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 9 : i32}
      aiex.dma_start_task(%102)
      %103 = aiex.dma_configure_task_for @memLN2_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 172032, 24576, [<size = 1, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%103)
      aiex.dma_await_task(%103)
      aiex.dma_await_task(%101)
      aiex.dma_await_task(%102)
      %104 = aiex.dma_configure_task_for @inQ_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 196608, 24576, [<size = 1, stride = 0>, <size = 12, stride = 64>, <size = 32, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%104)
      aiex.dma_free_task(%104)
      %105 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393216, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%105)
      %106 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786432, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%106)
      %107 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 0, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%107)
      aiex.dma_await_task(%105)
      aiex.dma_await_task(%106)
      aiex.dma_await_task(%107)
      %108 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393472, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%108)
      %109 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786688, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%109)
      %110 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 196608, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%110)
      aiex.dma_await_task(%108)
      aiex.dma_await_task(%109)
      aiex.dma_await_task(%110)
      %111 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393728, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%111)
      %112 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786944, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%112)
      %113 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 393216, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%113)
      aiex.dma_await_task(%111)
      aiex.dma_await_task(%112)
      aiex.dma_await_task(%113)
      %114 = aiex.dma_configure_task_for @inR_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 589824, 24576, [<size = 22, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 21 : i32}
      aiex.dma_start_task(%114)
      %115 = aiex.dma_configure_task_for @inR_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 589824, 24576, [<size = 10, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 9 : i32}
      aiex.dma_start_task(%115)
      %116 = aiex.dma_configure_task_for @memLN2_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 196608, 24576, [<size = 1, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%116)
      aiex.dma_await_task(%116)
      aiex.dma_await_task(%114)
      aiex.dma_await_task(%115)
      %117 = aiex.dma_configure_task_for @inQ_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 221184, 24576, [<size = 1, stride = 0>, <size = 12, stride = 64>, <size = 32, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%117)
      aiex.dma_free_task(%117)
      %118 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393216, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%118)
      %119 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786432, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%119)
      %120 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 0, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%120)
      aiex.dma_await_task(%118)
      aiex.dma_await_task(%119)
      aiex.dma_await_task(%120)
      %121 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393472, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%121)
      %122 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786688, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%122)
      %123 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 196608, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%123)
      aiex.dma_await_task(%121)
      aiex.dma_await_task(%122)
      aiex.dma_await_task(%123)
      %124 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393728, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%124)
      %125 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786944, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%125)
      %126 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 393216, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%126)
      aiex.dma_await_task(%124)
      aiex.dma_await_task(%125)
      aiex.dma_await_task(%126)
      %127 = aiex.dma_configure_task_for @inR_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 614400, 24576, [<size = 22, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 21 : i32}
      aiex.dma_start_task(%127)
      %128 = aiex.dma_configure_task_for @inR_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 614400, 24576, [<size = 10, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 9 : i32}
      aiex.dma_start_task(%128)
      %129 = aiex.dma_configure_task_for @memLN2_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 221184, 24576, [<size = 1, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%129)
      aiex.dma_await_task(%129)
      aiex.dma_await_task(%127)
      aiex.dma_await_task(%128)
      %130 = aiex.dma_configure_task_for @inQ_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 245760, 24576, [<size = 1, stride = 0>, <size = 12, stride = 64>, <size = 32, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%130)
      aiex.dma_free_task(%130)
      %131 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393216, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%131)
      %132 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786432, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%132)
      %133 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 0, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%133)
      aiex.dma_await_task(%131)
      aiex.dma_await_task(%132)
      aiex.dma_await_task(%133)
      %134 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393472, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%134)
      %135 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786688, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%135)
      %136 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 196608, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%136)
      aiex.dma_await_task(%134)
      aiex.dma_await_task(%135)
      aiex.dma_await_task(%136)
      %137 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393728, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%137)
      %138 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786944, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%138)
      %139 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 393216, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%139)
      aiex.dma_await_task(%137)
      aiex.dma_await_task(%138)
      aiex.dma_await_task(%139)
      %140 = aiex.dma_configure_task_for @inR_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 638976, 24576, [<size = 22, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 21 : i32}
      aiex.dma_start_task(%140)
      %141 = aiex.dma_configure_task_for @inR_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 638976, 24576, [<size = 10, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 9 : i32}
      aiex.dma_start_task(%141)
      %142 = aiex.dma_configure_task_for @memLN2_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 245760, 24576, [<size = 1, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%142)
      aiex.dma_await_task(%142)
      aiex.dma_await_task(%140)
      aiex.dma_await_task(%141)
      %143 = aiex.dma_configure_task_for @inQ_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 270336, 24576, [<size = 1, stride = 0>, <size = 12, stride = 64>, <size = 32, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%143)
      aiex.dma_free_task(%143)
      %144 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393216, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%144)
      %145 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786432, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%145)
      %146 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 0, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%146)
      aiex.dma_await_task(%144)
      aiex.dma_await_task(%145)
      aiex.dma_await_task(%146)
      %147 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393472, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%147)
      %148 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786688, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%148)
      %149 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 196608, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%149)
      aiex.dma_await_task(%147)
      aiex.dma_await_task(%148)
      aiex.dma_await_task(%149)
      %150 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393728, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%150)
      %151 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786944, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%151)
      %152 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 393216, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%152)
      aiex.dma_await_task(%150)
      aiex.dma_await_task(%151)
      aiex.dma_await_task(%152)
      %153 = aiex.dma_configure_task_for @inR_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 663552, 24576, [<size = 22, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 21 : i32}
      aiex.dma_start_task(%153)
      %154 = aiex.dma_configure_task_for @inR_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 663552, 24576, [<size = 10, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 9 : i32}
      aiex.dma_start_task(%154)
      %155 = aiex.dma_configure_task_for @memLN2_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 270336, 24576, [<size = 1, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%155)
      aiex.dma_await_task(%155)
      aiex.dma_await_task(%153)
      aiex.dma_await_task(%154)
      %156 = aiex.dma_configure_task_for @inQ_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 294912, 24576, [<size = 1, stride = 0>, <size = 12, stride = 64>, <size = 32, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%156)
      aiex.dma_free_task(%156)
      %157 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393216, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%157)
      %158 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786432, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%158)
      %159 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 0, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%159)
      aiex.dma_await_task(%157)
      aiex.dma_await_task(%158)
      aiex.dma_await_task(%159)
      %160 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393472, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%160)
      %161 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786688, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%161)
      %162 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 196608, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%162)
      aiex.dma_await_task(%160)
      aiex.dma_await_task(%161)
      aiex.dma_await_task(%162)
      %163 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393728, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%163)
      %164 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786944, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%164)
      %165 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 393216, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%165)
      aiex.dma_await_task(%163)
      aiex.dma_await_task(%164)
      aiex.dma_await_task(%165)
      %166 = aiex.dma_configure_task_for @inR_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 688128, 24576, [<size = 22, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 21 : i32}
      aiex.dma_start_task(%166)
      %167 = aiex.dma_configure_task_for @inR_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 688128, 24576, [<size = 10, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 9 : i32}
      aiex.dma_start_task(%167)
      %168 = aiex.dma_configure_task_for @memLN2_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 294912, 24576, [<size = 1, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%168)
      aiex.dma_await_task(%168)
      aiex.dma_await_task(%166)
      aiex.dma_await_task(%167)
      %169 = aiex.dma_configure_task_for @inQ_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 319488, 24576, [<size = 1, stride = 0>, <size = 12, stride = 64>, <size = 32, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%169)
      aiex.dma_free_task(%169)
      %170 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393216, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%170)
      %171 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786432, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%171)
      %172 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 0, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%172)
      aiex.dma_await_task(%170)
      aiex.dma_await_task(%171)
      aiex.dma_await_task(%172)
      %173 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393472, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%173)
      %174 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786688, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%174)
      %175 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 196608, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%175)
      aiex.dma_await_task(%173)
      aiex.dma_await_task(%174)
      aiex.dma_await_task(%175)
      %176 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393728, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%176)
      %177 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786944, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%177)
      %178 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 393216, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%178)
      aiex.dma_await_task(%176)
      aiex.dma_await_task(%177)
      aiex.dma_await_task(%178)
      %179 = aiex.dma_configure_task_for @inR_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 712704, 24576, [<size = 22, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 21 : i32}
      aiex.dma_start_task(%179)
      %180 = aiex.dma_configure_task_for @inR_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 712704, 24576, [<size = 10, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 9 : i32}
      aiex.dma_start_task(%180)
      %181 = aiex.dma_configure_task_for @memLN2_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 319488, 24576, [<size = 1, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%181)
      aiex.dma_await_task(%181)
      aiex.dma_await_task(%179)
      aiex.dma_await_task(%180)
      %182 = aiex.dma_configure_task_for @inQ_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 344064, 24576, [<size = 1, stride = 0>, <size = 12, stride = 64>, <size = 32, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%182)
      aiex.dma_free_task(%182)
      %183 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393216, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%183)
      %184 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786432, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%184)
      %185 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 0, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%185)
      aiex.dma_await_task(%183)
      aiex.dma_await_task(%184)
      aiex.dma_await_task(%185)
      %186 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393472, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%186)
      %187 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786688, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%187)
      %188 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 196608, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%188)
      aiex.dma_await_task(%186)
      aiex.dma_await_task(%187)
      aiex.dma_await_task(%188)
      %189 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393728, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%189)
      %190 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786944, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%190)
      %191 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 393216, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%191)
      aiex.dma_await_task(%189)
      aiex.dma_await_task(%190)
      aiex.dma_await_task(%191)
      %192 = aiex.dma_configure_task_for @inR_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 737280, 24576, [<size = 22, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 21 : i32}
      aiex.dma_start_task(%192)
      %193 = aiex.dma_configure_task_for @inR_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 737280, 24576, [<size = 10, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 9 : i32}
      aiex.dma_start_task(%193)
      %194 = aiex.dma_configure_task_for @memLN2_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 344064, 24576, [<size = 1, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%194)
      aiex.dma_await_task(%194)
      aiex.dma_await_task(%192)
      aiex.dma_await_task(%193)
      %195 = aiex.dma_configure_task_for @inQ_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 368640, 24576, [<size = 1, stride = 0>, <size = 12, stride = 64>, <size = 32, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%195)
      aiex.dma_free_task(%195)
      %196 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393216, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%196)
      %197 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786432, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%197)
      %198 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 0, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%198)
      aiex.dma_await_task(%196)
      aiex.dma_await_task(%197)
      aiex.dma_await_task(%198)
      %199 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393472, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%199)
      %200 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786688, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%200)
      %201 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 196608, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%201)
      aiex.dma_await_task(%199)
      aiex.dma_await_task(%200)
      aiex.dma_await_task(%201)
      %202 = aiex.dma_configure_task_for @inK_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 393728, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%202)
      %203 = aiex.dma_configure_task_for @inV_shim_alloc {
        aie.dma_bd(%arg1 : memref<1536x768xbf16>, 786944, 16384, [<size = 8, stride = 49152>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%203)
      %204 = aiex.dma_configure_task_for @inOW_shim_alloc {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 393216, 24576, [<size = 8, stride = 96>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%204)
      aiex.dma_await_task(%202)
      aiex.dma_await_task(%203)
      aiex.dma_await_task(%204)
      %205 = aiex.dma_configure_task_for @inR_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 761856, 24576, [<size = 22, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 21 : i32}
      aiex.dma_start_task(%205)
      %206 = aiex.dma_configure_task_for @inR_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 761856, 24576, [<size = 10, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 9 : i32}
      aiex.dma_start_task(%206)
      %207 = aiex.dma_configure_task_for @memLN2_shim_alloc {
        aie.dma_bd(%arg2 : memref<1024x768xbf16>, 368640, 24576, [<size = 1, stride = 0>, <size = 8, stride = 96>, <size = 32, stride = 768>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%207)
      aiex.dma_await_task(%207)
      aiex.dma_await_task(%205)
      aiex.dma_await_task(%206)
    }
    %mem_6_5 = aie.mem(%tile_6_5) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%ln1Replay_src_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%ln1Replay_src : memref<32x96xbf16>, 0, 3072) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%ln1Replay_src_empty, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%ln1Replay_dst_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%ln1Replay_dst : memref<32x96xbf16>, 0, 3072) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%ln1Replay_dst_full, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 1, ^bb5, ^bb7)
    ^bb5:  // 2 preds: ^bb4, ^bb6
      aie.use_lock(%ln1Norm_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ln1Norm_buff_0 : memref<32x96xbf16>, 0, 3072) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%ln1Norm_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%ln1Norm_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ln1Norm_buff_1 : memref<32x96xbf16>, 0, 3072) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%ln1Norm_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb7:  // pred: ^bb4
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb10)
    ^bb8:  // 2 preds: ^bb7, ^bb9
      aie.use_lock(%outOProjInput_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOProjInput_cons_buff_0 : memref<32x96xbf16>, 0, 3072) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%outOProjInput_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb9
    ^bb9:  // pred: ^bb8
      aie.use_lock(%outOProjInput_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOProjInput_cons_buff_1 : memref<32x96xbf16>, 0, 3072) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%outOProjInput_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb10:  // pred: ^bb7
      aie.end
    }
    %memtile_dma_5_1 = aie.memtile_dma(%mem_tile_5_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb9)
    ^bb1:  // 2 preds: ^bb0, ^bb8
      aie.use_lock(%ln1Replay_row_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%ln1Replay_row : memref<24576xbf16>, 0, 3072) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.dma_bd(%ln1Replay_row : memref<24576xbf16>, 3072, 3072) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.next_bd ^bb3
    ^bb3:  // pred: ^bb2
      aie.dma_bd(%ln1Replay_row : memref<24576xbf16>, 6144, 3072) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.dma_bd(%ln1Replay_row : memref<24576xbf16>, 9216, 3072) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.dma_bd(%ln1Replay_row : memref<24576xbf16>, 12288, 3072) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.dma_bd(%ln1Replay_row : memref<24576xbf16>, 15360, 3072) {bd_id = 5 : i32, next_bd_id = 6 : i32}
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.dma_bd(%ln1Replay_row : memref<24576xbf16>, 18432, 3072) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.dma_bd(%ln1Replay_row : memref<24576xbf16>, 21504, 3072) {bd_id = 7 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%ln1Replay_row_full, Release, 1)
      aie.next_bd ^bb1
    ^bb9:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb10, ^bb18)
    ^bb10:  // 2 preds: ^bb9, ^bb17
      aie.use_lock(%ln1Replay_row_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%ln1Replay_row : memref<24576xbf16>, 0, 3072) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.dma_bd(%ln1Replay_row : memref<24576xbf16>, 3072, 3072) {bd_id = 25 : i32, next_bd_id = 26 : i32}
      aie.next_bd ^bb12
    ^bb12:  // pred: ^bb11
      aie.dma_bd(%ln1Replay_row : memref<24576xbf16>, 6144, 3072) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.next_bd ^bb13
    ^bb13:  // pred: ^bb12
      aie.dma_bd(%ln1Replay_row : memref<24576xbf16>, 9216, 3072) {bd_id = 27 : i32, next_bd_id = 28 : i32}
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.dma_bd(%ln1Replay_row : memref<24576xbf16>, 12288, 3072) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.next_bd ^bb15
    ^bb15:  // pred: ^bb14
      aie.dma_bd(%ln1Replay_row : memref<24576xbf16>, 15360, 3072) {bd_id = 29 : i32, next_bd_id = 30 : i32}
      aie.next_bd ^bb16
    ^bb16:  // pred: ^bb15
      aie.dma_bd(%ln1Replay_row : memref<24576xbf16>, 18432, 3072) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.dma_bd(%ln1Replay_row : memref<24576xbf16>, 21504, 3072) {bd_id = 31 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%ln1Replay_row_empty, Release, 1)
      aie.next_bd ^bb10
    ^bb18:  // pred: ^bb9
      aie.end
    }
    %memtile_dma_6_1 = aie.memtile_dma(%mem_tile_6_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb9)
    ^bb1:  // 2 preds: ^bb0, ^bb8
      aie.use_lock(%ffnDownPart_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownPart_cons_buff_0 : memref<32x96xbf16>, 0, 3072) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%ffnDownPart_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%ffnDownPart_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownPart_cons_buff_1 : memref<32x96xbf16>, 0, 3072) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%ffnDownPart_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb3
    ^bb3:  // pred: ^bb2
      aie.use_lock(%ffnDownPart_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownPart_cons_buff_2 : memref<32x96xbf16>, 0, 3072) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%ffnDownPart_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%ffnDownPart_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownPart_cons_buff_3 : memref<32x96xbf16>, 0, 3072) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%ffnDownPart_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%ffnDownPart_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownPart_cons_buff_4 : memref<32x96xbf16>, 0, 3072) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%ffnDownPart_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%ffnDownPart_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownPart_cons_buff_5 : memref<32x96xbf16>, 0, 3072) {bd_id = 5 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%ffnDownPart_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%ffnDownPart_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownPart_cons_buff_6 : memref<32x96xbf16>, 0, 3072) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%ffnDownPart_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%ffnDownPart_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownPart_cons_buff_7 : memref<32x96xbf16>, 0, 3072) {bd_id = 7 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%ffnDownPart_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb9:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb10, ^bb18)
    ^bb10:  // 2 preds: ^bb9, ^bb17
      aie.use_lock(%ffnDownPart_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownPart_cons_buff_0 : memref<32x96xbf16>, 0, 3072) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%ffnDownPart_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%ffnDownPart_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownPart_cons_buff_1 : memref<32x96xbf16>, 0, 3072) {bd_id = 9 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%ffnDownPart_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb12
    ^bb12:  // pred: ^bb11
      aie.use_lock(%ffnDownPart_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownPart_cons_buff_2 : memref<32x96xbf16>, 0, 3072) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%ffnDownPart_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb13:  // pred: ^bb12
      aie.use_lock(%ffnDownPart_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownPart_cons_buff_3 : memref<32x96xbf16>, 0, 3072) {bd_id = 11 : i32, next_bd_id = 12 : i32}
      aie.use_lock(%ffnDownPart_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%ffnDownPart_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownPart_cons_buff_4 : memref<32x96xbf16>, 0, 3072) {bd_id = 12 : i32, next_bd_id = 13 : i32}
      aie.use_lock(%ffnDownPart_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb15
    ^bb15:  // pred: ^bb14
      aie.use_lock(%ffnDownPart_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownPart_cons_buff_5 : memref<32x96xbf16>, 0, 3072) {bd_id = 13 : i32, next_bd_id = 14 : i32}
      aie.use_lock(%ffnDownPart_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb16
    ^bb16:  // pred: ^bb15
      aie.use_lock(%ffnDownPart_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownPart_cons_buff_6 : memref<32x96xbf16>, 0, 3072) {bd_id = 14 : i32, next_bd_id = 15 : i32}
      aie.use_lock(%ffnDownPart_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%ffnDownPart_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownPart_cons_buff_7 : memref<32x96xbf16>, 0, 3072) {bd_id = 15 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%ffnDownPart_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb18:  // pred: ^bb9
      aie.end
    }
    %mem_7_4 = aie.mem(%tile_7_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%ffnDownAccum_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownAccum_cons_buff_0 : memref<32x96xbf16>, 0, 3072) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%ffnDownAccum_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%ffnDownPart_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownPart_buff_0 : memref<32x96xbf16>, 0, 3072) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%ffnDownPart_prod_lock_0, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %memtile_dma_7_1 = aie.memtile_dma(%mem_tile_7_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb9)
    ^bb1:  // 2 preds: ^bb0, ^bb8
      aie.use_lock(%ffnROut_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnROut_cons_buff_0 : memref<32x96xbf16>, 0, 3072) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%ffnROut_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%ffnROut_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnROut_cons_buff_1 : memref<32x96xbf16>, 0, 3072) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%ffnROut_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb3
    ^bb3:  // pred: ^bb2
      aie.use_lock(%ffnROut_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnROut_cons_buff_2 : memref<32x96xbf16>, 0, 3072) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%ffnROut_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%ffnROut_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnROut_cons_buff_3 : memref<32x96xbf16>, 0, 3072) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%ffnROut_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%ffnROut_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnROut_cons_buff_4 : memref<32x96xbf16>, 0, 3072) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%ffnROut_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%ffnROut_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnROut_cons_buff_5 : memref<32x96xbf16>, 0, 3072) {bd_id = 5 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%ffnROut_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%ffnROut_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnROut_cons_buff_6 : memref<32x96xbf16>, 0, 3072) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%ffnROut_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%ffnROut_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnROut_cons_buff_7 : memref<32x96xbf16>, 0, 3072) {bd_id = 7 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%ffnROut_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb9:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb10, ^bb18)
    ^bb10:  // 2 preds: ^bb9, ^bb17
      aie.use_lock(%ffnROut_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnROut_cons_buff_0 : memref<32x96xbf16>, 0, 3072) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%ffnROut_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%ffnROut_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnROut_cons_buff_1 : memref<32x96xbf16>, 0, 3072) {bd_id = 9 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%ffnROut_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb12
    ^bb12:  // pred: ^bb11
      aie.use_lock(%ffnROut_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnROut_cons_buff_2 : memref<32x96xbf16>, 0, 3072) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%ffnROut_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb13:  // pred: ^bb12
      aie.use_lock(%ffnROut_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnROut_cons_buff_3 : memref<32x96xbf16>, 0, 3072) {bd_id = 11 : i32, next_bd_id = 12 : i32}
      aie.use_lock(%ffnROut_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%ffnROut_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnROut_cons_buff_4 : memref<32x96xbf16>, 0, 3072) {bd_id = 12 : i32, next_bd_id = 13 : i32}
      aie.use_lock(%ffnROut_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb15
    ^bb15:  // pred: ^bb14
      aie.use_lock(%ffnROut_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnROut_cons_buff_5 : memref<32x96xbf16>, 0, 3072) {bd_id = 13 : i32, next_bd_id = 14 : i32}
      aie.use_lock(%ffnROut_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb16
    ^bb16:  // pred: ^bb15
      aie.use_lock(%ffnROut_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnROut_cons_buff_6 : memref<32x96xbf16>, 0, 3072) {bd_id = 14 : i32, next_bd_id = 15 : i32}
      aie.use_lock(%ffnROut_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%ffnROut_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnROut_cons_buff_7 : memref<32x96xbf16>, 0, 3072) {bd_id = 15 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%ffnROut_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb18:  // pred: ^bb9
      %2 = aie.dma_start(S2MM, 1, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%inR_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inR_cons_buff_0 : memref<32x96xbf16>, 0, 3072) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%inR_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%inR_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inR_cons_buff_1 : memref<32x96xbf16>, 0, 3072) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%inR_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      %3 = aie.dma_start(MM2S, 1, ^bb22, ^bb24)
    ^bb22:  // 2 preds: ^bb21, ^bb23
      aie.use_lock(%inR_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inR_cons_buff_0 : memref<32x96xbf16>, 0, 3072, [<size = 4, stride = 768>, <size = 12, stride = 8>, <size = 8, stride = 96>, <size = 8, stride = 1>]) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%inR_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%inR_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inR_cons_buff_1 : memref<32x96xbf16>, 0, 3072, [<size = 4, stride = 768>, <size = 12, stride = 8>, <size = 8, stride = 96>, <size = 8, stride = 1>]) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%inR_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb22
    ^bb24:  // pred: ^bb21
      %4 = aie.dma_start(MM2S, 2, ^bb25, ^bb27)
    ^bb25:  // 2 preds: ^bb24, ^bb26
      aie.use_lock(%outLN2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outLN2_cons_buff_0 : memref<32x96xbf16>, 0, 3072, [<size = 4, stride = 768>, <size = 8, stride = 8>, <size = 12, stride = 64>, <size = 8, stride = 1>]) {bd_id = 16 : i32, next_bd_id = 17 : i32}
      aie.use_lock(%outLN2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%outLN2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outLN2_cons_buff_1 : memref<32x96xbf16>, 0, 3072, [<size = 4, stride = 768>, <size = 8, stride = 8>, <size = 12, stride = 64>, <size = 8, stride = 1>]) {bd_id = 17 : i32, next_bd_id = 16 : i32}
      aie.use_lock(%outLN2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb25
    ^bb27:  // pred: ^bb24
      %5 = aie.dma_start(S2MM, 2, ^bb28, ^bb30)
    ^bb28:  // 2 preds: ^bb27, ^bb29
      aie.use_lock(%outLN2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outLN2_cons_buff_0 : memref<32x96xbf16>, 0, 3072) {bd_id = 18 : i32, next_bd_id = 19 : i32}
      aie.use_lock(%outLN2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      aie.use_lock(%outLN2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outLN2_cons_buff_1 : memref<32x96xbf16>, 0, 3072) {bd_id = 19 : i32, next_bd_id = 18 : i32}
      aie.use_lock(%outLN2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb28
    ^bb30:  // pred: ^bb27
      aie.end
    }
    %mem_7_5 = aie.mem(%tile_7_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%ffnRIn_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnRIn_cons_buff_0 : memref<32x96xbf16>, 0, 3072) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%ffnRIn_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%ffnRIn_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnRIn_cons_buff_1 : memref<32x96xbf16>, 0, 3072) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%ffnRIn_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%outLN2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outLN2_buff_0 : memref<32x96xbf16>, 0, 3072) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%outLN2_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%outLN2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outLN2_buff_1 : memref<32x96xbf16>, 0, 3072) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%outLN2_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    %mem_4_2 = aie.mem(%tile_4_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%ffnROut_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnROut_buff_0 : memref<32x96xbf16>, 0, 3072) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%ffnROut_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb5)
    ^bb3:  // 2 preds: ^bb2, ^bb4
      aie.use_lock(%memR_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memR_cons_buff_0 : memref<32x96xbf16>, 0, 3072) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%memR_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%memR_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memR_cons_buff_1 : memref<32x96xbf16>, 0, 3072) {bd_id = 2 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%memR_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb3
    ^bb5:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb8)
    ^bb6:  // 2 preds: ^bb5, ^bb7
      aie.use_lock(%ln1Norm_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ln1Norm_cons_buff_0 : memref<32x96xbf16>, 0, 3072) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%ln1Norm_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%ln1Norm_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ln1Norm_cons_buff_1 : memref<32x96xbf16>, 0, 3072) {bd_id = 4 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%ln1Norm_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb8:  // pred: ^bb5
      %3 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%outLNBroadcast_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outLNBroadcast_buff_0 : memref<32x96xbf16>, 0, 3072) {bd_id = 5 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%outLNBroadcast_prod_lock_0, Release, 1)
      aie.next_bd ^bb9
    ^bb10:  // pred: ^bb8
      aie.end
    }
    aie.shim_dma_allocation @inK_shim_alloc(%shim_noc_tile_1_0, MM2S, 0)
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb9)
    ^bb1:  // 2 preds: ^bb0, ^bb8
      aie.use_lock(%inK_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inK_cons_buff_0 : memref<64x256xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%inK_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%inK_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%inK_cons_buff_0 : memref<64x256xbf16>, 4096, 4096) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%inK_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb3
    ^bb3:  // pred: ^bb2
      aie.use_lock(%inK_cons_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%inK_cons_buff_0 : memref<64x256xbf16>, 8192, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%inK_cons_cons_lock_2, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%inK_cons_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%inK_cons_buff_0 : memref<64x256xbf16>, 12288, 4096) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%inK_cons_cons_lock_3, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%inK_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inK_cons_buff_1 : memref<64x256xbf16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%inK_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%inK_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%inK_cons_buff_1 : memref<64x256xbf16>, 4096, 4096) {bd_id = 5 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%inK_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%inK_cons_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%inK_cons_buff_1 : memref<64x256xbf16>, 8192, 4096) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%inK_cons_cons_lock_2, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%inK_cons_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%inK_cons_buff_1 : memref<64x256xbf16>, 12288, 4096) {bd_id = 7 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%inK_cons_cons_lock_3, Release, 1)
      aie.next_bd ^bb1
    ^bb9:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%inK_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inK_cons_buff_0 : memref<64x256xbf16>, 0, 4096, [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%inK_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%inK_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inK_cons_buff_1 : memref<64x256xbf16>, 0, 4096, [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {bd_id = 9 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%inK_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %2 = aie.dma_start(MM2S, 1, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%inK_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%inK_cons_buff_0 : memref<64x256xbf16>, 4096, 4096, [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%inK_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%inK_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%inK_cons_buff_1 : memref<64x256xbf16>, 4096, 4096, [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%inK_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %3 = aie.dma_start(MM2S, 2, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%inK_cons_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%inK_cons_buff_0 : memref<64x256xbf16>, 8192, 4096, [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%inK_cons_prod_lock_2, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%inK_cons_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%inK_cons_buff_1 : memref<64x256xbf16>, 8192, 4096, [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {bd_id = 11 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%inK_cons_prod_lock_2, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %4 = aie.dma_start(MM2S, 3, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%inK_cons_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%inK_cons_buff_0 : memref<64x256xbf16>, 12288, 4096, [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%inK_cons_prod_lock_3, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%inK_cons_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%inK_cons_buff_1 : memref<64x256xbf16>, 12288, 4096, [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%inK_cons_prod_lock_3, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%memK0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memK0_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%memK0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%memK0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memK0_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%memK0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%memQ0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memQ0_cons_buff_0 : memref<32x64xbf16>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%memQ0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%memQ0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memQ0_cons_buff_1 : memref<32x64xbf16>, 0, 2048) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%memQ0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%memA0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA0_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%memA0_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%memA0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA0_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%memA0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%memK1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memK1_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%memK1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%memK1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memK1_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%memK1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%memQ1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memQ1_cons_buff_0 : memref<32x64xbf16>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%memQ1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%memQ1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memQ1_cons_buff_1 : memref<32x64xbf16>, 0, 2048) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%memQ1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%memA1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA1_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%memA1_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%memA1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA1_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%memA1_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%memK2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memK2_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%memK2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%memK2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memK2_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%memK2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%memQ2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memQ2_cons_buff_0 : memref<32x64xbf16>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%memQ2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%memQ2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memQ2_cons_buff_1 : memref<32x64xbf16>, 0, 2048) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%memQ2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%memA2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA2_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%memA2_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%memA2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA2_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%memA2_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%memK3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memK3_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%memK3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%memK3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memK3_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%memK3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%memQ3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memQ3_cons_buff_0 : memref<32x64xbf16>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%memQ3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%memQ3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memQ3_cons_buff_1 : memref<32x64xbf16>, 0, 2048) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%memQ3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%memA3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA3_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%memA3_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%memA3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA3_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%memA3_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @inOW_shim_alloc(%shim_noc_tile_3_0, MM2S, 0)
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb9)
    ^bb1:  // 2 preds: ^bb0, ^bb8
      aie.use_lock(%inOW_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inOW_cons_buff_0 : memref<256x96xbf16>, 0, 6144) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%inOW_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%inOW_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%inOW_cons_buff_0 : memref<256x96xbf16>, 6144, 6144) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%inOW_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb3
    ^bb3:  // pred: ^bb2
      aie.use_lock(%inOW_cons_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%inOW_cons_buff_0 : memref<256x96xbf16>, 12288, 6144) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%inOW_cons_cons_lock_2, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%inOW_cons_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%inOW_cons_buff_0 : memref<256x96xbf16>, 18432, 6144) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%inOW_cons_cons_lock_3, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%inOW_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inOW_cons_buff_1 : memref<256x96xbf16>, 0, 6144) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%inOW_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%inOW_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%inOW_cons_buff_1 : memref<256x96xbf16>, 6144, 6144) {bd_id = 5 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%inOW_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%inOW_cons_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%inOW_cons_buff_1 : memref<256x96xbf16>, 12288, 6144) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%inOW_cons_cons_lock_2, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%inOW_cons_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%inOW_cons_buff_1 : memref<256x96xbf16>, 18432, 6144) {bd_id = 7 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%inOW_cons_cons_lock_3, Release, 1)
      aie.next_bd ^bb1
    ^bb9:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%inOW_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inOW_cons_buff_0 : memref<256x96xbf16>, 0, 6144, [<size = 8, stride = 768>, <size = 12, stride = 8>, <size = 8, stride = 96>, <size = 8, stride = 1>]) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%inOW_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%inOW_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inOW_cons_buff_1 : memref<256x96xbf16>, 0, 6144, [<size = 8, stride = 768>, <size = 12, stride = 8>, <size = 8, stride = 96>, <size = 8, stride = 1>]) {bd_id = 9 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%inOW_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %2 = aie.dma_start(MM2S, 1, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%inOW_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%inOW_cons_buff_0 : memref<256x96xbf16>, 6144, 6144, [<size = 8, stride = 768>, <size = 12, stride = 8>, <size = 8, stride = 96>, <size = 8, stride = 1>]) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%inOW_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%inOW_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%inOW_cons_buff_1 : memref<256x96xbf16>, 6144, 6144, [<size = 8, stride = 768>, <size = 12, stride = 8>, <size = 8, stride = 96>, <size = 8, stride = 1>]) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%inOW_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %3 = aie.dma_start(MM2S, 2, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%inOW_cons_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%inOW_cons_buff_0 : memref<256x96xbf16>, 12288, 6144, [<size = 8, stride = 768>, <size = 12, stride = 8>, <size = 8, stride = 96>, <size = 8, stride = 1>]) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%inOW_cons_prod_lock_2, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%inOW_cons_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%inOW_cons_buff_1 : memref<256x96xbf16>, 12288, 6144, [<size = 8, stride = 768>, <size = 12, stride = 8>, <size = 8, stride = 96>, <size = 8, stride = 1>]) {bd_id = 11 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%inOW_cons_prod_lock_2, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %4 = aie.dma_start(MM2S, 3, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%inOW_cons_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%inOW_cons_buff_0 : memref<256x96xbf16>, 18432, 6144, [<size = 8, stride = 768>, <size = 12, stride = 8>, <size = 8, stride = 96>, <size = 8, stride = 1>]) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%inOW_cons_prod_lock_3, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%inOW_cons_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%inOW_cons_buff_1 : memref<256x96xbf16>, 18432, 6144, [<size = 8, stride = 768>, <size = 12, stride = 8>, <size = 8, stride = 96>, <size = 8, stride = 1>]) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%inOW_cons_prod_lock_3, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      aie.end
    }
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%memOW0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memOW0_cons_buff_0 : memref<64x96xbf16>, 0, 6144) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%memOW0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%memOW0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memOW0_cons_buff_1 : memref<64x96xbf16>, 0, 6144) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%memOW0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      aie.end
    }
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%memOW1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memOW1_cons_buff_0 : memref<64x96xbf16>, 0, 6144) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%memOW1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%memOW1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memOW1_cons_buff_1 : memref<64x96xbf16>, 0, 6144) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%memOW1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      aie.end
    }
    %mem_2_5 = aie.mem(%tile_2_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%memOW2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memOW2_cons_buff_0 : memref<64x96xbf16>, 0, 6144) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%memOW2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%memOW2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memOW2_cons_buff_1 : memref<64x96xbf16>, 0, 6144) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%memOW2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      aie.end
    }
    %mem_3_5 = aie.mem(%tile_3_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%memOW3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memOW3_cons_buff_0 : memref<64x96xbf16>, 0, 6144) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%memOW3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%memOW3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memOW3_cons_buff_1 : memref<64x96xbf16>, 0, 6144) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%memOW3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%outOProjAccumIn3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOProjAccumIn3_cons_buff_0 : memref<32x96xbf16>, 0, 3072) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%outOProjAccumIn3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%outOProjAccumOut3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOProjAccumOut3_buff_0 : memref<32x96xbf16>, 0, 3072) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%outOProjAccumOut3_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(MM2S, 1, ^bb8, ^bb10)
    ^bb8:  // 2 preds: ^bb7, ^bb9
      aie.use_lock(%outOProjInput_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOProjInput_buff_0 : memref<32x96xbf16>, 0, 3072) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%outOProjInput_prod_lock_0, Release, 1)
      aie.next_bd ^bb9
    ^bb9:  // pred: ^bb8
      aie.use_lock(%outOProjInput_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOProjInput_buff_1 : memref<32x96xbf16>, 0, 3072) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%outOProjInput_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb10:  // pred: ^bb7
      aie.end
    }
    aie.shim_dma_allocation @inQ_shim_alloc(%shim_noc_tile_0_0, MM2S, 0)
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb9)
    ^bb1:  // 2 preds: ^bb0, ^bb8
      aie.use_lock(%inQ_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inQ_cons_buff_0 : memref<32x256xbf16>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%inQ_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%inQ_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%inQ_cons_buff_0 : memref<32x256xbf16>, 2048, 2048) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%inQ_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb3
    ^bb3:  // pred: ^bb2
      aie.use_lock(%inQ_cons_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%inQ_cons_buff_0 : memref<32x256xbf16>, 4096, 2048) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%inQ_cons_cons_lock_2, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%inQ_cons_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%inQ_cons_buff_0 : memref<32x256xbf16>, 6144, 2048) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%inQ_cons_cons_lock_3, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%inQ_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inQ_cons_buff_1 : memref<32x256xbf16>, 0, 2048) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%inQ_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%inQ_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%inQ_cons_buff_1 : memref<32x256xbf16>, 2048, 2048) {bd_id = 5 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%inQ_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%inQ_cons_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%inQ_cons_buff_1 : memref<32x256xbf16>, 4096, 2048) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%inQ_cons_cons_lock_2, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%inQ_cons_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%inQ_cons_buff_1 : memref<32x256xbf16>, 6144, 2048) {bd_id = 7 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%inQ_cons_cons_lock_3, Release, 1)
      aie.next_bd ^bb1
    ^bb9:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%inQ_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inQ_cons_buff_0 : memref<32x256xbf16>, 0, 2048, [<size = 4, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%inQ_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%inQ_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inQ_cons_buff_1 : memref<32x256xbf16>, 0, 2048, [<size = 4, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {bd_id = 9 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%inQ_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %2 = aie.dma_start(MM2S, 1, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%inQ_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%inQ_cons_buff_0 : memref<32x256xbf16>, 2048, 2048, [<size = 4, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%inQ_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%inQ_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%inQ_cons_buff_1 : memref<32x256xbf16>, 2048, 2048, [<size = 4, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%inQ_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %3 = aie.dma_start(MM2S, 2, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%inQ_cons_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%inQ_cons_buff_0 : memref<32x256xbf16>, 4096, 2048, [<size = 4, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%inQ_cons_prod_lock_2, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%inQ_cons_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%inQ_cons_buff_1 : memref<32x256xbf16>, 4096, 2048, [<size = 4, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {bd_id = 11 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%inQ_cons_prod_lock_2, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %4 = aie.dma_start(MM2S, 3, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%inQ_cons_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%inQ_cons_buff_0 : memref<32x256xbf16>, 6144, 2048, [<size = 4, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%inQ_cons_prod_lock_3, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%inQ_cons_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%inQ_cons_buff_1 : memref<32x256xbf16>, 6144, 2048, [<size = 4, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%inQ_cons_prod_lock_3, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      aie.end
    }
    aie.shim_dma_allocation @inR_shim_alloc(%shim_noc_tile_7_0, MM2S, 0)
    aie.shim_dma_allocation @inV_shim_alloc(%shim_noc_tile_2_0, MM2S, 0)
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb9)
    ^bb1:  // 2 preds: ^bb0, ^bb8
      aie.use_lock(%inV_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inV_cons_buff_0 : memref<64x256xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%inV_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%inV_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%inV_cons_buff_0 : memref<64x256xbf16>, 4096, 4096) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%inV_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb3
    ^bb3:  // pred: ^bb2
      aie.use_lock(%inV_cons_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%inV_cons_buff_0 : memref<64x256xbf16>, 8192, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%inV_cons_cons_lock_2, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%inV_cons_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%inV_cons_buff_0 : memref<64x256xbf16>, 12288, 4096) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%inV_cons_cons_lock_3, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%inV_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inV_cons_buff_1 : memref<64x256xbf16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%inV_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%inV_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%inV_cons_buff_1 : memref<64x256xbf16>, 4096, 4096) {bd_id = 5 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%inV_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%inV_cons_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%inV_cons_buff_1 : memref<64x256xbf16>, 8192, 4096) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%inV_cons_cons_lock_2, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%inV_cons_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%inV_cons_buff_1 : memref<64x256xbf16>, 12288, 4096) {bd_id = 7 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%inV_cons_cons_lock_3, Release, 1)
      aie.next_bd ^bb1
    ^bb9:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%inV_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inV_cons_buff_0 : memref<64x256xbf16>, 0, 4096, [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%inV_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%inV_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inV_cons_buff_1 : memref<64x256xbf16>, 0, 4096, [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {bd_id = 9 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%inV_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %2 = aie.dma_start(MM2S, 1, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%inV_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%inV_cons_buff_0 : memref<64x256xbf16>, 4096, 4096, [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%inV_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%inV_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%inV_cons_buff_1 : memref<64x256xbf16>, 4096, 4096, [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%inV_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %3 = aie.dma_start(MM2S, 2, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%inV_cons_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%inV_cons_buff_0 : memref<64x256xbf16>, 8192, 4096, [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%inV_cons_prod_lock_2, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%inV_cons_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%inV_cons_buff_1 : memref<64x256xbf16>, 8192, 4096, [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {bd_id = 11 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%inV_cons_prod_lock_2, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %4 = aie.dma_start(MM2S, 3, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%inV_cons_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%inV_cons_buff_0 : memref<64x256xbf16>, 12288, 4096, [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%inV_cons_prod_lock_3, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%inV_cons_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%inV_cons_buff_1 : memref<64x256xbf16>, 12288, 4096, [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%inV_cons_prod_lock_3, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      aie.end
    }
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%memV0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memV0_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%memV0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%memV0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memV0_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%memV0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%memP0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memP0_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%memP0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%memP0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memP0_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%memP0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%memV1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memV1_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%memV1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%memV1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memV1_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%memV1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%memP1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memP1_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%memP1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%memP1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memP1_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%memP1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    %mem_2_4 = aie.mem(%tile_2_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%memV2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memV2_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%memV2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%memV2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memV2_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%memV2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%memP2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memP2_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%memP2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%memP2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memP2_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%memP2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    %mem_3_4 = aie.mem(%tile_3_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%memV3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memV3_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%memV3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%memV3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memV3_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%memV3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%memP3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memP3_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%memP3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%memP3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memP3_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%memP3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%memA0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA0_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%memA0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%memA0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA0_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%memA0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%memP0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memP0_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%memP0_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%memP0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memP0_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%memP0_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%memA1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA1_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%memA1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%memA1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA1_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%memA1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%memP1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memP1_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%memP1_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%memP1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memP1_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%memP1_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    %mem_2_3 = aie.mem(%tile_2_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%memA2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA2_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%memA2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%memA2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA2_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%memA2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%memP2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memP2_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%memP2_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%memP2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memP2_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%memP2_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    %mem_3_3 = aie.mem(%tile_3_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%memA3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA3_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%memA3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%memA3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA3_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%memA3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%memP3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memP3_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%memP3_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%memP3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memP3_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%memP3_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @memLN2_shim_alloc(%shim_noc_tile_7_0, S2MM, 0)
    %mem_6_4 = aie.mem(%tile_6_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%outLNBroadcast_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outLNBroadcast_cons_buff_0 : memref<32x96xbf16>, 0, 3072) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%outLNBroadcast_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%outLNBroadcast_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outLNBroadcast_cons_buff_1 : memref<32x96xbf16>, 0, 3072) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%outLNBroadcast_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      aie.end
    }
    %memtile_dma_4_1 = aie.memtile_dma(%mem_tile_4_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb9)
    ^bb1:  // 2 preds: ^bb0, ^bb8
      aie.use_lock(%outOProjAccumOut3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOProjAccumOut3_cons_buff_0 : memref<32x96xbf16>, 0, 3072) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%outOProjAccumOut3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%outOProjAccumOut3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOProjAccumOut3_cons_buff_1 : memref<32x96xbf16>, 0, 3072) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%outOProjAccumOut3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb3
    ^bb3:  // pred: ^bb2
      aie.use_lock(%outOProjAccumOut3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOProjAccumOut3_cons_buff_2 : memref<32x96xbf16>, 0, 3072) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%outOProjAccumOut3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%outOProjAccumOut3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOProjAccumOut3_cons_buff_3 : memref<32x96xbf16>, 0, 3072) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%outOProjAccumOut3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%outOProjAccumOut3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOProjAccumOut3_cons_buff_4 : memref<32x96xbf16>, 0, 3072) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%outOProjAccumOut3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%outOProjAccumOut3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOProjAccumOut3_cons_buff_5 : memref<32x96xbf16>, 0, 3072) {bd_id = 5 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%outOProjAccumOut3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%outOProjAccumOut3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOProjAccumOut3_cons_buff_6 : memref<32x96xbf16>, 0, 3072) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%outOProjAccumOut3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%outOProjAccumOut3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOProjAccumOut3_cons_buff_7 : memref<32x96xbf16>, 0, 3072) {bd_id = 7 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%outOProjAccumOut3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb9:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb10, ^bb18)
    ^bb10:  // 2 preds: ^bb9, ^bb17
      aie.use_lock(%outOProjAccumOut3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOProjAccumOut3_cons_buff_0 : memref<32x96xbf16>, 0, 3072) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%outOProjAccumOut3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%outOProjAccumOut3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOProjAccumOut3_cons_buff_1 : memref<32x96xbf16>, 0, 3072) {bd_id = 9 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%outOProjAccumOut3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb12
    ^bb12:  // pred: ^bb11
      aie.use_lock(%outOProjAccumOut3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOProjAccumOut3_cons_buff_2 : memref<32x96xbf16>, 0, 3072) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%outOProjAccumOut3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb13:  // pred: ^bb12
      aie.use_lock(%outOProjAccumOut3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOProjAccumOut3_cons_buff_3 : memref<32x96xbf16>, 0, 3072) {bd_id = 11 : i32, next_bd_id = 12 : i32}
      aie.use_lock(%outOProjAccumOut3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%outOProjAccumOut3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOProjAccumOut3_cons_buff_4 : memref<32x96xbf16>, 0, 3072) {bd_id = 12 : i32, next_bd_id = 13 : i32}
      aie.use_lock(%outOProjAccumOut3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb15
    ^bb15:  // pred: ^bb14
      aie.use_lock(%outOProjAccumOut3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOProjAccumOut3_cons_buff_5 : memref<32x96xbf16>, 0, 3072) {bd_id = 13 : i32, next_bd_id = 14 : i32}
      aie.use_lock(%outOProjAccumOut3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb16
    ^bb16:  // pred: ^bb15
      aie.use_lock(%outOProjAccumOut3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOProjAccumOut3_cons_buff_6 : memref<32x96xbf16>, 0, 3072) {bd_id = 14 : i32, next_bd_id = 15 : i32}
      aie.use_lock(%outOProjAccumOut3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%outOProjAccumOut3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOProjAccumOut3_cons_buff_7 : memref<32x96xbf16>, 0, 3072) {bd_id = 15 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%outOProjAccumOut3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb18:  // pred: ^bb9
      aie.end
    }
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_1_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_1_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_2_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_2_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_3_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_3_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_7_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_7_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}
