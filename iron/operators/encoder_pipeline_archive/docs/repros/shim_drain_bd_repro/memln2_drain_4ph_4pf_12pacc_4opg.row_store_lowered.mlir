module {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_1_2 = aie.tile(1, 2)
    %tile_1_3 = aie.tile(1, 3)
    %tile_1_4 = aie.tile(1, 4)
    %tile_1_5 = aie.tile(1, 5)
    %tile_2_2 = aie.tile(2, 2)
    %tile_2_3 = aie.tile(2, 3)
    %tile_2_4 = aie.tile(2, 4)
    %tile_2_5 = aie.tile(2, 5)
    %tile_3_2 = aie.tile(3, 2)
    %tile_3_3 = aie.tile(3, 3)
    %tile_3_4 = aie.tile(3, 4)
    %tile_3_5 = aie.tile(3, 5)
    %tile_6_5 = aie.tile(6, 5)
    %tile_4_2 = aie.tile(4, 2)
    %tile_4_3 = aie.tile(4, 3)
    %tile_5_3 = aie.tile(5, 3)
    %tile_6_2 = aie.tile(6, 2)
    %tile_6_3 = aie.tile(6, 3)
    %tile_7_2 = aie.tile(7, 2)
    %tile_7_3 = aie.tile(7, 3)
    %tile_6_4 = aie.tile(6, 4)
    %tile_7_4 = aie.tile(7, 4)
    %tile_7_5 = aie.tile(7, 5)
    %mem_tile_7_1 = aie.tile(7, 1)
    %shim_noc_tile_5_0 = aie.tile(5, 0)
    %mem_tile_5_1 = aie.tile(5, 1)
    %shim_noc_tile_4_0 = aie.tile(4, 0)
    %mem_tile_4_1 = aie.tile(4, 1)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %mem_tile_3_1 = aie.tile(3, 1)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %mem_tile_2_1 = aie.tile(2, 1)
    %shim_noc_tile_6_0 = aie.tile(6, 0)
    %mem_tile_6_1 = aie.tile(6, 1)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %mem_tile_1_1 = aie.tile(1, 1)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %shim_noc_tile_7_0 = aie.tile(7, 0)
    aie.objectfifo @ffnDownGroup0(%tile_5_3, {%tile_6_3}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @ffnDownGroup2(%tile_7_3, {%tile_7_4}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @ffnDownOut0(%tile_6_3, {%tile_7_5}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @ffnDownOut1(%tile_7_4, {%tile_7_5}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @ffnRIn(%mem_tile_7_1, {%tile_7_5}, 12 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @ffnROut(%tile_4_2, {%mem_tile_7_1}, [1 : i32, 12 : i32]) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo.link [@ffnROut] -> [@ffnRIn]([] [0])
    aie.objectfifo @ffnUpOut(%tile_4_3, {%tile_5_3}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @ffnUpOut1(%tile_6_2, {%tile_6_3}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @ffnUpOut2(%tile_7_2, {%tile_7_3}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @ffnUpOut3(%tile_6_4, {%tile_7_4}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @inBDown(%shim_noc_tile_5_0, {%mem_tile_5_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @memBDown(%mem_tile_5_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_5_3}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo.link [@inBDown] -> [@memBDown]([] [0])
    aie.objectfifo @inBDown1(%shim_noc_tile_4_0, {%mem_tile_4_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @memBDown1(%mem_tile_4_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_6_3}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo.link [@inBDown1] -> [@memBDown1]([] [0])
    aie.objectfifo @inBDown2(%shim_noc_tile_3_0, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @memBDown2(%mem_tile_3_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_7_3}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo.link [@inBDown2] -> [@memBDown2]([] [0])
    aie.objectfifo @inBDown3(%shim_noc_tile_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @memBDown3(%mem_tile_2_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_7_4}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo.link [@inBDown3] -> [@memBDown3]([] [0])
    aie.objectfifo @inBUp(%shim_noc_tile_6_0, {%mem_tile_6_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @memBUp(%mem_tile_6_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_4_3}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo.link [@inBUp] -> [@memBUp]([] [0])
    aie.objectfifo @inBUp1(%shim_noc_tile_5_0, {%mem_tile_5_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @memBUp1(%mem_tile_5_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_6_2}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo.link [@inBUp1] -> [@memBUp1]([] [0])
    aie.objectfifo @inBUp2(%shim_noc_tile_4_0, {%mem_tile_4_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @memBUp2(%mem_tile_4_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_7_2}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo.link [@inBUp2] -> [@memBUp2]([] [0])
    aie.objectfifo @inBUp3(%shim_noc_tile_6_0, {%mem_tile_6_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @memBUp3(%mem_tile_6_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_6_4}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo.link [@inBUp3] -> [@memBUp3]([] [0])
    aie.objectfifo @inK(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<64x256xbf16>> 
    aie.objectfifo @memK0(%mem_tile_1_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @memK1(%mem_tile_1_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @memK2(%mem_tile_1_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_2_2}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @memK3(%mem_tile_1_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_3_2}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo.link [@inK] -> [@memK0, @memK1, @memK2, @memK3]([] [0, 4096, 8192, 12288])
    aie.objectfifo @inOW(%shim_noc_tile_3_0, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<256x64xbf16>> 
    aie.objectfifo @memOW0(%mem_tile_3_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @memOW1(%mem_tile_3_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_1_5}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @memOW2(%mem_tile_3_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_2_5}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @memOW3(%mem_tile_3_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_3_5}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo.link [@inOW] -> [@memOW0, @memOW1, @memOW2, @memOW3]([] [0, 4096, 8192, 12288])
    aie.objectfifo @inQ(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32x256xbf16>> 
    aie.objectfifo @memQ0(%mem_tile_0_1 dimensionsToStream [<size = 4, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @memQ1(%mem_tile_0_1 dimensionsToStream [<size = 4, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @memQ2(%mem_tile_0_1 dimensionsToStream [<size = 4, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_2_2}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @memQ3(%mem_tile_0_1 dimensionsToStream [<size = 4, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_3_2}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo.link [@inQ] -> [@memQ0, @memQ1, @memQ2, @memQ3]([] [0, 2048, 4096, 6144])
    aie.objectfifo @inR(%shim_noc_tile_7_0, {%mem_tile_7_1}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @memR(%mem_tile_7_1 dimensionsToStream [<size = 4, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_4_2}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo.link [@inR] -> [@memR]([] [0])
    aie.objectfifo @inV(%shim_noc_tile_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<64x256xbf16>> 
    aie.objectfifo @memV0(%mem_tile_2_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @memV1(%mem_tile_2_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_1_4}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @memV2(%mem_tile_2_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_2_4}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @memV3(%mem_tile_2_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_3_4}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo.link [@inV] -> [@memV0, @memV1, @memV2, @memV3]([] [0, 4096, 8192, 12288])
    aie.objectfifo @ln1Norm(%tile_6_5, {%tile_4_2}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @memA0(%tile_0_2 dimensionsToStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], {%tile_0_3 dimensionsFromStream [<size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @memA1(%tile_1_2 dimensionsToStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], {%tile_1_3 dimensionsFromStream [<size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @memA2(%tile_2_2 dimensionsToStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], {%tile_2_3 dimensionsFromStream [<size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @memA3(%tile_3_2 dimensionsToStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], {%tile_3_3 dimensionsFromStream [<size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @memLN2(%mem_tile_7_1 dimensionsToStream [<size = 4, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_7_0}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @outLN2(%tile_7_5, {%mem_tile_7_1}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo.link [@outLN2] -> [@memLN2]([] [0])
    aie.objectfifo @memP0(%tile_0_3 dimensionsToStream [<size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>], {%tile_0_4 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @memP1(%tile_1_3 dimensionsToStream [<size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>], {%tile_1_4 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @memP2(%tile_2_3 dimensionsToStream [<size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>], {%tile_2_4 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @memP3(%tile_3_3 dimensionsToStream [<size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>], {%tile_3_4 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @outLNBroadcast(%tile_4_2, {%tile_4_3, %tile_6_2, %tile_7_2, %tile_6_4}, [1 : i32, 2 : i32, 2 : i32, 2 : i32, 2 : i32]) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @outOGroupPart0(%tile_0_5, {%tile_1_5}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @outOGroupPart1(%tile_1_5, {%tile_2_5}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @outOGroupPart2(%tile_2_5, {%tile_3_5}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @outOProj0(%tile_0_4, {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @outOProj1(%tile_1_4, {%tile_1_5}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @outOProj2(%tile_2_4, {%tile_2_5}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @outOProj3(%tile_3_4, {%tile_3_5}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @outOProjInput(%tile_3_5, {%tile_6_5}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @scaleOF0(%tile_0_3, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<128xbf16>> 
    aie.objectfifo @scaleOF1(%tile_1_3, {%tile_1_4}, 2 : i32) : !aie.objectfifo<memref<128xbf16>> 
    aie.objectfifo @scaleOF2(%tile_2_3, {%tile_2_4}, 2 : i32) : !aie.objectfifo<memref<128xbf16>> 
    aie.objectfifo @scaleOF3(%tile_3_3, {%tile_3_4}, 2 : i32) : !aie.objectfifo<memref<128xbf16>> 
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
    %idx_buffer_qk_1 = aie.buffer(%tile_1_2) {sym_name = "idx_buffer_qk_1"} : memref<2xi32> = dense<0>
    %idx_buffer_softmax_1 = aie.buffer(%tile_1_3) {sym_name = "idx_buffer_softmax_1"} : memref<2xi32> = dense<0>
    %scale_buffer_softmax_1 = aie.buffer(%tile_1_3) {sym_name = "scale_buffer_softmax_1"} : memref<128xbf16> = dense<0.000000e+00>
    %idx_buffer_pv_1 = aie.buffer(%tile_1_4) {sym_name = "idx_buffer_pv_1"} : memref<2xi32> = dense<0>
    %o_proj_partial_scratch_1 = aie.buffer(%tile_1_5) {sym_name = "o_proj_partial_scratch_1"} : memref<32x64xbf16> 
    %o_proj_zero_scratch_1 = aie.buffer(%tile_1_5) {sym_name = "o_proj_zero_scratch_1"} : memref<32x64xbf16> 
    %o_proj_stats_sum_1 = aie.buffer(%tile_1_5) {sym_name = "o_proj_stats_sum_1"} : memref<32xf32> 
    %o_proj_stats_sumsq_1 = aie.buffer(%tile_1_5) {sym_name = "o_proj_stats_sumsq_1"} : memref<32xf32> 
    %idx_buffer_qk_2 = aie.buffer(%tile_2_2) {sym_name = "idx_buffer_qk_2"} : memref<2xi32> = dense<0>
    %idx_buffer_softmax_2 = aie.buffer(%tile_2_3) {sym_name = "idx_buffer_softmax_2"} : memref<2xi32> = dense<0>
    %scale_buffer_softmax_2 = aie.buffer(%tile_2_3) {sym_name = "scale_buffer_softmax_2"} : memref<128xbf16> = dense<0.000000e+00>
    %idx_buffer_pv_2 = aie.buffer(%tile_2_4) {sym_name = "idx_buffer_pv_2"} : memref<2xi32> = dense<0>
    %o_proj_partial_scratch_2 = aie.buffer(%tile_2_5) {sym_name = "o_proj_partial_scratch_2"} : memref<32x64xbf16> 
    %o_proj_zero_scratch_2 = aie.buffer(%tile_2_5) {sym_name = "o_proj_zero_scratch_2"} : memref<32x64xbf16> 
    %o_proj_stats_sum_2 = aie.buffer(%tile_2_5) {sym_name = "o_proj_stats_sum_2"} : memref<32xf32> 
    %o_proj_stats_sumsq_2 = aie.buffer(%tile_2_5) {sym_name = "o_proj_stats_sumsq_2"} : memref<32xf32> 
    %idx_buffer_qk_3 = aie.buffer(%tile_3_2) {sym_name = "idx_buffer_qk_3"} : memref<2xi32> = dense<0>
    %idx_buffer_softmax_3 = aie.buffer(%tile_3_3) {sym_name = "idx_buffer_softmax_3"} : memref<2xi32> = dense<0>
    %scale_buffer_softmax_3 = aie.buffer(%tile_3_3) {sym_name = "scale_buffer_softmax_3"} : memref<128xbf16> = dense<0.000000e+00>
    %idx_buffer_pv_3 = aie.buffer(%tile_3_4) {sym_name = "idx_buffer_pv_3"} : memref<2xi32> = dense<0>
    %mem_tile_0_1_0 = aie.tile(0, 1)
    %o_proj_partial_scratch_3 = aie.buffer(%tile_3_5) {sym_name = "o_proj_partial_scratch_3"} : memref<32x64xbf16> 
    %o_proj_zero_scratch_3 = aie.buffer(%tile_3_5) {sym_name = "o_proj_zero_scratch_3"} : memref<32x64xbf16> 
    %o_proj_stats_sum_3 = aie.buffer(%tile_3_5) {sym_name = "o_proj_stats_sum_3"} : memref<32xf32> 
    %o_proj_stats_sumsq_3 = aie.buffer(%tile_3_5) {sym_name = "o_proj_stats_sumsq_3"} : memref<32xf32> 
    %mem_tile_3_1_1 = aie.tile(3, 1)
    %ln1_norm_sum_buffer = aie.buffer(%tile_6_5) {sym_name = "ln1_norm_sum_buffer"} : memref<32xf32> 
    %ln1_norm_sumsq_buffer = aie.buffer(%tile_6_5) {sym_name = "ln1_norm_sumsq_buffer"} : memref<32xf32> 
    func.func private @fused_layer_norm_1outs(memref<32x64xbf16>, memref<32xf32>, memref<32xf32>, memref<32x64xbf16>, i32)
    func.func private @unpack_stats_bf16_packet_to_f32(memref<32x64xbf16>, memref<32xf32>, memref<32xf32>, i32)
    %static_ln1_weights = aie.buffer(%tile_4_2) {sym_name = "static_ln1_weights"} : memref<768xbf16> = dense<1.000000e+00>
    func.func private @ln_mul_add_1outs(memref<32x64xbf16>, memref<32x64xbf16>, memref<768xbf16>, memref<32x64xbf16>, i32)
    func.func private @ffn_zero_bf16_up_proj(memref<32x64xbf16>)
    func.func private @ffn_matmul_init_bf16_bf16_up_proj(memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>)
    func.func private @ffn_matmul_bf16_bf16_up_proj(memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>)
    func.func private @ffn_gelu_bf16(memref<32x64xbf16>, memref<32x64xbf16>, i32)
    func.func private @ffn_zero_bf16_down_proj(memref<32x64xbf16>)
    func.func private @ffn_matmul_init_bf16_bf16_down_proj(memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>)
    func.func private @ffn_matmul_with_acc_bf16_bf16_down_proj(memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>)
    %mem_tile_5_1_2 = aie.tile(5, 1)
    %mem_tile_7_1_3 = aie.tile(7, 1)
    %ln2_ffn_merge_buffer = aie.buffer(%tile_7_5) {sym_name = "ln2_ffn_merge_buffer"} : memref<32x64xbf16> 
    %ln2_sum_buffer = aie.buffer(%tile_7_5) {sym_name = "ln2_sum_buffer"} : memref<32xf32> 
    %ln2_sumsq_buffer = aie.buffer(%tile_7_5) {sym_name = "ln2_sumsq_buffer"} : memref<32xf32> 
    %static_ln2_weights = aie.buffer(%tile_7_5) {sym_name = "static_ln2_weights"} : memref<768xbf16> = dense<1.000000e+00>
    func.func private @fused_add_layer_norm_1outs(memref<32x64xbf16>, memref<32x64xbf16>, memref<768xbf16>, memref<32xf32>, memref<32xf32>, memref<32x64xbf16>, i32, i32)
    %outOProjAccum3_src = aie.buffer(%tile_3_5) {sym_name = "outOProjAccum3_src"} : memref<32x64xbf16> 
    %outOProjAccum3_dst = aie.buffer(%tile_3_5) {sym_name = "outOProjAccum3_dst"} : memref<32x64xbf16> 
    %outOProjAccum3_src_empty = aie.lock(%tile_3_5) {init = 1 : i32, sym_name = "outOProjAccum3_src_empty"}
    %outOProjAccum3_src_full = aie.lock(%tile_3_5) {init = 0 : i32, sym_name = "outOProjAccum3_src_full"}
    %outOProjAccum3_dst_empty = aie.lock(%tile_3_5) {init = 1 : i32, sym_name = "outOProjAccum3_dst_empty"}
    %outOProjAccum3_dst_full = aie.lock(%tile_3_5) {init = 0 : i32, sym_name = "outOProjAccum3_dst_full"}
    %outOProjAccum3_row_0 = aie.buffer(%mem_tile_0_1_0) {sym_name = "outOProjAccum3_row_0"} : memref<24576xbf16> 
    %outOProjAccum3_row_0_empty = aie.lock(%mem_tile_0_1_0) {init = 1 : i32, sym_name = "outOProjAccum3_row_0_empty"}
    %outOProjAccum3_row_0_full = aie.lock(%mem_tile_0_1_0) {init = 0 : i32, sym_name = "outOProjAccum3_row_0_full"}
    %outOProjAccum3_row_1 = aie.buffer(%mem_tile_0_1_0) {sym_name = "outOProjAccum3_row_1"} : memref<24576xbf16> 
    %outOProjAccum3_row_1_empty = aie.lock(%mem_tile_0_1_0) {init = 1 : i32, sym_name = "outOProjAccum3_row_1_empty"}
    %outOProjAccum3_row_1_full = aie.lock(%mem_tile_0_1_0) {init = 0 : i32, sym_name = "outOProjAccum3_row_1_full"}
    aie.flow(%tile_3_5, DMA : 0, %mem_tile_0_1_0, DMA : 5)
    aie.flow(%mem_tile_0_1_0, DMA : 5, %tile_3_5, DMA : 1)
    %ln1Replay_src = aie.buffer(%tile_6_5) {sym_name = "ln1Replay_src"} : memref<32x64xbf16> 
    %ln1Replay_dst = aie.buffer(%tile_6_5) {sym_name = "ln1Replay_dst"} : memref<32x64xbf16> 
    %ln1Replay_src_empty = aie.lock(%tile_6_5) {init = 1 : i32, sym_name = "ln1Replay_src_empty"}
    %ln1Replay_src_full = aie.lock(%tile_6_5) {init = 0 : i32, sym_name = "ln1Replay_src_full"}
    %ln1Replay_dst_empty = aie.lock(%tile_6_5) {init = 1 : i32, sym_name = "ln1Replay_dst_empty"}
    %ln1Replay_dst_full = aie.lock(%tile_6_5) {init = 0 : i32, sym_name = "ln1Replay_dst_full"}
    %ln1Replay_row_0 = aie.buffer(%mem_tile_3_1_1) {sym_name = "ln1Replay_row_0"} : memref<24576xbf16> 
    %ln1Replay_row_0_empty = aie.lock(%mem_tile_3_1_1) {init = 1 : i32, sym_name = "ln1Replay_row_0_empty"}
    %ln1Replay_row_0_full = aie.lock(%mem_tile_3_1_1) {init = 0 : i32, sym_name = "ln1Replay_row_0_full"}
    %ln1Replay_row_1 = aie.buffer(%mem_tile_3_1_1) {sym_name = "ln1Replay_row_1"} : memref<24576xbf16> 
    %ln1Replay_row_1_empty = aie.lock(%mem_tile_3_1_1) {init = 1 : i32, sym_name = "ln1Replay_row_1_empty"}
    %ln1Replay_row_1_full = aie.lock(%mem_tile_3_1_1) {init = 0 : i32, sym_name = "ln1Replay_row_1_full"}
    aie.flow(%tile_6_5, DMA : 0, %mem_tile_3_1_1, DMA : 0)
    aie.flow(%mem_tile_3_1_1, DMA : 1, %tile_6_5, DMA : 0)
    %ffnDownAccum1_src = aie.buffer(%tile_6_3) {sym_name = "ffnDownAccum1_src"} : memref<32x64xbf16> 
    %ffnDownAccum1_dst = aie.buffer(%tile_6_3) {sym_name = "ffnDownAccum1_dst"} : memref<32x64xbf16> 
    %ffnDownAccum1_src_empty = aie.lock(%tile_6_3) {init = 1 : i32, sym_name = "ffnDownAccum1_src_empty"}
    %ffnDownAccum1_src_full = aie.lock(%tile_6_3) {init = 0 : i32, sym_name = "ffnDownAccum1_src_full"}
    %ffnDownAccum1_dst_empty = aie.lock(%tile_6_3) {init = 1 : i32, sym_name = "ffnDownAccum1_dst_empty"}
    %ffnDownAccum1_dst_full = aie.lock(%tile_6_3) {init = 0 : i32, sym_name = "ffnDownAccum1_dst_full"}
    %ffnDownAccum1_row_0 = aie.buffer(%mem_tile_5_1_2) {sym_name = "ffnDownAccum1_row_0"} : memref<24576xbf16> 
    %ffnDownAccum1_row_0_empty = aie.lock(%mem_tile_5_1_2) {init = 1 : i32, sym_name = "ffnDownAccum1_row_0_empty"}
    %ffnDownAccum1_row_0_full = aie.lock(%mem_tile_5_1_2) {init = 0 : i32, sym_name = "ffnDownAccum1_row_0_full"}
    %ffnDownAccum1_row_1 = aie.buffer(%mem_tile_5_1_2) {sym_name = "ffnDownAccum1_row_1"} : memref<24576xbf16> 
    %ffnDownAccum1_row_1_empty = aie.lock(%mem_tile_5_1_2) {init = 1 : i32, sym_name = "ffnDownAccum1_row_1_empty"}
    %ffnDownAccum1_row_1_full = aie.lock(%mem_tile_5_1_2) {init = 0 : i32, sym_name = "ffnDownAccum1_row_1_full"}
    aie.flow(%tile_6_3, DMA : 0, %mem_tile_5_1_2, DMA : 0)
    aie.flow(%mem_tile_5_1_2, DMA : 0, %tile_6_3, DMA : 0)
    %ffnDownAccum3_src = aie.buffer(%tile_7_4) {sym_name = "ffnDownAccum3_src"} : memref<32x64xbf16> 
    %ffnDownAccum3_dst = aie.buffer(%tile_7_4) {sym_name = "ffnDownAccum3_dst"} : memref<32x64xbf16> 
    %ffnDownAccum3_src_empty = aie.lock(%tile_7_4) {init = 1 : i32, sym_name = "ffnDownAccum3_src_empty"}
    %ffnDownAccum3_src_full = aie.lock(%tile_7_4) {init = 0 : i32, sym_name = "ffnDownAccum3_src_full"}
    %ffnDownAccum3_dst_empty = aie.lock(%tile_7_4) {init = 1 : i32, sym_name = "ffnDownAccum3_dst_empty"}
    %ffnDownAccum3_dst_full = aie.lock(%tile_7_4) {init = 0 : i32, sym_name = "ffnDownAccum3_dst_full"}
    %ffnDownAccum3_row_0 = aie.buffer(%mem_tile_7_1_3) {sym_name = "ffnDownAccum3_row_0"} : memref<24576xbf16> 
    %ffnDownAccum3_row_0_empty = aie.lock(%mem_tile_7_1_3) {init = 1 : i32, sym_name = "ffnDownAccum3_row_0_empty"}
    %ffnDownAccum3_row_0_full = aie.lock(%mem_tile_7_1_3) {init = 0 : i32, sym_name = "ffnDownAccum3_row_0_full"}
    %ffnDownAccum3_row_1 = aie.buffer(%mem_tile_7_1_3) {sym_name = "ffnDownAccum3_row_1"} : memref<24576xbf16> 
    %ffnDownAccum3_row_1_empty = aie.lock(%mem_tile_7_1_3) {init = 1 : i32, sym_name = "ffnDownAccum3_row_1_empty"}
    %ffnDownAccum3_row_1_full = aie.lock(%mem_tile_7_1_3) {init = 0 : i32, sym_name = "ffnDownAccum3_row_1_full"}
    aie.flow(%tile_7_4, DMA : 0, %mem_tile_7_1_3, DMA : 0)
    aie.flow(%mem_tile_7_1_3, DMA : 0, %tile_7_4, DMA : 0)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_4 = arith.constant 0 : index
        %c0_i32 = arith.constant 0 : i32
        memref.store %c0_i32, %idx_buffer_qk_0[%c0_4] : memref<2xi32>
        %c1_5 = arith.constant 1 : index
        %c0_i32_6 = arith.constant 0 : i32
        memref.store %c0_i32_6, %idx_buffer_qk_0[%c1_5] : memref<2xi32>
        %c0_7 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_8 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c3 step %c1_8 {
          %0 = aie.objectfifo.acquire @memQ0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %c0_9 = arith.constant 0 : index
          %c1_10 = arith.constant 1 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_9 to %c1_10 step %c1_11 {
            %4 = aie.objectfifo.acquire @memK0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %6 = aie.objectfifo.acquire @memA0(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @zero_bf16(%7) : (memref<32x64xbf16>) -> ()
            func.call @matmul_bf16_bf16_wrapper(%1, %5, %7, %idx_buffer_qk_0) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<2xi32>) -> ()
            aie.objectfifo.release @memK0(Consume, 1)
            aie.objectfifo.release @memA0(Produce, 1)
            %c0_17 = arith.constant 0 : index
            %8 = memref.load %idx_buffer_qk_0[%c0_17] : memref<2xi32>
            %c0_i32_18 = arith.constant 0 : i32
            %9 = arith.addi %8, %c0_i32_18 : i32
            %c0_19 = arith.constant 0 : index
            memref.store %9, %idx_buffer_qk_0[%c0_19] : memref<2xi32>
          }
          %c0_12 = arith.constant 0 : index
          %c0_i32_13 = arith.constant 0 : i32
          memref.store %c0_i32_13, %idx_buffer_qk_0[%c0_12] : memref<2xi32>
          %c1_14 = arith.constant 1 : index
          %2 = memref.load %idx_buffer_qk_0[%c1_14] : memref<2xi32>
          %c0_i32_15 = arith.constant 0 : i32
          %3 = arith.addi %2, %c0_i32_15 : i32
          %c1_16 = arith.constant 1 : index
          memref.store %3, %idx_buffer_qk_0[%c1_16] : memref<2xi32>
          aie.objectfifo.release @memQ0(Consume, 1)
        }
      }
      aie.end
    } {link_with = "ep_12h_64s_64e_4ph_12pa_4g_4pf_d-1_m0_fa_n1-1_n2-1_me_c57caa9b6610_kernels.a", stack_size = 3328 : i32}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_4 = arith.constant 0 : index
        %c0_i32 = arith.constant 0 : i32
        memref.store %c0_i32, %idx_buffer_softmax_0[%c0_4] : memref<2xi32>
        %c1_5 = arith.constant 1 : index
        %c0_i32_6 = arith.constant 0 : i32
        memref.store %c0_i32_6, %idx_buffer_softmax_0[%c1_5] : memref<2xi32>
        %c0_7 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_8 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c3 step %c1_8 {
          %c32_i32 = arith.constant 32 : i32
          func.call @init_scale_buffer(%scale_buffer_softmax_0, %c32_i32) : (memref<128xbf16>, i32) -> ()
          %c0_9 = arith.constant 0 : index
          %c1_10 = arith.constant 1 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_9 to %c1_10 step %c1_11 {
            %2 = aie.objectfifo.acquire @memP0(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            %4 = aie.objectfifo.acquire @memA0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            %6 = aie.objectfifo.acquire @scaleOF0(Produce, 1) : !aie.objectfifosubview<memref<128xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
            %cst = arith.constant 1.806640e-01 : bf16
            %c32_i32_17 = arith.constant 32 : i32
            %c64_i32 = arith.constant 64 : i32
            %c64_i32_18 = arith.constant 64 : i32
            %c64_i32_19 = arith.constant 64 : i32
            func.call @partial_softmax(%5, %3, %scale_buffer_softmax_0, %idx_buffer_softmax_0, %cst, %c32_i32_17, %c64_i32, %c64_i32_18, %c64_i32_19) : (memref<32x64xbf16>, memref<32x64xbf16>, memref<128xbf16>, memref<2xi32>, bf16, i32, i32, i32, i32) -> ()
            %c128_i32 = arith.constant 128 : i32
            func.call @passThroughLine(%scale_buffer_softmax_0, %7, %c128_i32) : (memref<128xbf16>, memref<128xbf16>, i32) -> ()
            aie.objectfifo.release @memA0(Consume, 1)
            aie.objectfifo.release @memP0(Produce, 1)
            aie.objectfifo.release @scaleOF0(Produce, 1)
            %c0_20 = arith.constant 0 : index
            %8 = memref.load %idx_buffer_softmax_0[%c0_20] : memref<2xi32>
            %c0_i32_21 = arith.constant 0 : i32
            %9 = arith.addi %8, %c0_i32_21 : i32
            %c0_22 = arith.constant 0 : index
            memref.store %9, %idx_buffer_softmax_0[%c0_22] : memref<2xi32>
          }
          %c0_12 = arith.constant 0 : index
          %c0_i32_13 = arith.constant 0 : i32
          memref.store %c0_i32_13, %idx_buffer_softmax_0[%c0_12] : memref<2xi32>
          %c1_14 = arith.constant 1 : index
          %0 = memref.load %idx_buffer_softmax_0[%c1_14] : memref<2xi32>
          %c0_i32_15 = arith.constant 0 : i32
          %1 = arith.addi %0, %c0_i32_15 : i32
          %c1_16 = arith.constant 1 : index
          memref.store %1, %idx_buffer_softmax_0[%c1_16] : memref<2xi32>
        }
      }
      aie.end
    } {link_with = "ep_12h_64s_64e_4ph_12pa_4g_4pf_d-1_m0_fa_n1-1_n2-1_me_c57caa9b6610_kernels.a", stack_size = 3328 : i32}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_4 = arith.constant 0 : index
        %c0_i32 = arith.constant 0 : i32
        memref.store %c0_i32, %idx_buffer_pv_0[%c0_4] : memref<2xi32>
        %c1_5 = arith.constant 1 : index
        %c0_i32_6 = arith.constant 0 : i32
        memref.store %c0_i32_6, %idx_buffer_pv_0[%c1_5] : memref<2xi32>
        %c0_7 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_8 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c3 step %c1_8 {
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
          %c0_i32_9 = arith.constant 0 : i32
          func.call @matmul_PV(%3, %5, %1, %7, %c32_i32, %c0_i32_9, %idx_buffer_pv_0) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<128xbf16>, i32, i32, memref<2xi32>) -> ()
          aie.objectfifo.release @memP0(Consume, 1)
          aie.objectfifo.release @memV0(Consume, 1)
          aie.objectfifo.release @scaleOF0(Consume, 1)
          %c0_10 = arith.constant 0 : index
          %8 = memref.load %idx_buffer_pv_0[%c0_10] : memref<2xi32>
          %c0_i32_11 = arith.constant 0 : i32
          %9 = arith.addi %8, %c0_i32_11 : i32
          %c0_12 = arith.constant 0 : index
          memref.store %9, %idx_buffer_pv_0[%c0_12] : memref<2xi32>
          %c32_i32_13 = arith.constant 32 : i32
          func.call @rescale_O(%1, %7, %c32_i32_13, %idx_buffer_pv_0) : (memref<32x64xbf16>, memref<128xbf16>, i32, memref<2xi32>) -> ()
          %c0_14 = arith.constant 0 : index
          %10 = memref.load %idx_buffer_pv_0[%c0_14] : memref<2xi32>
          %c0_i32_15 = arith.constant 0 : i32
          %11 = arith.addi %10, %c0_i32_15 : i32
          %c0_16 = arith.constant 0 : index
          memref.store %11, %idx_buffer_pv_0[%c0_16] : memref<2xi32>
          %c0_17 = arith.constant 0 : index
          %c0_i32_18 = arith.constant 0 : i32
          memref.store %c0_i32_18, %idx_buffer_pv_0[%c0_17] : memref<2xi32>
          %c1_19 = arith.constant 1 : index
          %12 = memref.load %idx_buffer_pv_0[%c1_19] : memref<2xi32>
          %c0_i32_20 = arith.constant 0 : i32
          %13 = arith.addi %12, %c0_i32_20 : i32
          %c1_21 = arith.constant 1 : index
          memref.store %13, %idx_buffer_pv_0[%c1_21] : memref<2xi32>
          aie.objectfifo.release @outOProj0(Produce, 1)
        }
      }
      aie.end
    } {link_with = "ep_12h_64s_64e_4ph_12pa_4g_4pf_d-1_m0_fa_n1-1_n2-1_me_c57caa9b6610_kernels.a", stack_size = 3328 : i32}
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        func.call @zero_bf16_o_proj(%o_proj_zero_scratch_0) : (memref<32x64xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c3 step %c1_5 {
          %0 = aie.objectfifo.acquire @outOProj0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %c0_8 = arith.constant 0 : index
          %c12_9 = arith.constant 12 : index
          %c1_10 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %c12_9 step %c1_10 {
            %2 = aie.objectfifo.acquire @memOW0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            func.call @matmul_with_acc_bf16_bf16_o_proj(%1, %3, %o_proj_zero_scratch_0, %o_proj_partial_scratch_0) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
            %4 = aie.objectfifo.acquire @outOGroupPart0(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            %c2048_i32 = arith.constant 2048 : i32
            func.call @passThroughLine_o_proj(%o_proj_partial_scratch_0, %5, %c2048_i32) : (memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
            aie.objectfifo.release @outOGroupPart0(Produce, 1)
            aie.objectfifo.release @memOW0(Consume, 1)
          }
          aie.objectfifo.release @outOProj0(Consume, 1)
        }
        %c0_6 = arith.constant 0 : index
        %c12 = arith.constant 12 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %c12 step %c1_7 {
        }
      }
      aie.end
    } {link_with = "ep_12h_64s_64e_4ph_12pa_4g_4pf_d-1_m0_fa_n1-1_n2-1_me_c57caa9b6610_kernels.a", stack_size = 3328 : i32}
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_4 = arith.constant 0 : index
        %c0_i32 = arith.constant 0 : i32
        memref.store %c0_i32, %idx_buffer_qk_1[%c0_4] : memref<2xi32>
        %c1_5 = arith.constant 1 : index
        %c0_i32_6 = arith.constant 0 : i32
        memref.store %c0_i32_6, %idx_buffer_qk_1[%c1_5] : memref<2xi32>
        %c0_7 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_8 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c3 step %c1_8 {
          %0 = aie.objectfifo.acquire @memQ1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %c0_9 = arith.constant 0 : index
          %c1_10 = arith.constant 1 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_9 to %c1_10 step %c1_11 {
            %4 = aie.objectfifo.acquire @memK1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %6 = aie.objectfifo.acquire @memA1(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @zero_bf16(%7) : (memref<32x64xbf16>) -> ()
            func.call @matmul_bf16_bf16_wrapper(%1, %5, %7, %idx_buffer_qk_1) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<2xi32>) -> ()
            aie.objectfifo.release @memK1(Consume, 1)
            aie.objectfifo.release @memA1(Produce, 1)
            %c0_17 = arith.constant 0 : index
            %8 = memref.load %idx_buffer_qk_1[%c0_17] : memref<2xi32>
            %c0_i32_18 = arith.constant 0 : i32
            %9 = arith.addi %8, %c0_i32_18 : i32
            %c0_19 = arith.constant 0 : index
            memref.store %9, %idx_buffer_qk_1[%c0_19] : memref<2xi32>
          }
          %c0_12 = arith.constant 0 : index
          %c0_i32_13 = arith.constant 0 : i32
          memref.store %c0_i32_13, %idx_buffer_qk_1[%c0_12] : memref<2xi32>
          %c1_14 = arith.constant 1 : index
          %2 = memref.load %idx_buffer_qk_1[%c1_14] : memref<2xi32>
          %c0_i32_15 = arith.constant 0 : i32
          %3 = arith.addi %2, %c0_i32_15 : i32
          %c1_16 = arith.constant 1 : index
          memref.store %3, %idx_buffer_qk_1[%c1_16] : memref<2xi32>
          aie.objectfifo.release @memQ1(Consume, 1)
        }
      }
      aie.end
    } {link_with = "ep_12h_64s_64e_4ph_12pa_4g_4pf_d-1_m0_fa_n1-1_n2-1_me_c57caa9b6610_kernels.a", stack_size = 3328 : i32}
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_4 = arith.constant 0 : index
        %c0_i32 = arith.constant 0 : i32
        memref.store %c0_i32, %idx_buffer_softmax_1[%c0_4] : memref<2xi32>
        %c1_5 = arith.constant 1 : index
        %c0_i32_6 = arith.constant 0 : i32
        memref.store %c0_i32_6, %idx_buffer_softmax_1[%c1_5] : memref<2xi32>
        %c0_7 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_8 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c3 step %c1_8 {
          %c32_i32 = arith.constant 32 : i32
          func.call @init_scale_buffer(%scale_buffer_softmax_1, %c32_i32) : (memref<128xbf16>, i32) -> ()
          %c0_9 = arith.constant 0 : index
          %c1_10 = arith.constant 1 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_9 to %c1_10 step %c1_11 {
            %2 = aie.objectfifo.acquire @memP1(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            %4 = aie.objectfifo.acquire @memA1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            %6 = aie.objectfifo.acquire @scaleOF1(Produce, 1) : !aie.objectfifosubview<memref<128xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
            %cst = arith.constant 1.806640e-01 : bf16
            %c32_i32_17 = arith.constant 32 : i32
            %c64_i32 = arith.constant 64 : i32
            %c64_i32_18 = arith.constant 64 : i32
            %c64_i32_19 = arith.constant 64 : i32
            func.call @partial_softmax(%5, %3, %scale_buffer_softmax_1, %idx_buffer_softmax_1, %cst, %c32_i32_17, %c64_i32, %c64_i32_18, %c64_i32_19) : (memref<32x64xbf16>, memref<32x64xbf16>, memref<128xbf16>, memref<2xi32>, bf16, i32, i32, i32, i32) -> ()
            %c128_i32 = arith.constant 128 : i32
            func.call @passThroughLine(%scale_buffer_softmax_1, %7, %c128_i32) : (memref<128xbf16>, memref<128xbf16>, i32) -> ()
            aie.objectfifo.release @memA1(Consume, 1)
            aie.objectfifo.release @memP1(Produce, 1)
            aie.objectfifo.release @scaleOF1(Produce, 1)
            %c0_20 = arith.constant 0 : index
            %8 = memref.load %idx_buffer_softmax_1[%c0_20] : memref<2xi32>
            %c0_i32_21 = arith.constant 0 : i32
            %9 = arith.addi %8, %c0_i32_21 : i32
            %c0_22 = arith.constant 0 : index
            memref.store %9, %idx_buffer_softmax_1[%c0_22] : memref<2xi32>
          }
          %c0_12 = arith.constant 0 : index
          %c0_i32_13 = arith.constant 0 : i32
          memref.store %c0_i32_13, %idx_buffer_softmax_1[%c0_12] : memref<2xi32>
          %c1_14 = arith.constant 1 : index
          %0 = memref.load %idx_buffer_softmax_1[%c1_14] : memref<2xi32>
          %c0_i32_15 = arith.constant 0 : i32
          %1 = arith.addi %0, %c0_i32_15 : i32
          %c1_16 = arith.constant 1 : index
          memref.store %1, %idx_buffer_softmax_1[%c1_16] : memref<2xi32>
        }
      }
      aie.end
    } {link_with = "ep_12h_64s_64e_4ph_12pa_4g_4pf_d-1_m0_fa_n1-1_n2-1_me_c57caa9b6610_kernels.a", stack_size = 3328 : i32}
    %core_1_4 = aie.core(%tile_1_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_4 = arith.constant 0 : index
        %c0_i32 = arith.constant 0 : i32
        memref.store %c0_i32, %idx_buffer_pv_1[%c0_4] : memref<2xi32>
        %c1_5 = arith.constant 1 : index
        %c0_i32_6 = arith.constant 0 : i32
        memref.store %c0_i32_6, %idx_buffer_pv_1[%c1_5] : memref<2xi32>
        %c0_7 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_8 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c3 step %c1_8 {
          %0 = aie.objectfifo.acquire @outOProj1(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @zero_bf16(%1) : (memref<32x64xbf16>) -> ()
          %2 = aie.objectfifo.acquire @memP1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %4 = aie.objectfifo.acquire @memV1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
          %6 = aie.objectfifo.acquire @scaleOF1(Consume, 1) : !aie.objectfifosubview<memref<128xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
          %c32_i32 = arith.constant 32 : i32
          %c0_i32_9 = arith.constant 0 : i32
          func.call @matmul_PV(%3, %5, %1, %7, %c32_i32, %c0_i32_9, %idx_buffer_pv_1) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<128xbf16>, i32, i32, memref<2xi32>) -> ()
          aie.objectfifo.release @memP1(Consume, 1)
          aie.objectfifo.release @memV1(Consume, 1)
          aie.objectfifo.release @scaleOF1(Consume, 1)
          %c0_10 = arith.constant 0 : index
          %8 = memref.load %idx_buffer_pv_1[%c0_10] : memref<2xi32>
          %c0_i32_11 = arith.constant 0 : i32
          %9 = arith.addi %8, %c0_i32_11 : i32
          %c0_12 = arith.constant 0 : index
          memref.store %9, %idx_buffer_pv_1[%c0_12] : memref<2xi32>
          %c32_i32_13 = arith.constant 32 : i32
          func.call @rescale_O(%1, %7, %c32_i32_13, %idx_buffer_pv_1) : (memref<32x64xbf16>, memref<128xbf16>, i32, memref<2xi32>) -> ()
          %c0_14 = arith.constant 0 : index
          %10 = memref.load %idx_buffer_pv_1[%c0_14] : memref<2xi32>
          %c0_i32_15 = arith.constant 0 : i32
          %11 = arith.addi %10, %c0_i32_15 : i32
          %c0_16 = arith.constant 0 : index
          memref.store %11, %idx_buffer_pv_1[%c0_16] : memref<2xi32>
          %c0_17 = arith.constant 0 : index
          %c0_i32_18 = arith.constant 0 : i32
          memref.store %c0_i32_18, %idx_buffer_pv_1[%c0_17] : memref<2xi32>
          %c1_19 = arith.constant 1 : index
          %12 = memref.load %idx_buffer_pv_1[%c1_19] : memref<2xi32>
          %c0_i32_20 = arith.constant 0 : i32
          %13 = arith.addi %12, %c0_i32_20 : i32
          %c1_21 = arith.constant 1 : index
          memref.store %13, %idx_buffer_pv_1[%c1_21] : memref<2xi32>
          aie.objectfifo.release @outOProj1(Produce, 1)
        }
      }
      aie.end
    } {link_with = "ep_12h_64s_64e_4ph_12pa_4g_4pf_d-1_m0_fa_n1-1_n2-1_me_c57caa9b6610_kernels.a", stack_size = 3328 : i32}
    %core_1_5 = aie.core(%tile_1_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        func.call @zero_bf16_o_proj(%o_proj_zero_scratch_1) : (memref<32x64xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c3 step %c1_5 {
          %0 = aie.objectfifo.acquire @outOProj1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %c0_8 = arith.constant 0 : index
          %c12_9 = arith.constant 12 : index
          %c1_10 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %c12_9 step %c1_10 {
            %2 = aie.objectfifo.acquire @memOW1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            func.call @matmul_with_acc_bf16_bf16_o_proj(%1, %3, %o_proj_zero_scratch_1, %o_proj_partial_scratch_1) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
            %4 = aie.objectfifo.acquire @outOGroupPart0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            %c2048_i32 = arith.constant 2048 : i32
            func.call @eltwise_add_bf16_vector_o_proj(%5, %o_proj_partial_scratch_1, %o_proj_partial_scratch_1, %c2048_i32) : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
            aie.objectfifo.release @outOGroupPart0(Consume, 1)
            %6 = aie.objectfifo.acquire @outOGroupPart1(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            %c2048_i32_11 = arith.constant 2048 : i32
            func.call @passThroughLine_o_proj(%o_proj_partial_scratch_1, %7, %c2048_i32_11) : (memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
            aie.objectfifo.release @outOGroupPart1(Produce, 1)
            aie.objectfifo.release @memOW1(Consume, 1)
          }
          aie.objectfifo.release @outOProj1(Consume, 1)
        }
        %c0_6 = arith.constant 0 : index
        %c12 = arith.constant 12 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %c12 step %c1_7 {
        }
      }
      aie.end
    } {link_with = "ep_12h_64s_64e_4ph_12pa_4g_4pf_d-1_m0_fa_n1-1_n2-1_me_c57caa9b6610_kernels.a", stack_size = 3328 : i32}
    %core_2_2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_4 = arith.constant 0 : index
        %c0_i32 = arith.constant 0 : i32
        memref.store %c0_i32, %idx_buffer_qk_2[%c0_4] : memref<2xi32>
        %c1_5 = arith.constant 1 : index
        %c0_i32_6 = arith.constant 0 : i32
        memref.store %c0_i32_6, %idx_buffer_qk_2[%c1_5] : memref<2xi32>
        %c0_7 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_8 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c3 step %c1_8 {
          %0 = aie.objectfifo.acquire @memQ2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %c0_9 = arith.constant 0 : index
          %c1_10 = arith.constant 1 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_9 to %c1_10 step %c1_11 {
            %4 = aie.objectfifo.acquire @memK2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %6 = aie.objectfifo.acquire @memA2(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @zero_bf16(%7) : (memref<32x64xbf16>) -> ()
            func.call @matmul_bf16_bf16_wrapper(%1, %5, %7, %idx_buffer_qk_2) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<2xi32>) -> ()
            aie.objectfifo.release @memK2(Consume, 1)
            aie.objectfifo.release @memA2(Produce, 1)
            %c0_17 = arith.constant 0 : index
            %8 = memref.load %idx_buffer_qk_2[%c0_17] : memref<2xi32>
            %c0_i32_18 = arith.constant 0 : i32
            %9 = arith.addi %8, %c0_i32_18 : i32
            %c0_19 = arith.constant 0 : index
            memref.store %9, %idx_buffer_qk_2[%c0_19] : memref<2xi32>
          }
          %c0_12 = arith.constant 0 : index
          %c0_i32_13 = arith.constant 0 : i32
          memref.store %c0_i32_13, %idx_buffer_qk_2[%c0_12] : memref<2xi32>
          %c1_14 = arith.constant 1 : index
          %2 = memref.load %idx_buffer_qk_2[%c1_14] : memref<2xi32>
          %c0_i32_15 = arith.constant 0 : i32
          %3 = arith.addi %2, %c0_i32_15 : i32
          %c1_16 = arith.constant 1 : index
          memref.store %3, %idx_buffer_qk_2[%c1_16] : memref<2xi32>
          aie.objectfifo.release @memQ2(Consume, 1)
        }
      }
      aie.end
    } {link_with = "ep_12h_64s_64e_4ph_12pa_4g_4pf_d-1_m0_fa_n1-1_n2-1_me_c57caa9b6610_kernels.a", stack_size = 3328 : i32}
    %core_2_3 = aie.core(%tile_2_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_4 = arith.constant 0 : index
        %c0_i32 = arith.constant 0 : i32
        memref.store %c0_i32, %idx_buffer_softmax_2[%c0_4] : memref<2xi32>
        %c1_5 = arith.constant 1 : index
        %c0_i32_6 = arith.constant 0 : i32
        memref.store %c0_i32_6, %idx_buffer_softmax_2[%c1_5] : memref<2xi32>
        %c0_7 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_8 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c3 step %c1_8 {
          %c32_i32 = arith.constant 32 : i32
          func.call @init_scale_buffer(%scale_buffer_softmax_2, %c32_i32) : (memref<128xbf16>, i32) -> ()
          %c0_9 = arith.constant 0 : index
          %c1_10 = arith.constant 1 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_9 to %c1_10 step %c1_11 {
            %2 = aie.objectfifo.acquire @memP2(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            %4 = aie.objectfifo.acquire @memA2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            %6 = aie.objectfifo.acquire @scaleOF2(Produce, 1) : !aie.objectfifosubview<memref<128xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
            %cst = arith.constant 1.806640e-01 : bf16
            %c32_i32_17 = arith.constant 32 : i32
            %c64_i32 = arith.constant 64 : i32
            %c64_i32_18 = arith.constant 64 : i32
            %c64_i32_19 = arith.constant 64 : i32
            func.call @partial_softmax(%5, %3, %scale_buffer_softmax_2, %idx_buffer_softmax_2, %cst, %c32_i32_17, %c64_i32, %c64_i32_18, %c64_i32_19) : (memref<32x64xbf16>, memref<32x64xbf16>, memref<128xbf16>, memref<2xi32>, bf16, i32, i32, i32, i32) -> ()
            %c128_i32 = arith.constant 128 : i32
            func.call @passThroughLine(%scale_buffer_softmax_2, %7, %c128_i32) : (memref<128xbf16>, memref<128xbf16>, i32) -> ()
            aie.objectfifo.release @memA2(Consume, 1)
            aie.objectfifo.release @memP2(Produce, 1)
            aie.objectfifo.release @scaleOF2(Produce, 1)
            %c0_20 = arith.constant 0 : index
            %8 = memref.load %idx_buffer_softmax_2[%c0_20] : memref<2xi32>
            %c0_i32_21 = arith.constant 0 : i32
            %9 = arith.addi %8, %c0_i32_21 : i32
            %c0_22 = arith.constant 0 : index
            memref.store %9, %idx_buffer_softmax_2[%c0_22] : memref<2xi32>
          }
          %c0_12 = arith.constant 0 : index
          %c0_i32_13 = arith.constant 0 : i32
          memref.store %c0_i32_13, %idx_buffer_softmax_2[%c0_12] : memref<2xi32>
          %c1_14 = arith.constant 1 : index
          %0 = memref.load %idx_buffer_softmax_2[%c1_14] : memref<2xi32>
          %c0_i32_15 = arith.constant 0 : i32
          %1 = arith.addi %0, %c0_i32_15 : i32
          %c1_16 = arith.constant 1 : index
          memref.store %1, %idx_buffer_softmax_2[%c1_16] : memref<2xi32>
        }
      }
      aie.end
    } {link_with = "ep_12h_64s_64e_4ph_12pa_4g_4pf_d-1_m0_fa_n1-1_n2-1_me_c57caa9b6610_kernels.a", stack_size = 3328 : i32}
    %core_2_4 = aie.core(%tile_2_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_4 = arith.constant 0 : index
        %c0_i32 = arith.constant 0 : i32
        memref.store %c0_i32, %idx_buffer_pv_2[%c0_4] : memref<2xi32>
        %c1_5 = arith.constant 1 : index
        %c0_i32_6 = arith.constant 0 : i32
        memref.store %c0_i32_6, %idx_buffer_pv_2[%c1_5] : memref<2xi32>
        %c0_7 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_8 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c3 step %c1_8 {
          %0 = aie.objectfifo.acquire @outOProj2(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @zero_bf16(%1) : (memref<32x64xbf16>) -> ()
          %2 = aie.objectfifo.acquire @memP2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %4 = aie.objectfifo.acquire @memV2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
          %6 = aie.objectfifo.acquire @scaleOF2(Consume, 1) : !aie.objectfifosubview<memref<128xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
          %c32_i32 = arith.constant 32 : i32
          %c0_i32_9 = arith.constant 0 : i32
          func.call @matmul_PV(%3, %5, %1, %7, %c32_i32, %c0_i32_9, %idx_buffer_pv_2) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<128xbf16>, i32, i32, memref<2xi32>) -> ()
          aie.objectfifo.release @memP2(Consume, 1)
          aie.objectfifo.release @memV2(Consume, 1)
          aie.objectfifo.release @scaleOF2(Consume, 1)
          %c0_10 = arith.constant 0 : index
          %8 = memref.load %idx_buffer_pv_2[%c0_10] : memref<2xi32>
          %c0_i32_11 = arith.constant 0 : i32
          %9 = arith.addi %8, %c0_i32_11 : i32
          %c0_12 = arith.constant 0 : index
          memref.store %9, %idx_buffer_pv_2[%c0_12] : memref<2xi32>
          %c32_i32_13 = arith.constant 32 : i32
          func.call @rescale_O(%1, %7, %c32_i32_13, %idx_buffer_pv_2) : (memref<32x64xbf16>, memref<128xbf16>, i32, memref<2xi32>) -> ()
          %c0_14 = arith.constant 0 : index
          %10 = memref.load %idx_buffer_pv_2[%c0_14] : memref<2xi32>
          %c0_i32_15 = arith.constant 0 : i32
          %11 = arith.addi %10, %c0_i32_15 : i32
          %c0_16 = arith.constant 0 : index
          memref.store %11, %idx_buffer_pv_2[%c0_16] : memref<2xi32>
          %c0_17 = arith.constant 0 : index
          %c0_i32_18 = arith.constant 0 : i32
          memref.store %c0_i32_18, %idx_buffer_pv_2[%c0_17] : memref<2xi32>
          %c1_19 = arith.constant 1 : index
          %12 = memref.load %idx_buffer_pv_2[%c1_19] : memref<2xi32>
          %c0_i32_20 = arith.constant 0 : i32
          %13 = arith.addi %12, %c0_i32_20 : i32
          %c1_21 = arith.constant 1 : index
          memref.store %13, %idx_buffer_pv_2[%c1_21] : memref<2xi32>
          aie.objectfifo.release @outOProj2(Produce, 1)
        }
      }
      aie.end
    } {link_with = "ep_12h_64s_64e_4ph_12pa_4g_4pf_d-1_m0_fa_n1-1_n2-1_me_c57caa9b6610_kernels.a", stack_size = 3328 : i32}
    %core_2_5 = aie.core(%tile_2_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        func.call @zero_bf16_o_proj(%o_proj_zero_scratch_2) : (memref<32x64xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c3 step %c1_5 {
          %0 = aie.objectfifo.acquire @outOProj2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %c0_8 = arith.constant 0 : index
          %c12_9 = arith.constant 12 : index
          %c1_10 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %c12_9 step %c1_10 {
            %2 = aie.objectfifo.acquire @memOW2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            func.call @matmul_with_acc_bf16_bf16_o_proj(%1, %3, %o_proj_zero_scratch_2, %o_proj_partial_scratch_2) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
            %4 = aie.objectfifo.acquire @outOGroupPart1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            %c2048_i32 = arith.constant 2048 : i32
            func.call @eltwise_add_bf16_vector_o_proj(%5, %o_proj_partial_scratch_2, %o_proj_partial_scratch_2, %c2048_i32) : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
            aie.objectfifo.release @outOGroupPart1(Consume, 1)
            %6 = aie.objectfifo.acquire @outOGroupPart2(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            %c2048_i32_11 = arith.constant 2048 : i32
            func.call @passThroughLine_o_proj(%o_proj_partial_scratch_2, %7, %c2048_i32_11) : (memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
            aie.objectfifo.release @outOGroupPart2(Produce, 1)
            aie.objectfifo.release @memOW2(Consume, 1)
          }
          aie.objectfifo.release @outOProj2(Consume, 1)
        }
        %c0_6 = arith.constant 0 : index
        %c12 = arith.constant 12 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %c12 step %c1_7 {
        }
      }
      aie.end
    } {link_with = "ep_12h_64s_64e_4ph_12pa_4g_4pf_d-1_m0_fa_n1-1_n2-1_me_c57caa9b6610_kernels.a", stack_size = 3328 : i32}
    %core_3_2 = aie.core(%tile_3_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_4 = arith.constant 0 : index
        %c0_i32 = arith.constant 0 : i32
        memref.store %c0_i32, %idx_buffer_qk_3[%c0_4] : memref<2xi32>
        %c1_5 = arith.constant 1 : index
        %c0_i32_6 = arith.constant 0 : i32
        memref.store %c0_i32_6, %idx_buffer_qk_3[%c1_5] : memref<2xi32>
        %c0_7 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_8 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c3 step %c1_8 {
          %0 = aie.objectfifo.acquire @memQ3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %c0_9 = arith.constant 0 : index
          %c1_10 = arith.constant 1 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_9 to %c1_10 step %c1_11 {
            %4 = aie.objectfifo.acquire @memK3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %6 = aie.objectfifo.acquire @memA3(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @zero_bf16(%7) : (memref<32x64xbf16>) -> ()
            func.call @matmul_bf16_bf16_wrapper(%1, %5, %7, %idx_buffer_qk_3) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<2xi32>) -> ()
            aie.objectfifo.release @memK3(Consume, 1)
            aie.objectfifo.release @memA3(Produce, 1)
            %c0_17 = arith.constant 0 : index
            %8 = memref.load %idx_buffer_qk_3[%c0_17] : memref<2xi32>
            %c0_i32_18 = arith.constant 0 : i32
            %9 = arith.addi %8, %c0_i32_18 : i32
            %c0_19 = arith.constant 0 : index
            memref.store %9, %idx_buffer_qk_3[%c0_19] : memref<2xi32>
          }
          %c0_12 = arith.constant 0 : index
          %c0_i32_13 = arith.constant 0 : i32
          memref.store %c0_i32_13, %idx_buffer_qk_3[%c0_12] : memref<2xi32>
          %c1_14 = arith.constant 1 : index
          %2 = memref.load %idx_buffer_qk_3[%c1_14] : memref<2xi32>
          %c0_i32_15 = arith.constant 0 : i32
          %3 = arith.addi %2, %c0_i32_15 : i32
          %c1_16 = arith.constant 1 : index
          memref.store %3, %idx_buffer_qk_3[%c1_16] : memref<2xi32>
          aie.objectfifo.release @memQ3(Consume, 1)
        }
      }
      aie.end
    } {link_with = "ep_12h_64s_64e_4ph_12pa_4g_4pf_d-1_m0_fa_n1-1_n2-1_me_c57caa9b6610_kernels.a", stack_size = 3328 : i32}
    %core_3_3 = aie.core(%tile_3_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_4 = arith.constant 0 : index
        %c0_i32 = arith.constant 0 : i32
        memref.store %c0_i32, %idx_buffer_softmax_3[%c0_4] : memref<2xi32>
        %c1_5 = arith.constant 1 : index
        %c0_i32_6 = arith.constant 0 : i32
        memref.store %c0_i32_6, %idx_buffer_softmax_3[%c1_5] : memref<2xi32>
        %c0_7 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_8 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c3 step %c1_8 {
          %c32_i32 = arith.constant 32 : i32
          func.call @init_scale_buffer(%scale_buffer_softmax_3, %c32_i32) : (memref<128xbf16>, i32) -> ()
          %c0_9 = arith.constant 0 : index
          %c1_10 = arith.constant 1 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_9 to %c1_10 step %c1_11 {
            %2 = aie.objectfifo.acquire @memP3(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            %4 = aie.objectfifo.acquire @memA3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            %6 = aie.objectfifo.acquire @scaleOF3(Produce, 1) : !aie.objectfifosubview<memref<128xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
            %cst = arith.constant 1.806640e-01 : bf16
            %c32_i32_17 = arith.constant 32 : i32
            %c64_i32 = arith.constant 64 : i32
            %c64_i32_18 = arith.constant 64 : i32
            %c64_i32_19 = arith.constant 64 : i32
            func.call @partial_softmax(%5, %3, %scale_buffer_softmax_3, %idx_buffer_softmax_3, %cst, %c32_i32_17, %c64_i32, %c64_i32_18, %c64_i32_19) : (memref<32x64xbf16>, memref<32x64xbf16>, memref<128xbf16>, memref<2xi32>, bf16, i32, i32, i32, i32) -> ()
            %c128_i32 = arith.constant 128 : i32
            func.call @passThroughLine(%scale_buffer_softmax_3, %7, %c128_i32) : (memref<128xbf16>, memref<128xbf16>, i32) -> ()
            aie.objectfifo.release @memA3(Consume, 1)
            aie.objectfifo.release @memP3(Produce, 1)
            aie.objectfifo.release @scaleOF3(Produce, 1)
            %c0_20 = arith.constant 0 : index
            %8 = memref.load %idx_buffer_softmax_3[%c0_20] : memref<2xi32>
            %c0_i32_21 = arith.constant 0 : i32
            %9 = arith.addi %8, %c0_i32_21 : i32
            %c0_22 = arith.constant 0 : index
            memref.store %9, %idx_buffer_softmax_3[%c0_22] : memref<2xi32>
          }
          %c0_12 = arith.constant 0 : index
          %c0_i32_13 = arith.constant 0 : i32
          memref.store %c0_i32_13, %idx_buffer_softmax_3[%c0_12] : memref<2xi32>
          %c1_14 = arith.constant 1 : index
          %0 = memref.load %idx_buffer_softmax_3[%c1_14] : memref<2xi32>
          %c0_i32_15 = arith.constant 0 : i32
          %1 = arith.addi %0, %c0_i32_15 : i32
          %c1_16 = arith.constant 1 : index
          memref.store %1, %idx_buffer_softmax_3[%c1_16] : memref<2xi32>
        }
      }
      aie.end
    } {link_with = "ep_12h_64s_64e_4ph_12pa_4g_4pf_d-1_m0_fa_n1-1_n2-1_me_c57caa9b6610_kernels.a", stack_size = 3328 : i32}
    %core_3_4 = aie.core(%tile_3_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_4 = arith.constant 0 : index
        %c0_i32 = arith.constant 0 : i32
        memref.store %c0_i32, %idx_buffer_pv_3[%c0_4] : memref<2xi32>
        %c1_5 = arith.constant 1 : index
        %c0_i32_6 = arith.constant 0 : i32
        memref.store %c0_i32_6, %idx_buffer_pv_3[%c1_5] : memref<2xi32>
        %c0_7 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_8 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c3 step %c1_8 {
          %0 = aie.objectfifo.acquire @outOProj3(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @zero_bf16(%1) : (memref<32x64xbf16>) -> ()
          %2 = aie.objectfifo.acquire @memP3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %4 = aie.objectfifo.acquire @memV3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
          %6 = aie.objectfifo.acquire @scaleOF3(Consume, 1) : !aie.objectfifosubview<memref<128xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
          %c32_i32 = arith.constant 32 : i32
          %c0_i32_9 = arith.constant 0 : i32
          func.call @matmul_PV(%3, %5, %1, %7, %c32_i32, %c0_i32_9, %idx_buffer_pv_3) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<128xbf16>, i32, i32, memref<2xi32>) -> ()
          aie.objectfifo.release @memP3(Consume, 1)
          aie.objectfifo.release @memV3(Consume, 1)
          aie.objectfifo.release @scaleOF3(Consume, 1)
          %c0_10 = arith.constant 0 : index
          %8 = memref.load %idx_buffer_pv_3[%c0_10] : memref<2xi32>
          %c0_i32_11 = arith.constant 0 : i32
          %9 = arith.addi %8, %c0_i32_11 : i32
          %c0_12 = arith.constant 0 : index
          memref.store %9, %idx_buffer_pv_3[%c0_12] : memref<2xi32>
          %c32_i32_13 = arith.constant 32 : i32
          func.call @rescale_O(%1, %7, %c32_i32_13, %idx_buffer_pv_3) : (memref<32x64xbf16>, memref<128xbf16>, i32, memref<2xi32>) -> ()
          %c0_14 = arith.constant 0 : index
          %10 = memref.load %idx_buffer_pv_3[%c0_14] : memref<2xi32>
          %c0_i32_15 = arith.constant 0 : i32
          %11 = arith.addi %10, %c0_i32_15 : i32
          %c0_16 = arith.constant 0 : index
          memref.store %11, %idx_buffer_pv_3[%c0_16] : memref<2xi32>
          %c0_17 = arith.constant 0 : index
          %c0_i32_18 = arith.constant 0 : i32
          memref.store %c0_i32_18, %idx_buffer_pv_3[%c0_17] : memref<2xi32>
          %c1_19 = arith.constant 1 : index
          %12 = memref.load %idx_buffer_pv_3[%c1_19] : memref<2xi32>
          %c0_i32_20 = arith.constant 0 : i32
          %13 = arith.addi %12, %c0_i32_20 : i32
          %c1_21 = arith.constant 1 : index
          memref.store %13, %idx_buffer_pv_3[%c1_21] : memref<2xi32>
          aie.objectfifo.release @outOProj3(Produce, 1)
        }
      }
      aie.end
    } {link_with = "ep_12h_64s_64e_4ph_12pa_4g_4pf_d-1_m0_fa_n1-1_n2-1_me_c57caa9b6610_kernels.a", stack_size = 3328 : i32}
    %core_3_5 = aie.core(%tile_3_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c32_i32 = arith.constant 32 : i32
        func.call @ln_zero_f32(%o_proj_stats_sum_3, %c32_i32) : (memref<32xf32>, i32) -> ()
        %c32_i32_4 = arith.constant 32 : i32
        func.call @ln_zero_f32(%o_proj_stats_sumsq_3, %c32_i32_4) : (memref<32xf32>, i32) -> ()
        func.call @zero_bf16_o_proj(%o_proj_zero_scratch_3) : (memref<32x64xbf16>) -> ()
        %c0_5 = arith.constant 0 : index
        %c12 = arith.constant 12 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_5 to %c12 step %c1_6 {
          aie.use_lock(%outOProjAccum3_src_empty, AcquireGreaterEqual, 1)
          func.call @zero_bf16_o_proj(%outOProjAccum3_src) : (memref<32x64xbf16>) -> ()
          aie.use_lock(%outOProjAccum3_src_full, Release, 1)
        }
        %c0_7 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_8 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c3 step %c1_8 {
          %2 = aie.objectfifo.acquire @outOProj3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %c0_13 = arith.constant 0 : index
          %c12_14 = arith.constant 12 : index
          %c1_15 = arith.constant 1 : index
          scf.for %arg2 = %c0_13 to %c12_14 step %c1_15 {
            %4 = aie.objectfifo.acquire @memOW3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            func.call @matmul_with_acc_bf16_bf16_o_proj(%3, %5, %o_proj_zero_scratch_3, %o_proj_partial_scratch_3) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
            %6 = aie.objectfifo.acquire @outOGroupPart2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            %c2048_i32 = arith.constant 2048 : i32
            func.call @eltwise_add_bf16_vector_o_proj(%7, %o_proj_partial_scratch_3, %o_proj_partial_scratch_3, %c2048_i32) : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
            aie.objectfifo.release @outOGroupPart2(Consume, 1)
            aie.use_lock(%outOProjAccum3_dst_full, AcquireGreaterEqual, 1)
            aie.use_lock(%outOProjAccum3_src_empty, AcquireGreaterEqual, 1)
            %c2048_i32_16 = arith.constant 2048 : i32
            func.call @eltwise_add_bf16_vector_o_proj(%outOProjAccum3_dst, %o_proj_partial_scratch_3, %outOProjAccum3_src, %c2048_i32_16) : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
            aie.use_lock(%outOProjAccum3_src_full, Release, 1)
            aie.use_lock(%outOProjAccum3_dst_empty, Release, 1)
            aie.objectfifo.release @memOW3(Consume, 1)
          }
          aie.objectfifo.release @outOProj3(Consume, 1)
        }
        %c0_9 = arith.constant 0 : index
        %c12_10 = arith.constant 12 : index
        %c1_11 = arith.constant 1 : index
        scf.for %arg1 = %c0_9 to %c12_10 step %c1_11 {
          aie.use_lock(%outOProjAccum3_dst_full, AcquireGreaterEqual, 1)
          func.call @ln_calc_sum_sumsq(%outOProjAccum3_dst, %o_proj_stats_sum_3, %o_proj_stats_sumsq_3) : (memref<32x64xbf16>, memref<32xf32>, memref<32xf32>) -> ()
          %2 = aie.objectfifo.acquire @outOProjInput(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %c2048_i32 = arith.constant 2048 : i32
          func.call @passThroughLine_o_proj(%outOProjAccum3_dst, %3, %c2048_i32) : (memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
          aie.objectfifo.release @outOProjInput(Produce, 1)
          aie.use_lock(%outOProjAccum3_dst_empty, Release, 1)
        }
        %0 = aie.objectfifo.acquire @outOProjInput(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        %c32_i32_12 = arith.constant 32 : i32
        func.call @pack_stats_f32_to_bf16_packet(%o_proj_stats_sum_3, %o_proj_stats_sumsq_3, %1, %c32_i32_12) : (memref<32xf32>, memref<32xf32>, memref<32x64xbf16>, i32) -> ()
        aie.objectfifo.release @outOProjInput(Produce, 1)
      }
      aie.end
    } {link_with = "ep_12h_64s_64e_4ph_12pa_4g_4pf_d-1_m0_fa_n1-1_n2-1_me_c57caa9b6610_kernels.a", stack_size = 3328 : i32}
    %core_6_5 = aie.core(%tile_6_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_4 = arith.constant 0 : index
        %c12 = arith.constant 12 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c12 step %c1_5 {
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
        %c0_6 = arith.constant 0 : index
        %c12_7 = arith.constant 12 : index
        %c1_8 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %c12_7 step %c1_8 {
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
        %c0_9 = arith.constant 0 : index
        %c10 = arith.constant 10 : index
        %c1_10 = arith.constant 1 : index
        scf.for %arg1 = %c0_9 to %c10 step %c1_10 {
          %c0_14 = arith.constant 0 : index
          %c12_15 = arith.constant 12 : index
          %c1_16 = arith.constant 1 : index
          scf.for %arg2 = %c0_14 to %c12_15 step %c1_16 {
            aie.use_lock(%ln1Replay_dst_full, AcquireGreaterEqual, 1)
            %2 = aie.objectfifo.acquire @ln1Norm(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            %c2048_i32 = arith.constant 2048 : i32
            func.call @passThroughLine_o_proj(%ln1Replay_dst, %3, %c2048_i32) : (memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
            aie.objectfifo.release @ln1Norm(Produce, 1)
            aie.use_lock(%ln1Replay_src_empty, AcquireGreaterEqual, 1)
            %c2048_i32_17 = arith.constant 2048 : i32
            func.call @passThroughLine_o_proj(%ln1Replay_dst, %ln1Replay_src, %c2048_i32_17) : (memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
            aie.use_lock(%ln1Replay_src_full, Release, 1)
            aie.use_lock(%ln1Replay_dst_empty, Release, 1)
          }
        }
        %c0_11 = arith.constant 0 : index
        %c12_12 = arith.constant 12 : index
        %c1_13 = arith.constant 1 : index
        scf.for %arg1 = %c0_11 to %c12_12 step %c1_13 {
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
    } {link_with = "ep_12h_64s_64e_4ph_12pa_4g_4pf_d-1_m0_fa_n1-1_n2-1_me_c57caa9b6610_kernels.a"}
    %core_4_2 = aie.core(%tile_4_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_4 = arith.constant 0 : index
        %c12 = arith.constant 12 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c12 step %c1_5 {
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
        %c0_6 = arith.constant 0 : index
        %c11 = arith.constant 11 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %c11 step %c1_7 {
          %c0_11 = arith.constant 0 : index
          %c12_12 = arith.constant 12 : index
          %c1_13 = arith.constant 1 : index
          scf.for %arg2 = %c0_11 to %c12_12 step %c1_13 {
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
        %c0_8 = arith.constant 0 : index
        %c0_9 = arith.constant 0 : index
        %c1_10 = arith.constant 1 : index
        scf.for %arg1 = %c0_8 to %c0_9 step %c1_10 {
          %c0_11 = arith.constant 0 : index
          %c12_12 = arith.constant 12 : index
          %c1_13 = arith.constant 1 : index
          scf.for %arg2 = %c0_11 to %c12_12 step %c1_13 {
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
    } {link_with = "ep_12h_64s_64e_4ph_12pa_4g_4pf_d-1_m0_fa_n1-1_n2-1_me_c57caa9b6610_kernels.a"}
    %core_4_3 = aie.core(%tile_4_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_4 = arith.constant 0 : index
        %c12 = arith.constant 12 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c12 step %c1_5 {
          %0 = index.casts %arg1 : index to i32
          %c12_i32 = arith.constant 12 : i32
          %1 = arith.cmpi slt, %0, %c12_i32 : i32
          scf.if %1 {
            %2 = aie.objectfifo.acquire @ffnUpOut(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            %c0_6 = arith.constant 0 : index
            %c12_7 = arith.constant 12 : index
            %c1_8 = arith.constant 1 : index
            scf.for %arg2 = %c0_6 to %c12_7 step %c1_8 {
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
            %c0_6 = arith.constant 0 : index
            %c12_7 = arith.constant 12 : index
            %c1_8 = arith.constant 1 : index
            scf.for %arg2 = %c0_6 to %c12_7 step %c1_8 {
              %2 = aie.objectfifo.acquire @outLNBroadcast(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
              %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
              aie.objectfifo.release @outLNBroadcast(Consume, 1)
            }
          }
        }
      }
      aie.end
    } {link_with = "ep_12h_64s_64e_4ph_12pa_4g_4pf_d-1_m0_fa_n1-1_n2-1_me_c57caa9b6610_kernels.a", stack_size = 1792 : i32}
    %core_5_3 = aie.core(%tile_5_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_4 = arith.constant 0 : index
        %c12 = arith.constant 12 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c12 step %c1_5 {
          %0 = aie.objectfifo.acquire @ffnUpOut(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %c0_6 = arith.constant 0 : index
          %c12_7 = arith.constant 12 : index
          %c1_8 = arith.constant 1 : index
          scf.for %arg2 = %c0_6 to %c12_7 step %c1_8 {
            %2 = aie.objectfifo.acquire @ffnDownGroup0(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            %4 = aie.objectfifo.acquire @memBDown(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            func.call @ffn_matmul_init_bf16_bf16_down_proj(%1, %5, %3) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
            aie.objectfifo.release @memBDown(Consume, 1)
            aie.objectfifo.release @ffnDownGroup0(Produce, 1)
          }
          aie.objectfifo.release @ffnUpOut(Consume, 1)
        }
      }
      aie.end
    } {link_with = "ep_12h_64s_64e_4ph_12pa_4g_4pf_d-1_m0_fa_n1-1_n2-1_me_c57caa9b6610_kernels.a", stack_size = 3840 : i32}
    %core_6_2 = aie.core(%tile_6_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_4 = arith.constant 0 : index
        %c12 = arith.constant 12 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c12 step %c1_5 {
          %0 = index.casts %arg1 : index to i32
          %c12_i32 = arith.constant 12 : i32
          %1 = arith.cmpi slt, %0, %c12_i32 : i32
          scf.if %1 {
            %2 = aie.objectfifo.acquire @ffnUpOut1(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            %c0_6 = arith.constant 0 : index
            %c12_7 = arith.constant 12 : index
            %c1_8 = arith.constant 1 : index
            scf.for %arg2 = %c0_6 to %c12_7 step %c1_8 {
              %4 = aie.objectfifo.acquire @outLNBroadcast(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
              %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
              %6 = aie.objectfifo.acquire @memBUp1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %8 = index.casts %arg2 : index to i32
              %c0_i32 = arith.constant 0 : i32
              %9 = arith.cmpi eq, %8, %c0_i32 : i32
              scf.if %9 {
                func.call @ffn_matmul_init_bf16_bf16_up_proj(%5, %7, %3) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
              } else {
                func.call @ffn_matmul_bf16_bf16_up_proj(%5, %7, %3) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
              }
              aie.objectfifo.release @memBUp1(Consume, 1)
              aie.objectfifo.release @outLNBroadcast(Consume, 1)
            }
            %c2048_i32 = arith.constant 2048 : i32
            func.call @ffn_gelu_bf16(%3, %3, %c2048_i32) : (memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
            aie.objectfifo.release @ffnUpOut1(Produce, 1)
          } else {
            %c0_6 = arith.constant 0 : index
            %c12_7 = arith.constant 12 : index
            %c1_8 = arith.constant 1 : index
            scf.for %arg2 = %c0_6 to %c12_7 step %c1_8 {
              %2 = aie.objectfifo.acquire @outLNBroadcast(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
              %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
              aie.objectfifo.release @outLNBroadcast(Consume, 1)
            }
          }
        }
      }
      aie.end
    } {link_with = "ep_12h_64s_64e_4ph_12pa_4g_4pf_d-1_m0_fa_n1-1_n2-1_me_c57caa9b6610_kernels.a", stack_size = 1792 : i32}
    %core_6_3 = aie.core(%tile_6_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @ffnUpOut1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        %c0_4 = arith.constant 0 : index
        %c12 = arith.constant 12 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c12 step %c1_5 {
          aie.use_lock(%ffnDownAccum1_src_empty, AcquireGreaterEqual, 1)
          %2 = aie.objectfifo.acquire @ffnDownGroup0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %c2048_i32 = arith.constant 2048 : i32
          func.call @passThroughLine_o_proj(%3, %ffnDownAccum1_src, %c2048_i32) : (memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
          aie.objectfifo.release @ffnDownGroup0(Consume, 1)
          %4 = aie.objectfifo.acquire @memBDown1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
          func.call @ffn_matmul_with_acc_bf16_bf16_down_proj(%1, %5, %ffnDownAccum1_src, %ffnDownAccum1_src) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
          aie.objectfifo.release @memBDown1(Consume, 1)
          aie.use_lock(%ffnDownAccum1_src_full, Release, 1)
        }
        aie.objectfifo.release @ffnUpOut1(Consume, 1)
        %c0_6 = arith.constant 0 : index
        %c11 = arith.constant 11 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %c11 step %c1_7 {
          %2 = aie.objectfifo.acquire @ffnUpOut1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %c0_14 = arith.constant 0 : index
          %c12_15 = arith.constant 12 : index
          %c1_16 = arith.constant 1 : index
          scf.for %arg2 = %c0_14 to %c12_15 step %c1_16 {
            aie.use_lock(%ffnDownAccum1_dst_full, AcquireGreaterEqual, 1)
            aie.use_lock(%ffnDownAccum1_src_empty, AcquireGreaterEqual, 1)
            %4 = aie.objectfifo.acquire @ffnDownGroup0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            %c2048_i32 = arith.constant 2048 : i32
            func.call @passThroughLine_o_proj(%5, %ffnDownAccum1_src, %c2048_i32) : (memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
            aie.objectfifo.release @ffnDownGroup0(Consume, 1)
            %6 = aie.objectfifo.acquire @memBDown1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            func.call @ffn_matmul_with_acc_bf16_bf16_down_proj(%3, %7, %ffnDownAccum1_src, %ffnDownAccum1_src) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
            aie.objectfifo.release @memBDown1(Consume, 1)
            %c2048_i32_17 = arith.constant 2048 : i32
            func.call @eltwise_add_bf16_vector_o_proj(%ffnDownAccum1_src, %ffnDownAccum1_dst, %ffnDownAccum1_src, %c2048_i32_17) : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
            aie.use_lock(%ffnDownAccum1_src_full, Release, 1)
            aie.use_lock(%ffnDownAccum1_dst_empty, Release, 1)
          }
          aie.objectfifo.release @ffnUpOut1(Consume, 1)
        }
        %c0_8 = arith.constant 0 : index
        %c12_9 = arith.constant 12 : index
        %c1_10 = arith.constant 1 : index
        scf.for %arg1 = %c0_8 to %c12_9 step %c1_10 {
          %2 = aie.objectfifo.acquire @ffnDownOut0(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          aie.use_lock(%ffnDownAccum1_dst_full, AcquireGreaterEqual, 1)
          %c2048_i32 = arith.constant 2048 : i32
          func.call @passThroughLine_o_proj(%ffnDownAccum1_dst, %3, %c2048_i32) : (memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
          aie.use_lock(%ffnDownAccum1_src_empty, AcquireGreaterEqual, 1)
          %c2048_i32_14 = arith.constant 2048 : i32
          func.call @passThroughLine_o_proj(%ffnDownAccum1_dst, %ffnDownAccum1_src, %c2048_i32_14) : (memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
          aie.use_lock(%ffnDownAccum1_src_full, Release, 1)
          aie.use_lock(%ffnDownAccum1_dst_empty, Release, 1)
          aie.objectfifo.release @ffnDownOut0(Produce, 1)
        }
        %c0_11 = arith.constant 0 : index
        %c12_12 = arith.constant 12 : index
        %c1_13 = arith.constant 1 : index
        scf.for %arg1 = %c0_11 to %c12_12 step %c1_13 {
          aie.use_lock(%ffnDownAccum1_dst_full, AcquireGreaterEqual, 1)
          %2 = aie.objectfifo.acquire @ffnDownOut0(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %c2048_i32 = arith.constant 2048 : i32
          func.call @passThroughLine_o_proj(%ffnDownAccum1_dst, %3, %c2048_i32) : (memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
          aie.use_lock(%ffnDownAccum1_dst_empty, Release, 1)
          aie.objectfifo.release @ffnDownOut0(Produce, 1)
        }
      }
      aie.end
    } {link_with = "ep_12h_64s_64e_4ph_12pa_4g_4pf_d-1_m0_fa_n1-1_n2-1_me_c57caa9b6610_kernels.a", stack_size = 3840 : i32}
    %core_7_2 = aie.core(%tile_7_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_4 = arith.constant 0 : index
        %c12 = arith.constant 12 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c12 step %c1_5 {
          %0 = index.casts %arg1 : index to i32
          %c12_i32 = arith.constant 12 : i32
          %1 = arith.cmpi slt, %0, %c12_i32 : i32
          scf.if %1 {
            %2 = aie.objectfifo.acquire @ffnUpOut2(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            %c0_6 = arith.constant 0 : index
            %c12_7 = arith.constant 12 : index
            %c1_8 = arith.constant 1 : index
            scf.for %arg2 = %c0_6 to %c12_7 step %c1_8 {
              %4 = aie.objectfifo.acquire @outLNBroadcast(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
              %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
              %6 = aie.objectfifo.acquire @memBUp2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %8 = index.casts %arg2 : index to i32
              %c0_i32 = arith.constant 0 : i32
              %9 = arith.cmpi eq, %8, %c0_i32 : i32
              scf.if %9 {
                func.call @ffn_matmul_init_bf16_bf16_up_proj(%5, %7, %3) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
              } else {
                func.call @ffn_matmul_bf16_bf16_up_proj(%5, %7, %3) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
              }
              aie.objectfifo.release @memBUp2(Consume, 1)
              aie.objectfifo.release @outLNBroadcast(Consume, 1)
            }
            %c2048_i32 = arith.constant 2048 : i32
            func.call @ffn_gelu_bf16(%3, %3, %c2048_i32) : (memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
            aie.objectfifo.release @ffnUpOut2(Produce, 1)
          } else {
            %c0_6 = arith.constant 0 : index
            %c12_7 = arith.constant 12 : index
            %c1_8 = arith.constant 1 : index
            scf.for %arg2 = %c0_6 to %c12_7 step %c1_8 {
              %2 = aie.objectfifo.acquire @outLNBroadcast(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
              %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
              aie.objectfifo.release @outLNBroadcast(Consume, 1)
            }
          }
        }
      }
      aie.end
    } {link_with = "ep_12h_64s_64e_4ph_12pa_4g_4pf_d-1_m0_fa_n1-1_n2-1_me_c57caa9b6610_kernels.a", stack_size = 1792 : i32}
    %core_7_3 = aie.core(%tile_7_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_4 = arith.constant 0 : index
        %c12 = arith.constant 12 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c12 step %c1_5 {
          %0 = aie.objectfifo.acquire @ffnUpOut2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %c0_6 = arith.constant 0 : index
          %c12_7 = arith.constant 12 : index
          %c1_8 = arith.constant 1 : index
          scf.for %arg2 = %c0_6 to %c12_7 step %c1_8 {
            %2 = aie.objectfifo.acquire @ffnDownGroup2(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            %4 = aie.objectfifo.acquire @memBDown2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            func.call @ffn_matmul_init_bf16_bf16_down_proj(%1, %5, %3) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
            aie.objectfifo.release @memBDown2(Consume, 1)
            aie.objectfifo.release @ffnDownGroup2(Produce, 1)
          }
          aie.objectfifo.release @ffnUpOut2(Consume, 1)
        }
      }
      aie.end
    } {link_with = "ep_12h_64s_64e_4ph_12pa_4g_4pf_d-1_m0_fa_n1-1_n2-1_me_c57caa9b6610_kernels.a", stack_size = 3840 : i32}
    %core_6_4 = aie.core(%tile_6_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_4 = arith.constant 0 : index
        %c12 = arith.constant 12 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c12 step %c1_5 {
          %0 = index.casts %arg1 : index to i32
          %c12_i32 = arith.constant 12 : i32
          %1 = arith.cmpi slt, %0, %c12_i32 : i32
          scf.if %1 {
            %2 = aie.objectfifo.acquire @ffnUpOut3(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            %c0_6 = arith.constant 0 : index
            %c12_7 = arith.constant 12 : index
            %c1_8 = arith.constant 1 : index
            scf.for %arg2 = %c0_6 to %c12_7 step %c1_8 {
              %4 = aie.objectfifo.acquire @outLNBroadcast(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
              %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
              %6 = aie.objectfifo.acquire @memBUp3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %8 = index.casts %arg2 : index to i32
              %c0_i32 = arith.constant 0 : i32
              %9 = arith.cmpi eq, %8, %c0_i32 : i32
              scf.if %9 {
                func.call @ffn_matmul_init_bf16_bf16_up_proj(%5, %7, %3) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
              } else {
                func.call @ffn_matmul_bf16_bf16_up_proj(%5, %7, %3) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
              }
              aie.objectfifo.release @memBUp3(Consume, 1)
              aie.objectfifo.release @outLNBroadcast(Consume, 1)
            }
            %c2048_i32 = arith.constant 2048 : i32
            func.call @ffn_gelu_bf16(%3, %3, %c2048_i32) : (memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
            aie.objectfifo.release @ffnUpOut3(Produce, 1)
          } else {
            %c0_6 = arith.constant 0 : index
            %c12_7 = arith.constant 12 : index
            %c1_8 = arith.constant 1 : index
            scf.for %arg2 = %c0_6 to %c12_7 step %c1_8 {
              %2 = aie.objectfifo.acquire @outLNBroadcast(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
              %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
              aie.objectfifo.release @outLNBroadcast(Consume, 1)
            }
          }
        }
      }
      aie.end
    } {link_with = "ep_12h_64s_64e_4ph_12pa_4g_4pf_d-1_m0_fa_n1-1_n2-1_me_c57caa9b6610_kernels.a", stack_size = 1792 : i32}
    %core_7_4 = aie.core(%tile_7_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @ffnUpOut3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        %c0_4 = arith.constant 0 : index
        %c12 = arith.constant 12 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c12 step %c1_5 {
          aie.use_lock(%ffnDownAccum3_src_empty, AcquireGreaterEqual, 1)
          %2 = aie.objectfifo.acquire @ffnDownGroup2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %c2048_i32 = arith.constant 2048 : i32
          func.call @passThroughLine_o_proj(%3, %ffnDownAccum3_src, %c2048_i32) : (memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
          aie.objectfifo.release @ffnDownGroup2(Consume, 1)
          %4 = aie.objectfifo.acquire @memBDown3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
          func.call @ffn_matmul_with_acc_bf16_bf16_down_proj(%1, %5, %ffnDownAccum3_src, %ffnDownAccum3_src) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
          aie.objectfifo.release @memBDown3(Consume, 1)
          aie.use_lock(%ffnDownAccum3_src_full, Release, 1)
        }
        aie.objectfifo.release @ffnUpOut3(Consume, 1)
        %c0_6 = arith.constant 0 : index
        %c11 = arith.constant 11 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %c11 step %c1_7 {
          %2 = aie.objectfifo.acquire @ffnUpOut3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %c0_14 = arith.constant 0 : index
          %c12_15 = arith.constant 12 : index
          %c1_16 = arith.constant 1 : index
          scf.for %arg2 = %c0_14 to %c12_15 step %c1_16 {
            aie.use_lock(%ffnDownAccum3_dst_full, AcquireGreaterEqual, 1)
            aie.use_lock(%ffnDownAccum3_src_empty, AcquireGreaterEqual, 1)
            %4 = aie.objectfifo.acquire @ffnDownGroup2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            %c2048_i32 = arith.constant 2048 : i32
            func.call @passThroughLine_o_proj(%5, %ffnDownAccum3_src, %c2048_i32) : (memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
            aie.objectfifo.release @ffnDownGroup2(Consume, 1)
            %6 = aie.objectfifo.acquire @memBDown3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            func.call @ffn_matmul_with_acc_bf16_bf16_down_proj(%3, %7, %ffnDownAccum3_src, %ffnDownAccum3_src) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
            aie.objectfifo.release @memBDown3(Consume, 1)
            %c2048_i32_17 = arith.constant 2048 : i32
            func.call @eltwise_add_bf16_vector_o_proj(%ffnDownAccum3_src, %ffnDownAccum3_dst, %ffnDownAccum3_src, %c2048_i32_17) : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
            aie.use_lock(%ffnDownAccum3_src_full, Release, 1)
            aie.use_lock(%ffnDownAccum3_dst_empty, Release, 1)
          }
          aie.objectfifo.release @ffnUpOut3(Consume, 1)
        }
        %c0_8 = arith.constant 0 : index
        %c12_9 = arith.constant 12 : index
        %c1_10 = arith.constant 1 : index
        scf.for %arg1 = %c0_8 to %c12_9 step %c1_10 {
          %2 = aie.objectfifo.acquire @ffnDownOut1(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          aie.use_lock(%ffnDownAccum3_dst_full, AcquireGreaterEqual, 1)
          %c2048_i32 = arith.constant 2048 : i32
          func.call @passThroughLine_o_proj(%ffnDownAccum3_dst, %3, %c2048_i32) : (memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
          aie.use_lock(%ffnDownAccum3_src_empty, AcquireGreaterEqual, 1)
          %c2048_i32_14 = arith.constant 2048 : i32
          func.call @passThroughLine_o_proj(%ffnDownAccum3_dst, %ffnDownAccum3_src, %c2048_i32_14) : (memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
          aie.use_lock(%ffnDownAccum3_src_full, Release, 1)
          aie.use_lock(%ffnDownAccum3_dst_empty, Release, 1)
          aie.objectfifo.release @ffnDownOut1(Produce, 1)
        }
        %c0_11 = arith.constant 0 : index
        %c12_12 = arith.constant 12 : index
        %c1_13 = arith.constant 1 : index
        scf.for %arg1 = %c0_11 to %c12_12 step %c1_13 {
          aie.use_lock(%ffnDownAccum3_dst_full, AcquireGreaterEqual, 1)
          %2 = aie.objectfifo.acquire @ffnDownOut1(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %c2048_i32 = arith.constant 2048 : i32
          func.call @passThroughLine_o_proj(%ffnDownAccum3_dst, %3, %c2048_i32) : (memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
          aie.use_lock(%ffnDownAccum3_dst_empty, Release, 1)
          aie.objectfifo.release @ffnDownOut1(Produce, 1)
        }
      }
      aie.end
    } {link_with = "ep_12h_64s_64e_4ph_12pa_4g_4pf_d-1_m0_fa_n1-1_n2-1_me_c57caa9b6610_kernels.a", stack_size = 3840 : i32}
    %core_7_5 = aie.core(%tile_7_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c32_i32 = arith.constant 32 : i32
        func.call @ln_zero_f32(%ln2_sum_buffer, %c32_i32) : (memref<32xf32>, i32) -> ()
        %c32_i32_4 = arith.constant 32 : i32
        func.call @ln_zero_f32(%ln2_sumsq_buffer, %c32_i32_4) : (memref<32xf32>, i32) -> ()
        %c0_5 = arith.constant 0 : index
        %c12 = arith.constant 12 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_5 to %c12 step %c1_6 {
          %0 = aie.objectfifo.acquire @ffnDownOut0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %2 = aie.objectfifo.acquire @ffnDownOut1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %c2048_i32 = arith.constant 2048 : i32
          func.call @eltwise_add_bf16_vector_o_proj(%1, %3, %ln2_ffn_merge_buffer, %c2048_i32) : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
          func.call @ln_calc_sum_sumsq(%ln2_ffn_merge_buffer, %ln2_sum_buffer, %ln2_sumsq_buffer) : (memref<32x64xbf16>, memref<32xf32>, memref<32xf32>) -> ()
          aie.objectfifo.release @ffnDownOut1(Consume, 1)
          aie.objectfifo.release @ffnDownOut0(Consume, 1)
        }
        %c0_7 = arith.constant 0 : index
        %c12_8 = arith.constant 12 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c12_8 step %c1_9 {
          %0 = index.casts %arg1 : index to i32
          %1 = aie.objectfifo.acquire @ffnDownOut0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %2 = aie.objectfifo.subview.access %1[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %3 = aie.objectfifo.acquire @ffnDownOut1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %c2048_i32 = arith.constant 2048 : i32
          func.call @eltwise_add_bf16_vector_o_proj(%2, %4, %ln2_ffn_merge_buffer, %c2048_i32) : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>, i32) -> ()
          %5 = aie.objectfifo.acquire @ffnRIn(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %7 = aie.objectfifo.acquire @outLN2(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          %c768_i32 = arith.constant 768 : i32
          func.call @fused_add_layer_norm_1outs(%ln2_ffn_merge_buffer, %6, %static_ln2_weights, %ln2_sum_buffer, %ln2_sumsq_buffer, %8, %c768_i32, %0) : (memref<32x64xbf16>, memref<32x64xbf16>, memref<768xbf16>, memref<32xf32>, memref<32xf32>, memref<32x64xbf16>, i32, i32) -> ()
          aie.objectfifo.release @outLN2(Produce, 1)
          aie.objectfifo.release @ffnDownOut1(Consume, 1)
          aie.objectfifo.release @ffnDownOut0(Consume, 1)
          aie.objectfifo.release @ffnRIn(Consume, 1)
        }
      }
      aie.end
    } {link_with = "ep_12h_64s_64e_4ph_12pa_4g_4pf_d-1_m0_fa_n1-1_n2-1_me_c57caa9b6610_kernels.a", stack_size = 3840 : i32}
    aie.runtime_sequence(%arg0: memref<768x768xbf16>, %arg1: memref<192x768xbf16>, %arg2: memref<128x768xbf16>, %arg3: memref<2359296xbf16>, %arg4: memref<2359296xbf16>) {
      %0 = aiex.dma_configure_task_for @inQ {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 0, 24576, [<size = 1, stride = 0>, <size = 12, stride = 64>, <size = 32, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      aiex.dma_free_task(%0)
      %1 = aiex.dma_configure_task_for @inK {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 49152, 16384, [<size = 1, stride = 0>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @inV {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 98304, 16384, [<size = 1, stride = 0>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @inOW {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 0, 16384, [<size = 12, stride = 64>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 11 : i32}
      aiex.dma_start_task(%3)
      aiex.dma_await_task(%1)
      aiex.dma_await_task(%2)
      aiex.dma_await_task(%3)
      %4 = aiex.dma_configure_task_for @inK {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 49408, 16384, [<size = 1, stride = 0>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @inV {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 98560, 16384, [<size = 1, stride = 0>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @inOW {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 196608, 16384, [<size = 12, stride = 64>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 11 : i32}
      aiex.dma_start_task(%6)
      aiex.dma_await_task(%4)
      aiex.dma_await_task(%5)
      aiex.dma_await_task(%6)
      %7 = aiex.dma_configure_task_for @inK {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 49664, 16384, [<size = 1, stride = 0>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @inV {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 98816, 16384, [<size = 1, stride = 0>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @inOW {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 393216, 16384, [<size = 12, stride = 64>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 11 : i32}
      aiex.dma_start_task(%9)
      aiex.dma_await_task(%7)
      aiex.dma_await_task(%8)
      aiex.dma_await_task(%9)
      %10 = aiex.dma_configure_task_for @inR {
        aie.dma_bd(%arg2 : memref<128x768xbf16>, 49152, 24576, [<size = 12, stride = 0>, <size = 12, stride = 64>, <size = 32, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 11 : i32}
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @inBUp {
        aie.dma_bd(%arg3 : memref<2359296xbf16>, 0, 49152, [<size = 12, stride = 64>, <size = 12, stride = 196608>, <size = 64, stride = 3072>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 11 : i32}
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @inBUp1 {
        aie.dma_bd(%arg3 : memref<2359296xbf16>, 768, 49152, [<size = 12, stride = 64>, <size = 12, stride = 196608>, <size = 64, stride = 3072>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 11 : i32}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @inBUp2 {
        aie.dma_bd(%arg3 : memref<2359296xbf16>, 1536, 49152, [<size = 12, stride = 64>, <size = 12, stride = 196608>, <size = 64, stride = 3072>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 11 : i32}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @inBUp3 {
        aie.dma_bd(%arg3 : memref<2359296xbf16>, 2304, 49152, [<size = 12, stride = 64>, <size = 12, stride = 196608>, <size = 64, stride = 3072>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 11 : i32}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @inBDown3 {
        aie.dma_bd(%arg4 : memref<2359296xbf16>, 1769472, 49152, [<size = 12, stride = 49152>, <size = 12, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 11 : i32}
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @inBDown2 {
        aie.dma_bd(%arg4 : memref<2359296xbf16>, 1179648, 49152, [<size = 12, stride = 49152>, <size = 12, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 11 : i32}
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @inBDown1 {
        aie.dma_bd(%arg4 : memref<2359296xbf16>, 589824, 49152, [<size = 12, stride = 49152>, <size = 12, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 11 : i32}
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @inBDown {
        aie.dma_bd(%arg4 : memref<2359296xbf16>, 0, 49152, [<size = 12, stride = 49152>, <size = 12, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 11 : i32}
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @memLN2 {
        aie.dma_bd(%arg2 : memref<128x768xbf16>, 0, 24576, [<size = 1, stride = 0>, <size = 12, stride = 64>, <size = 32, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%19)
      aiex.dma_await_task(%19)
      aiex.dma_await_task(%10)
      aiex.dma_free_task(%11)
      aiex.dma_free_task(%12)
      aiex.dma_free_task(%13)
      aiex.dma_free_task(%14)
      aiex.dma_free_task(%15)
      aiex.dma_free_task(%16)
      aiex.dma_free_task(%17)
      aiex.dma_free_task(%18)
      %20 = aiex.dma_configure_task_for @inQ {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 24576, 24576, [<size = 1, stride = 0>, <size = 12, stride = 64>, <size = 32, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%20)
      aiex.dma_free_task(%20)
      %21 = aiex.dma_configure_task_for @inK {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 49152, 16384, [<size = 1, stride = 0>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @inV {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 98304, 16384, [<size = 1, stride = 0>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @inOW {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 0, 16384, [<size = 12, stride = 64>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 11 : i32}
      aiex.dma_start_task(%23)
      aiex.dma_await_task(%21)
      aiex.dma_await_task(%22)
      aiex.dma_await_task(%23)
      %24 = aiex.dma_configure_task_for @inK {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 49408, 16384, [<size = 1, stride = 0>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%24)
      %25 = aiex.dma_configure_task_for @inV {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 98560, 16384, [<size = 1, stride = 0>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @inOW {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 196608, 16384, [<size = 12, stride = 64>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 11 : i32}
      aiex.dma_start_task(%26)
      aiex.dma_await_task(%24)
      aiex.dma_await_task(%25)
      aiex.dma_await_task(%26)
      %27 = aiex.dma_configure_task_for @inK {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 49664, 16384, [<size = 1, stride = 0>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%27)
      %28 = aiex.dma_configure_task_for @inV {
        aie.dma_bd(%arg1 : memref<192x768xbf16>, 98816, 16384, [<size = 1, stride = 0>, <size = 4, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%28)
      %29 = aiex.dma_configure_task_for @inOW {
        aie.dma_bd(%arg0 : memref<768x768xbf16>, 393216, 16384, [<size = 12, stride = 64>, <size = 4, stride = 49152>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 11 : i32}
      aiex.dma_start_task(%29)
      aiex.dma_await_task(%27)
      aiex.dma_await_task(%28)
      aiex.dma_await_task(%29)
      %30 = aiex.dma_configure_task_for @inR {
        aie.dma_bd(%arg2 : memref<128x768xbf16>, 73728, 24576, [<size = 12, stride = 0>, <size = 12, stride = 64>, <size = 32, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 11 : i32}
      aiex.dma_start_task(%30)
      %31 = aiex.dma_configure_task_for @inBUp {
        aie.dma_bd(%arg3 : memref<2359296xbf16>, 0, 49152, [<size = 12, stride = 64>, <size = 12, stride = 196608>, <size = 64, stride = 3072>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 11 : i32}
      aiex.dma_start_task(%31)
      %32 = aiex.dma_configure_task_for @inBUp1 {
        aie.dma_bd(%arg3 : memref<2359296xbf16>, 768, 49152, [<size = 12, stride = 64>, <size = 12, stride = 196608>, <size = 64, stride = 3072>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 11 : i32}
      aiex.dma_start_task(%32)
      %33 = aiex.dma_configure_task_for @inBUp2 {
        aie.dma_bd(%arg3 : memref<2359296xbf16>, 1536, 49152, [<size = 12, stride = 64>, <size = 12, stride = 196608>, <size = 64, stride = 3072>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 11 : i32}
      aiex.dma_start_task(%33)
      %34 = aiex.dma_configure_task_for @inBUp3 {
        aie.dma_bd(%arg3 : memref<2359296xbf16>, 2304, 49152, [<size = 12, stride = 64>, <size = 12, stride = 196608>, <size = 64, stride = 3072>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 11 : i32}
      aiex.dma_start_task(%34)
      %35 = aiex.dma_configure_task_for @inBDown3 {
        aie.dma_bd(%arg4 : memref<2359296xbf16>, 1769472, 49152, [<size = 12, stride = 49152>, <size = 12, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 11 : i32}
      aiex.dma_start_task(%35)
      %36 = aiex.dma_configure_task_for @inBDown2 {
        aie.dma_bd(%arg4 : memref<2359296xbf16>, 1179648, 49152, [<size = 12, stride = 49152>, <size = 12, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 11 : i32}
      aiex.dma_start_task(%36)
      %37 = aiex.dma_configure_task_for @inBDown1 {
        aie.dma_bd(%arg4 : memref<2359296xbf16>, 589824, 49152, [<size = 12, stride = 49152>, <size = 12, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 11 : i32}
      aiex.dma_start_task(%37)
      %38 = aiex.dma_configure_task_for @inBDown {
        aie.dma_bd(%arg4 : memref<2359296xbf16>, 0, 49152, [<size = 12, stride = 49152>, <size = 12, stride = 64>, <size = 64, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 11 : i32}
      aiex.dma_start_task(%38)
      %39 = aiex.dma_configure_task_for @memLN2 {
        aie.dma_bd(%arg2 : memref<128x768xbf16>, 24576, 24576, [<size = 1, stride = 0>, <size = 12, stride = 64>, <size = 32, stride = 768>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%39)
      aiex.dma_await_task(%39)
      aiex.dma_await_task(%30)
      aiex.dma_free_task(%31)
      aiex.dma_free_task(%32)
      aiex.dma_free_task(%33)
      aiex.dma_free_task(%34)
      aiex.dma_free_task(%35)
      aiex.dma_free_task(%36)
      aiex.dma_free_task(%37)
      aiex.dma_free_task(%38)
    }
    %mem_3_5 = aie.mem(%tile_3_5) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%outOProjAccum3_src_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOProjAccum3_src : memref<32x64xbf16>, 0, 2048)
      aie.use_lock(%outOProjAccum3_src_empty, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%outOProjAccum3_dst_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOProjAccum3_dst : memref<32x64xbf16>, 0, 2048)
      aie.use_lock(%outOProjAccum3_dst_full, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1_0) {
      %0 = aie.dma_start(S2MM, 5, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%outOProjAccum3_row_0_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOProjAccum3_row_0 : memref<24576xbf16>, 0, 24576)
      aie.use_lock(%outOProjAccum3_row_0_full, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%outOProjAccum3_row_1_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOProjAccum3_row_1 : memref<24576xbf16>, 0, 24576)
      aie.use_lock(%outOProjAccum3_row_1_full, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 5, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%outOProjAccum3_row_0_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOProjAccum3_row_0 : memref<24576xbf16>, 0, 24576)
      aie.use_lock(%outOProjAccum3_row_0_empty, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%outOProjAccum3_row_1_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOProjAccum3_row_1 : memref<24576xbf16>, 0, 24576)
      aie.use_lock(%outOProjAccum3_row_1_empty, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    %mem_6_5 = aie.mem(%tile_6_5) {
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
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1_1) {
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
    %mem_6_3 = aie.mem(%tile_6_3) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%ffnDownAccum1_src_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownAccum1_src : memref<32x64xbf16>, 0, 2048)
      aie.use_lock(%ffnDownAccum1_src_empty, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%ffnDownAccum1_dst_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownAccum1_dst : memref<32x64xbf16>, 0, 2048)
      aie.use_lock(%ffnDownAccum1_dst_full, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %memtile_dma_5_1 = aie.memtile_dma(%mem_tile_5_1_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%ffnDownAccum1_row_0_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownAccum1_row_0 : memref<24576xbf16>, 0, 24576)
      aie.use_lock(%ffnDownAccum1_row_0_full, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%ffnDownAccum1_row_1_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownAccum1_row_1 : memref<24576xbf16>, 0, 24576)
      aie.use_lock(%ffnDownAccum1_row_1_full, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%ffnDownAccum1_row_0_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownAccum1_row_0 : memref<24576xbf16>, 0, 24576)
      aie.use_lock(%ffnDownAccum1_row_0_empty, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%ffnDownAccum1_row_1_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownAccum1_row_1 : memref<24576xbf16>, 0, 24576)
      aie.use_lock(%ffnDownAccum1_row_1_empty, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    %mem_7_4 = aie.mem(%tile_7_4) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%ffnDownAccum3_src_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownAccum3_src : memref<32x64xbf16>, 0, 2048)
      aie.use_lock(%ffnDownAccum3_src_empty, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%ffnDownAccum3_dst_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownAccum3_dst : memref<32x64xbf16>, 0, 2048)
      aie.use_lock(%ffnDownAccum3_dst_full, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %memtile_dma_7_1 = aie.memtile_dma(%mem_tile_7_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%ffnDownAccum3_row_0_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownAccum3_row_0 : memref<24576xbf16>, 0, 24576)
      aie.use_lock(%ffnDownAccum3_row_0_full, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%ffnDownAccum3_row_1_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownAccum3_row_1 : memref<24576xbf16>, 0, 24576)
      aie.use_lock(%ffnDownAccum3_row_1_full, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%ffnDownAccum3_row_0_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownAccum3_row_0 : memref<24576xbf16>, 0, 24576)
      aie.use_lock(%ffnDownAccum3_row_0_empty, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%ffnDownAccum3_row_1_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffnDownAccum3_row_1 : memref<24576xbf16>, 0, 24576)
      aie.use_lock(%ffnDownAccum3_row_1_empty, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
  }
}

