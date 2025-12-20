#!/bin/bash
rm -r build/
pytest operators/gemm/
pytest operators/transpose/
pytest operators/elementwise_mul/
pytest operators/elementwise_add/
pytest operators/softmax/
pytest operators/layer_norm/