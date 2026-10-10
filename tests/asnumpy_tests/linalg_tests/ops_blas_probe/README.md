# ops-blas 前置探针（issue #28 第三阶段）

实现之前用于核查 Ascend 910B(arch22) 上哪些 GEMM 入口真实可用，以及行主序→列主序的
交换操作数映射是否正确。结果已记录在 `docs/matmul_backend.md`。

```bash
# 1) 构建 ops-blas（910B 只需 gemm_strided_batched；只编库时关闭测试目标）
git clone https://gitcode.com/cann/ops-blas.git && cd ops-blas
cmake -S . -B build-lib -DSOC_VERSION=ascend910b3 -DBUILD_TEST=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-lib -j

# 2) 编译并运行探针
g++ -std=c++17 -O2 opsblas_gemm_probe.cpp -o gemm_probe \
    -I<ops-blas>/include -I${ASCEND_HOME_PATH}/include \
    -L<ops-blas>/build-lib -L${ASCEND_HOME_PATH}/lib64 -lops_blas -lascendcl -ldl
LD_LIBRARY_PATH=<ops-blas>/build-lib:${ASCEND_HOME_PATH}/lib64 ./gemm_probe

# 3) 边界/写入位置核查（可选）
g++ -std=c++17 -O2 opsblas_boundary.cpp -o boundary \
    -I<ops-blas>/include -I${ASCEND_HOME_PATH}/include \
    -L<ops-blas>/build-lib -L${ASCEND_HOME_PATH}/lib64 -lops_blas -lascendcl
LD_LIBRARY_PATH=<ops-blas>/build-lib:${ASCEND_HOME_PATH}/lib64 ./boundary 0
```

期望：`aclblasSgemm`/`aclblasGemmEx`/`aclblasSgemmBatched` ABSENT，
`aclblasSgemmStridedBatched` PRESENT，非方阵全部 `rel err < 1e-6`、`inputs_intact=yes`，
`SUMMARY: PROBE_OK`。

注意：kernel 提交到 handle 的 stream，**读回结果前必须同步该 stream 或整设备**
（`aclrtSynchronizeDevice()`）；只同步默认 stream 会读到未写入的 C。
