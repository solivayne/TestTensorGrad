# build and run

构建项目与运行

```bash
mkdir build && cd build
cmake ../TensorGrad
make -j$(nproc)
```
# run examples

```bash
./train_circle # 训练三层全连接网络，判断二维点是否在圆内
./train_xor # 训练 XOR 模型
./train_bench # 代码自带 bench
```

# run tests

```bash
./Test/tests # run tests suites
```

# apply diff

通过 apply diff 撤销修改，从而运行测试 suites 查看报错结果

```bash
git apply ../patch.diff # 撤销所有修复修改
git apply -R ../patch.diff # 恢复为修复后的状态
```