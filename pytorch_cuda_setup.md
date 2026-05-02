# PyTorch CUDA 环境说明

## 当前机器判断

- GPU：NVIDIA GeForce RTX 4060 Laptop，显存约 8GB。
- 驱动：566.07。
- `nvidia-smi` 显示 CUDA Version：12.7。
- 项目 Python：3.12.13。

## 安装策略

使用项目隔离目录安装 PyTorch，不动全局 Conda，也不改系统 PATH。

选择 PyTorch `cu126`：

- 本项目当前锁定 PyTorch 2.7.0，配套 `torchvision==0.22.0`、`torchaudio==2.7.0`。
- 官方 Windows + pip 支持 CUDA 12.6 和 12.8。
- 本机驱动显示 CUDA 12.7，因此优先选择 CUDA 12.6 轮子，兼容性更稳。

安装位置：

```text
C:\budostudy\only_for_codex\iscc\.deps\torch-cu126-direct
```

安装命令：

```cmd
scripts\install_torch_cuda.cmd
```

国内源安装命令：

```cmd
scripts\install_torch_cuda_fast.cmd
```

说明：

- PyTorch CUDA 大包走阿里云 `pytorch-wheels/cu126`。
- 普通 PyPI 依赖走清华 `https://pypi.tuna.tsinghua.edu.cn/simple`。
- 我实测清华 `pytorch-wheels/cu126` 当前返回 404，所以不把 PyTorch CUDA 大包指向清华。
- 国内源脚本使用阿里云的扁平 wheel 列表和 `+cu126` 版本锁定，避免 pip 从普通 PyPI 源误装 CPU 版。

验证命令：

```cmd
scripts\check_torch_cuda.cmd
```

运行深度学习脚本：

```cmd
scripts\run_torch.cmd work\data_security\题目名\src\train.py
scripts\run_torch.cmd work\data_security\题目名\src\predict.py
```

## 注意事项

- 不需要单独安装 CUDA Toolkit，PyTorch pip 轮子自带 CUDA runtime；只需要 NVIDIA 驱动正常。
- RTX 4060 Laptop 约 8GB 显存，训练时默认从小 batch size 开始。
- 数据安全赛仍然必须遵守“只能使用官方数据集”的规则。预训练权重、外部语料、第三方 API、爬虫数据都不要默认使用，除非官方题面明确允许。
- 如果题目只是表格数据，仍优先用 LightGBM / XGBoost / CatBoost；PyTorch 用于深度表征、文本、图像、序列或传统模型明显不够的场景。
- 更新 PyTorch CUDA 环境时，先修改 `requirements-torch-cu126.txt`，再运行 `scripts\install_torch_cuda.cmd` 或 `scripts\install_torch_cuda_fast.cmd`。更新后必须运行 `scripts\check_torch_cuda.cmd`；如果版本显示 `+cpu`，说明装错包，不能继续用该环境训练。
