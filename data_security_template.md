# 数据安全赛题目名称

## 项目说明

简述题目任务、数据含义、预测目标、评价指标和提交格式。

## 环境配置

- 操作系统：Windows / WSL / Docker
- Python 版本：项目 Python 3.12.13
- 常规机器学习入口：`scripts\run_py.cmd`
- 常规依赖目录：`C:\budostudy\only_for_codex\iscc\.deps\py312`
- 深度学习 GPU 入口：`scripts\run_torch.cmd`
- 深度学习依赖目录：`C:\budostudy\only_for_codex\iscc\.deps\torch-cu126-direct`
- 深度学习依赖：`torch==2.7.0+cu126`，`torchvision==0.22.0+cu126`，`torchaudio==2.7.0+cu126`
- 硬件环境：NVIDIA GeForce RTX 4060 Laptop，驱动 566.07，PyTorch CUDA runtime 12.6

GPU 验证命令：

```powershell
scripts\check_torch_cuda.cmd
```

验证通过标准：`torch` 版本带 `+cu126`，`cuda available: True`，设备名显示 NVIDIA GeForce RTX 4060 Laptop GPU。

环境更新方式：

- 普通依赖：修改根目录 `requirements-data.txt`，安装到 `.deps\py312`，不要安装到 Anaconda 或全局 Python。
- PyTorch CUDA 依赖：修改根目录 `requirements-torch-cu126.txt`，然后运行 `scripts\install_torch_cuda.cmd` 或 `scripts\install_torch_cuda_fast.cmd`。
- 国内源注意：普通 PyPI 依赖走清华源；PyTorch CUDA 大包走阿里云 `pytorch-wheels/cu126`；版本必须带 `+cu126`，否则可能误装 CPU 版。
- 更新后必须重新运行 `scripts\check_torch_cuda.cmd` 并把结果记录到本文档。

普通依赖更新命令：

```powershell
"C:\Users\Asus\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m pip install --upgrade --target ".deps\py312" -r requirements-data.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

PyTorch CUDA 国内源更新命令：

```powershell
scripts\install_torch_cuda_fast.cmd
scripts\check_torch_cuda.cmd
```

## 数据文件

- 训练集：
- 测试集：
- 样例提交：
- 外部数据：无。本次竞赛禁止引入外部数据，包括网络爬虫、第三方 API、购买数据、公开数据集、互联网补充标注等。
- 数据增强/模拟数据：无 / 有，方法、输入来源、随机种子：

## 数据处理

说明训练集、测试集、缺失值、异常值、特征工程和数据划分方式。

## 模型设置

- 模型：
- 参数：
- 随机种子：
- 验证方案：

## 训练过程

普通机器学习：

```powershell
scripts\run_py.cmd work\data_security\题目名\src\train.py
```

深度学习 / GPU：

```powershell
scripts\run_torch.cmd work\data_security\题目名\src\train.py
```

## 预测与提交

普通机器学习：

```powershell
scripts\run_py.cmd work\data_security\题目名\src\predict.py
```

深度学习 / GPU：

```powershell
scripts\run_torch.cmd work\data_security\题目名\src\predict.py
```

说明生成的提交文件路径和格式校验结果。

## 结果

- 本地验证分数：
- A 榜分数：
- B 榜分数：
- 最终提交文件：

## 复现步骤

从原始数据到最终提交文件的完整命令序列。
