# ISCC2026 数据安全赛准备方案

## 一、必须记住的硬要求

- 数据安全赛按“每道赛题”分别排名，不是只做一道总题。
- 每道赛题每天最多提交 5 次。
- 榜单分为 A 榜和 B 榜。
- A 榜显示历史最优成绩，只用于过程参考。
- B 榜在比赛结束后显示，并以“最后一次提交的预测结果”作为最终成绩依据。
- 比赛结束后 24 小时内，必须把模型、代码、数据集等材料发送到 `iscc2026_bigdata@163.com`。
- 不提交材料、材料跑不通、复现结果和线上结果差距过大、材料与他队高度雷同，都会按违规处理。
- 单题分没超过官方基线值，不参与最终评奖。基线值将在比赛中期发布。
- 只能使用官方提供的数据集，禁止引入外部数据，包括网络爬虫、第三方 API、购买数据、公开数据集、互联网补充标注等。
- 允许数据生成、模拟、扩充，但必须基于官方数据、合理假设和可复现算法，不能破坏公平性和真实性。

## 二、赛后要交什么

根据官方“模型提交模板”，至少要准备：

- `提交结果（必交）`
- `模型（必交）`
- `源码（必交）`
- `docker容器（选交）`
- `README.md`
- `requirements.txt`

README 至少要覆盖：

- 项目说明
- 环境配置
- 模型设置
- 训练过程
- 预测过程

## 三、我们现在就要搭好的目录

每道数据安全题单独建目录，建议结构：

```text
work/data_security/<题目名>/
  raw/
  processed/
  notebooks/
  src/
  models/
  submissions/
  logs/
  report/
```

其中：

- `raw/`：原始训练集、测试集、样例提交，禁止覆盖。
- `processed/`：清洗后数据、特征缓存。
- `src/`：训练、验证、预测、特征工程脚本。
- `models/`：模型权重、参数、折内模型。
- `submissions/`：每次提交文件，文件名带时间戳和分数。
- `logs/`：实验日志、参数、随机种子、线上分记录。
- `report/`：最终 README、复现说明、截图。

## 四、比赛中的标准工作流

### 1. 开题后 30 分钟内

- 读题面，确认任务类型：分类、回归、检测、排序、生成。
- 确认评价指标：AUC、F1、RMSE、LogLoss、Accuracy、Recall、MAP 等。
- 确认提交格式：列名、行数、id 顺序、文件格式。
- 建一个“合法 baseline”，先确保能成功提交。

### 2. 第一阶段：先拿到稳定可提交流水线

- 写 `load_data.py` 或统一数据读取函数。
- 写最小可跑特征处理。
- 用一个最稳的 baseline 模型先出第一版结果：
  - 表格题优先 `LightGBM / XGBoost / CatBoost`
  - 文本题优先 `TF-IDF + Linear / LightGBM`
  - 时序题优先简单窗口统计 + GBDT
- 先把提交格式打通，再谈提分。

### 3. 第二阶段：严格管理每天 5 次提交

每天 5 次非常少，不能乱试。建议：

- 第 1 次：验证提交流水线正常。
- 第 2 次：单模型强化版。
- 第 3 次：特征工程或参数明显改进版。
- 第 4 次：融合版。
- 第 5 次：保底或最终版。

禁止为了试格式浪费后 3 次。

### 4. 第三阶段：围绕“最终一次提交”做策略

B 榜按最后一次提交计分，所以：

- 不要把明显未验证的结果作为当天最后一次提交。
- 每天留 1 次给“保底版本”。
- 如果新模型不确定优于当前最优，不要在临近结束时梭哈。

## 五、我们实际怎么做

我建议你把数据安全赛按下面顺序推进：

1. 先做最容易建立 baseline 的题。
2. 每题先拿可复现的第一版，不要上来就卷复杂模型。
3. 表格题优先树模型，文本题优先稀疏特征或轻量预训练，异常检测题先做统计特征和阈值法。
4. 每次线上提交前，先把本地输出文件行数、列名、空值、排序检查一遍。

## 六、每次实验必须记录

- 题目名
- 数据版本
- 代码版本
- 特征集合
- 模型与参数
- 随机种子
- 本地验证分数
- 提交文件名
- 提交时间
- A 榜分数
- 备注

## 七、赛后材料不能翻车的最低标准

赛后你至少要保证：

- `train.py` 能训练出模型并保存。
- `test.py` 能加载模型并生成与线上同格式的预测文件。
- `requirements.txt` 可安装依赖。
- README 写清从原始数据到最终提交文件的完整命令。
- 模型文件、源码、提交结果一一对应。
- 如果使用数据增强或模拟数据，README 必须写清生成方法、输入来源、随机种子和合理性。

## 八、当前可用环境

Windows 侧已经准备好两套互不污染的项目隔离环境：

- 常规机器学习环境：`.deps\py312`，入口是 `scripts\run_py.cmd`。
- 深度学习 GPU 环境：`.deps\torch-cu126-direct`，入口是 `scripts\run_torch.cmd`。
- 当前深度学习版本：`torch==2.7.0+cu126`、`torchvision==0.22.0+cu126`、`torchaudio==2.7.0+cu126`。
- 当前硬件：NVIDIA GeForce RTX 4060 Laptop，驱动 566.07，`nvidia-smi` 显示 CUDA 12.7；PyTorch 使用自带 CUDA 12.6 runtime，不需要单独安装 CUDA Toolkit。
- GPU 验证命令：

```powershell
scripts\check_torch_cuda.cmd
```

正确状态应包含：

```text
torch: 2.7.0+cu126
cuda available: True
device: NVIDIA GeForce RTX 4060 Laptop GPU
```

运行普通机器学习脚本：

```powershell
scripts\run_py.cmd work\data_security\题目名\src\train.py
scripts\run_py.cmd work\data_security\题目名\src\predict.py
```

运行 PyTorch / GPU 深度学习脚本：

```powershell
scripts\run_torch.cmd work\data_security\题目名\src\train.py
scripts\run_torch.cmd work\data_security\题目名\src\predict.py
```

### 环境更新方式

- 普通依赖更新：先改 `requirements-data.txt`，再用项目 Python 安装到 `.deps\py312`；不要装进 Anaconda 或全局 `site-packages`。
- PyTorch CUDA 更新：先改 `requirements-torch-cu126.txt`，再运行 `scripts\install_torch_cuda.cmd` 或国内源脚本 `scripts\install_torch_cuda_fast.cmd`。
- 国内源规则：普通 PyPI 依赖走清华源；PyTorch CUDA 大包走阿里云 `https://mirrors.aliyun.com/pytorch-wheels/cu126/`，并且必须锁定 `+cu126`，避免误装 CPU 版。
- 更新后必须运行 `scripts\check_torch_cuda.cmd`，确认 `torch.__version__` 带 `+cu126` 且 `cuda available: True`。
- 如果更新后出现 `torch: ...+cpu`，说明装错包；不要继续训练，重新检查安装源和版本锁定。

普通依赖更新命令：

```powershell
"C:\Users\Asus\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m pip install --upgrade --target ".deps\py312" -r requirements-data.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

PyTorch CUDA 国内源更新命令：

```powershell
scripts\install_torch_cuda_fast.cmd
scripts\check_torch_cuda.cmd
```

## 九、当前建议

你现在不要先想“高分模型”，先准备这三件事：

1. 固定每道题的目录结构。
2. 准备实验记录表。
3. 准备统一的训练、预测脚本骨架。

等你把第一道数据安全题的题面、数据文件名、样例提交和评价指标给我，我就直接给你建 baseline。
