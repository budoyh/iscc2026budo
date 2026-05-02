# ISCC2026 本地协作最高优先级说明

本目录用于 ISCC2026 线上挑战赛备赛和赛时协作。任何 AI、自动化代理或脚本在本目录工作时，必须优先遵守本文档。

## 语言与沟通

- 默认使用简体中文。
- 回答要直接、可执行，少讲空泛背景。
- 做题时优先给出可复现命令、脚本、判断依据和下一步。
- 不要把未经验证的猜测当成结论。

## 环境边界

- 不要修改系统级 Python、Anaconda、PATH、注册表或用户级全局配置。
- 不要向 `C:\Anaconda3`、系统目录或用户全局 `site-packages` 安装包。
- 不要默认使用 `.venv`；它可能是创建失败后的残留环境。
- Windows 侧隔离 Python 依赖目录是 `C:\budostudy\only_for_codex\iscc\.deps\py312`。
- Windows 侧推荐运行入口是 `scripts\run_py.cmd`，例如：

```cmd
scripts\run_py.cmd -c "import pandas, sklearn, pwn; print('ok')"
```

- PWN、深度逆向、Docker 靶场优先使用 WSL2 Ubuntu 环境，不要塞进 Windows 全局环境。
- WSL/Docker/系统级工具需要管理员权限、联网下载和可能重启，必须由用户明确执行或批准。

## 赛规红线

- 只分析比赛题目明确给出的附件、靶机地址、API、下载链接和题目范围。
- 不攻击比赛平台本身、登录系统、非题目目标或第三方站点。
- 不做高强度端口扫描、目录爆破、压力测试或可能造成服务器异常的操作。
- 不共享、传播或请求他人 flag、解题思路、writeup。
- 不伪造过程，不提交无法复现、非原创、与他人高度雷同的材料。
- 题目 writeup、数据安全赛模型、代码、结果必须能复现。

## 破阵夺旗赛工作方式

每道题单独建目录，建议结构：

```text
work/ctf/<category>/<challenge>/
  attachments/
  scripts/
  notes/
  writeup.md
```

创建目录优先使用：

```cmd
scripts\new_challenge.cmd WEB 题目名
```

解题优先级：

1. CHOICE
2. WEB
3. MISC
4. Crypto
5. Reverse / MOBILE
6. PWN

每题必须记录题名、类别、分值、题面原文、附件路径或靶机地址、实际执行命令、关键发现、最终脚本、flag 来源和提交时间。

## 靶机地址处理

- 如果题目给的是靶机地址，可以把地址交给 AI 分析，但必须确认这是比赛题目内的目标。
- 允许做低强度、针对性的 HTTP 请求、页面查看、参数测试和题目要求的交互。
- 不默认做全端口扫描、大规模目录爆破、撞库、弱口令枚举、压力测试。
- 如果需要扫描或爆破，必须先说明强度、范围、速率和理由。

## WSL / Docker 方案

系统级保障方案见 `environment_setup_plan.md`。

WSL 内工具链脚本：

```bash
cd /mnt/c/budostudy/only_for_codex/iscc
bash scripts/setup_wsl_ctf.sh
bash scripts/install_docker_wsl.sh
bash scripts/verify_wsl_ctf.sh
```

做 PWN/逆向题前，在 WSL 里先激活虚拟环境：

```bash
source ~/ctf-venv/bin/activate
```

## 数据安全赛工作方式

- 数据安全赛按赛题分别处理，不要假设只有一道题。
- 保留原始数据，不覆盖；清洗、特征、训练、预测、提交文件分目录保存。
- 每次提交记录参数、随机种子、验证分数、线上分数和生成命令。
- 普通机器学习使用 `scripts\run_py.cmd` 和 `.deps\py312`。
- 深度学习 / GPU 训练使用 `scripts\run_torch.cmd` 和 `.deps\torch-cu126-direct`；当前已验证 `torch==2.7.0+cu126` 可用。
- 使用 PyTorch 前先运行 `scripts\check_torch_cuda.cmd`，确认 `cuda available: True` 且版本带 `+cu126`。
- 更新普通依赖先改 `requirements-data.txt`；更新 PyTorch CUDA 依赖先改 `requirements-torch-cu126.txt`，再运行 `scripts\install_torch_cuda.cmd` 或 `scripts\install_torch_cuda_fast.cmd`，更新后必须重新验证 CUDA。
- 赛后材料使用 `data_security_template.md` 补齐环境、代码、模型和复现说明。
