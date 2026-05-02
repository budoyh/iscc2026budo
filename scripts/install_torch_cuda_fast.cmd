@echo off
setlocal

set "ROOT=%~dp0.."
set "PYTHON=C:\Users\Asus\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
set "TARGET=%ROOT%\.deps\torch-cu126-direct"
set "PIPTMP=%ROOT%\.tmp\piptemp"

if not exist "%PIPTMP%" mkdir "%PIPTMP%"
if not exist "%TARGET%" mkdir "%TARGET%"

set "TEMP=%PIPTMP%"
set "TMP=%PIPTMP%"

"%PYTHON%" -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
"%PYTHON%" -m pip install --upgrade --target "%TARGET%" ^
  torch==2.7.0+cu126 torchvision==0.22.0+cu126 torchaudio==2.7.0+cu126 ^
  --find-links https://mirrors.aliyun.com/pytorch-wheels/cu126/ ^
  -i https://pypi.tuna.tsinghua.edu.cn/simple ^
  --trusted-host mirrors.aliyun.com ^
  --trusted-host pypi.tuna.tsinghua.edu.cn ^
  --timeout 120 ^
  --retries 5

echo.
echo PyTorch CUDA packages installed into:
echo   %TARGET%
echo.
echo Verify with:
echo   scripts\check_torch_cuda.cmd
