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

"%PYTHON%" -m pip install --upgrade pip
"%PYTHON%" -m pip install --upgrade --target "%TARGET%" -r "%ROOT%\requirements-torch-cu126.txt"

echo.
echo PyTorch CUDA packages installed into:
echo   %TARGET%
echo.
echo Verify with:
echo   scripts\check_torch_cuda.cmd
