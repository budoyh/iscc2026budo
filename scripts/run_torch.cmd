@echo off
setlocal

set "ROOT=%~dp0.."
set "PYTHONPATH=%ROOT%\.deps\torch-cu126-direct;%ROOT%\.deps\py312"
set "TEMP=%ROOT%\.tmp\pytemp"
set "TMP=%ROOT%\.tmp\pytemp"

if not exist "%TEMP%" mkdir "%TEMP%"

"C:\Users\Asus\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" %*
