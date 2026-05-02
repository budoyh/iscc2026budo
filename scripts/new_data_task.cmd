@echo off
set "ROOT=%~dp0.."
set "PYTHONPATH=%ROOT%\.deps\py312"
"C:\Users\Asus\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" "%~dp0new_data_task.py" %*

