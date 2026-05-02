@echo off
set "ROOT=%~dp0.."
set "PYTHONPATH=%ROOT%\.deps\py312"
if not exist "%ROOT%\.tmp\pytemp" mkdir "%ROOT%\.tmp\pytemp"
if not exist "%ROOT%\.tmp\pwncache\.pwntools-cache-3.12" mkdir "%ROOT%\.tmp\pwncache\.pwntools-cache-3.12"
if not exist "%ROOT%\.tmp\pwncache\.pwntools-cache-3.12\update" echo never>"%ROOT%\.tmp\pwncache\.pwntools-cache-3.12\update"
set "TEMP=%ROOT%\.tmp\pytemp"
set "TMP=%ROOT%\.tmp\pytemp"
set "XDG_CACHE_HOME=%ROOT%\.tmp\pwncache"
set "XDG_CONFIG_HOME=%ROOT%\.pwnconfig"
"C:\Users\Asus\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" %*
