@echo off
setlocal

call "%~dp0run_torch.cmd" -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda); print('device count:', torch.cuda.device_count()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'); x=torch.randn(2048,2048,device='cuda' if torch.cuda.is_available() else 'cpu'); y=x@x; print('matmul ok:', tuple(y.shape), y.device)"

