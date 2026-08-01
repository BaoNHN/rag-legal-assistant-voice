@echo off
echo =============================================
echo   RAG Legal Assistant - Install Script
echo =============================================
echo.

REM Check if NVIDIA GPU exists
nvidia-smi >nul 2>&1
IF %ERRORLEVEL% == 0 (
    echo [GPU DETECTED] Installing CUDA PyTorch for faster OCR...
    echo Checking CUDA version...
    FOR /F "tokens=*" %%i IN ('nvidia-smi ^| findstr "CUDA Version"') DO SET CUDA_LINE=%%i
    echo %CUDA_LINE%
    echo.
    echo Choose PyTorch CUDA version:
    echo   1. CUDA 11.8  (older GPUs / driver)
    echo   2. CUDA 12.1  (newer GPUs / driver)
    echo   3. CPU only   (skip GPU)
    echo.
    set /p CUDA_CHOICE="Enter 1, 2 or 3: "

    REM torch>=2.6 required: transformers refuses to load model checkpoints
    REM (incl. the BAAI/bge-m3 embedding model) via torch.load below that
    REM version — see CVE-2025-32434. Pinned here so a stale/older cached
    REM wheel never gets installed by accident.
    IF "%CUDA_CHOICE%"=="1" (
        echo Installing PyTorch CUDA 11.8...
        pip uninstall torch torchvision -y
        pip install "torch>=2.6" torchvision --index-url https://download.pytorch.org/whl/cu118
        echo DEVICE=cuda> .device_config
    ) ELSE IF "%CUDA_CHOICE%"=="2" (
        echo Installing PyTorch CUDA 12.1...
        pip uninstall torch torchvision -y
        pip install "torch>=2.6" torchvision --index-url https://download.pytorch.org/whl/cu121
        echo DEVICE=cuda> .device_config
    ) ELSE (
        echo Installing CPU-only PyTorch...
        pip install "torch>=2.6" torchvision --index-url https://download.pytorch.org/whl/cpu
        echo DEVICE=cpu> .device_config
    )
) ELSE (
    echo [NO GPU DETECTED] Installing CPU-only PyTorch ^(~200MB^)...
    pip install "torch>=2.6" torchvision --index-url https://download.pytorch.org/whl/cpu
    echo DEVICE=cpu> .device_config
)

echo.
echo Installing all other dependencies...
pip install -r requirements.txt

echo.
echo =============================================
echo   Installation complete!
echo.
echo   Next steps:
echo     1. python app.py
echo     2. Open http://127.0.0.1:8000, log in as admin1 / Admin@123
echo     3. Go to "Import van ban" to import law PDF/DOCX, scenario DOCX,
echo        or a dataset .xlsx - this builds chroma_db (no manual script needed)
echo =============================================
pause