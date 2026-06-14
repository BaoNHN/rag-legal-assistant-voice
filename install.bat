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

    IF "%CUDA_CHOICE%"=="1" (
        echo Installing PyTorch CUDA 11.8...
        pip uninstall torch torchvision -y
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
        echo DEVICE=cuda> .device_config
    ) ELSE IF "%CUDA_CHOICE%"=="2" (
        echo Installing PyTorch CUDA 12.1...
        pip uninstall torch torchvision -y
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        echo DEVICE=cuda> .device_config
    ) ELSE (
        echo Installing CPU-only PyTorch...
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        echo DEVICE=cpu> .device_config
    )
) ELSE (
    echo [NO GPU DETECTED] Installing CPU-only PyTorch ^(~200MB^)...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
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
echo     1. python database/build_db_from_pdf.py
echo     2. python app.py
echo =============================================
pause