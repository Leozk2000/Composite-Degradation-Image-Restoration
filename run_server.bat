@echo off
echo Starting setup...

REM Check if Anaconda is available
where conda >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Anaconda is not found. Please install Anaconda and try again.
    pause
    exit /b 1
)

REM Check if requirements.txt exists
IF NOT EXIST requirements.txt (
    echo requirements.txt not found in current directory!
    pause
    exit /b 1
)

REM Check if app.py exists
IF NOT EXIST app.py (
    echo app.py not found in current directory!
    pause
    exit /b 1
)

REM Check if environment exists
conda env list | find "onerestore" > nul
IF %ERRORLEVEL% EQU 0 (
    echo Environment 'onerestore' already exists. Activating...
    CALL conda activate onerestore
) ELSE (
    echo Creating new environment 'onerestore'...
    CALL conda create -n onerestore python=3.10 -y
    CALL conda activate onerestore
)

REM Check for NVIDIA GPU and install appropriate PyTorch version
echo Checking for NVIDIA GPU...
where nvidia-smi >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    echo NVIDIA GPU detected. Installing PyTorch with CUDA 11.7 support...
    CALL pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
) ELSE (
    echo No NVIDIA GPU detected. Installing CPU-only version of PyTorch...
    CALL pip install torch==2.0.1 torchvision==0.15.2
)

REM Install other packages
echo Installing additional packages from requirements.txt...
CALL pip install -r requirements.txt

REM Start the Python script
echo Starting app.py...
start python app.py

REM Wait a few seconds for the server to start
echo Waiting for server to start...
timeout /t 5 /nobreak

REM Open the browser
echo Opening browser...
start http://127.0.0.1:7860/

echo Setup complete! Server should be running at http://127.0.0.1:7860/
echo Press any key to close this window...
pause
