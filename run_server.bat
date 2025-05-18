@echo off
echo Starting setup...

REM Check if Anaconda is available
where conda >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Anaconda is not found. Please install Anaconda and try again.
    pause
    exit /b 1
)

REM Check if environment.yml exists
IF NOT EXIST environment.yml (
    echo environment.yml not found in current directory!
    pause
    exit /b 1
)

REM Check if app.py exists
IF NOT EXIST app.py (
    echo app.py not found in current directory!
    pause
    exit /b 1
)

REM Get environment name from yml file
for /f "tokens=2" %%i in ('type environment.yml ^| findstr /B "name:"') do set ENV_NAME=%%i

REM Check if environment exists
conda env list | find "%ENV_NAME%" > nul
IF %ERRORLEVEL% EQU 0 (
    echo Environment %ENV_NAME% exists, updating...
    CALL conda env update -f environment.yml
) ELSE (
    echo Creating new environment from environment.yml...
    CALL conda env create -f environment.yml
)

REM Activate the environment
echo Activating environment %ENV_NAME%...
CALL conda activate %ENV_NAME%

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