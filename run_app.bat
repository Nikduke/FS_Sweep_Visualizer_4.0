@echo off
setlocal

cd /d "%~dp0"

set "APP_FILE=fs_sweep_app_spline.py"
set "PYTHON_EXE="

if not exist "%APP_FILE%" (
    echo Cannot find %APP_FILE% in:
    echo %CD%
    pause
    exit /b 1
)

if defined CONDA_PREFIX (
    if exist "%CONDA_PREFIX%\python.exe" (
        set "PYTHON_EXE=%CONDA_PREFIX%\python.exe"
    )
)

if not defined PYTHON_EXE (
    if exist "%USERPROFILE%\anaconda3\python.exe" (
        set "PYTHON_EXE=%USERPROFILE%\anaconda3\python.exe"
    )
)

if not defined PYTHON_EXE (
    if exist "%LOCALAPPDATA%\anaconda3\python.exe" (
        set "PYTHON_EXE=%LOCALAPPDATA%\anaconda3\python.exe"
    )
)

if not defined PYTHON_EXE (
    echo Anaconda Python was not found.
    echo Checked:
    echo   %USERPROFILE%\anaconda3\python.exe
    echo   %LOCALAPPDATA%\anaconda3\python.exe
    echo.
    echo If you use a named Conda environment, start this file from an activated Anaconda Prompt.
    pause
    exit /b 1
)

"%PYTHON_EXE%" -m streamlit --version >nul 2>nul
if errorlevel 1 (
    echo Streamlit is not installed for this Anaconda Python environment.
    echo Installing dependencies from requirements.txt...
    "%PYTHON_EXE%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Dependency installation failed.
        pause
        exit /b 1
    )
)

echo Starting FS Sweep Visualizer...
echo App: %APP_FILE%
echo Python: %PYTHON_EXE%
echo.

"%PYTHON_EXE%" -m streamlit run "%APP_FILE%"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo App exited with code %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%
