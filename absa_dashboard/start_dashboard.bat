@echo off
REM Quick Start Script for ABSA Dashboard
REM =======================================

echo.
echo ========================================
echo    ABSA Dashboard Quick Start
echo ========================================
echo.

REM Check if we're in the right directory
if not exist "dashboard.py" (
    echo ERROR: dashboard.py not found!
    echo Please run this script from the absa_dashboard directory.
    pause
    exit /b 1
)

REM Check if preprocessed data exists
if not exist "data\preprocessed_data.parquet" (
    echo WARNING: Preprocessed data not found!
    echo Please run preprocessing first in the Jupyter notebook.
    echo.
    pause
    exit /b 1
)

echo Starting Streamlit dashboard...
echo Dashboard will open in your browser at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the dashboard
echo.

REM Launch Streamlit
streamlit run dashboard.py

pause
