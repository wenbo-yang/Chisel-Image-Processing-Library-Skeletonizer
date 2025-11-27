@echo off
REM Test runner script for Chisel Image Processing Library (Windows)
REM This script runs all tests with coverage reporting

setlocal enabledelayedexpansion

REM Colors and formatting
set "YELLOW=[1;33m"
set "GREEN=[0;32m"
set "RED=[0;31m"
set "NC=[0m"

REM Print header
echo.
echo ========================================
echo Running Tests for Chisel Library
echo ========================================
echo.

REM Check if virtual environment is activated
if "%VIRTUAL_ENV%"=="" (
    echo WARNING: Virtual environment not activated
    echo Consider running: venv\Scripts\activate
    echo.
)

REM Check if pytest is installed
where pytest >nul 2>nul
if errorlevel 1 (
    echo ERROR: pytest is not installed
    echo Please install dependencies: pip install -r requirements.txt
    exit /b 1
)

REM Run tests with coverage
echo Running pytest with coverage...
echo.

pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

set TEST_EXIT_CODE=!errorlevel!

REM Print results
echo.
echo ========================================
if %TEST_EXIT_CODE% equ 0 (
    echo Tests PASSED!
    echo Coverage report generated in htmlcov\index.html
) else (
    echo Tests FAILED with exit code %TEST_EXIT_CODE%
)
echo ========================================
echo.

exit /b %TEST_EXIT_CODE%
