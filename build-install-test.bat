@echo off

:: Step 1: Build the Python package
python -m build
if %errorlevel% neq 0 (
    echo Failed to build the package.
    exit /b 1
)

:: Step 2: Find the latest .whl file in the dist directory
set "latest_whl="
for /f "delims=" %%f in ('dir /b /o-d dist\*.whl') do (
    set "latest_whl=%%f"
    goto :found_whl
)

:found_whl
if not defined latest_whl (
    echo No .whl file found in the dist directory.
    exit /b 1
)

:: Step 3: Install the latest .whl file using pip
pip install dist\%latest_whl%
if %errorlevel% neq 0 (
    echo Failed to install the package.
    exit /b 1
)


:: Step 4: Run the main.py script
pip install -U nose2
python -m nose2