@echo off
setlocal
cd /d "%~dp0"
if not "%~1"=="" (
  "%~1" -c "import sys;sys.exit(0 if sys.version_info[:2]==(3,12) else 1)"
  if errorlevel 1 goto python_error
  "%~1" -m venv .venv
) else (
  py -3.12 -m venv .venv
)
if errorlevel 1 goto python_error
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m scripts.download_detector
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m scripts.check_environment
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pytest -q
exit /b %errorlevel%
:python_error
echo Python 3.12 is required. Install it, or run setup.cmd with its full python.exe path as an argument.
exit /b 1
