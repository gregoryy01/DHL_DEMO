@echo off
cd /d "%~dp0"

echo =====================================
echo Creating virtual environment...
echo =====================================

if not exist .venv (
    python -m venv .venv
) else (
    echo Virtual environment already exists.
)

call .venv\Scripts\activate

echo.
echo =====================================
echo Upgrading pip...
echo =====================================
python -m pip install --upgrade pip

echo.
echo =====================================
echo Installing requirements...
echo =====================================
python -m pip install -r requirements.txt

echo.
echo =====================================
echo Setup complete.
echo =====================================
echo To activate manually later, run:
echo call .venv\Scripts\activate
pause