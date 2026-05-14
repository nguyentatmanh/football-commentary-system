@echo off
echo Creating virtual environment .venv...
python -m venv .venv
if %ERRORLEVEL% neq 0 (
    echo Failed to create virtual environment.
    exit /b %ERRORLEVEL%
)

echo Activating virtual environment...
call .venv\Scripts\activate

echo Installing project dependencies...
pip install --upgrade pip
pip install -e .

echo Setup complete!
echo To activate environment in terminal run: .venv\Scripts\activate
pause
