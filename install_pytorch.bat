@echo off
REM Put this file inside the last_mile_verifier folder and double-click it.
REM It installs PyTorch into the project's venv and checks that it works.

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate
python -m pip install --upgrade pip

echo.
echo Installing PyTorch (CPU version, works on every PC)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

REM --- Have an NVIDIA GPU? Delete the CPU line above and use this one instead:
REM pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo.
echo Checking install...
python -c "import torch; print('PyTorch version:', torch.__version__, '| GPU available:', torch.cuda.is_available())"
echo.
pause
