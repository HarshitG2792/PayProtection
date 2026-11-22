@echo off
REM Windows one-click setup: creates venv and installs requirements
python -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Setup done. Now in VS Code: Ctrl+Shift+P -> "Python: Select Interpreter" -> pick .\venv
pause
