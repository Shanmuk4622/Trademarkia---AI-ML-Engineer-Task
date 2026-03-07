@echo off
REM setup_venv.bat — Create and populate a venv for running the service
REM Usage: scripts\setup_venv.bat (run from project root)

echo Creating virtual environment in .\venv ...
python -m venv venv

echo Activating venv ...
call venv\Scripts\activate.bat

echo Installing requirements ...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo === Setup complete ===
echo To activate the venv in future:   venv\Scripts\activate.bat
echo To run the pipeline:              python scripts\run_pipeline.py
echo To start the API:                 uvicorn src.api:app --host 0.0.0.0 --port 8000
echo.
