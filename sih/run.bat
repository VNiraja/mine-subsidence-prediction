@echo off

echo =====================================
echo Mine Subsidence AI
echo =====================================

call .venv\Scripts\activate

echo.
echo Starting FastAPI...

start "FastAPI" cmd /k ^
"uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000"

timeout /t 3 /nobreak >nul

echo.
echo Starting Streamlit...

start "Dashboard" cmd /k ^
"streamlit run dashboard/app.py"

echo.
echo =====================================
echo Backend: http://127.0.0.1:8000
echo Dashboard: http://localhost:8501
echo =====================================

pause