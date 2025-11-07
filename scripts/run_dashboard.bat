@echo off
echo ========================================
echo CSK IPL Prediction - Dashboard
echo ========================================

echo Starting Streamlit dashboard...
echo 🌐 Dashboard will open at: http://localhost:8501

cd /d "%~dp0\.."
streamlit run dashboards/streamlit_app.py

pause
