@echo off
echo ========================================
echo CSK IPL Prediction - Complete Pipeline
echo ========================================

echo Running complete end-to-end pipeline...

echo.
echo 📥 Step 1: ETL Pipeline
call "%~dp0\run_etl.bat"
if %ERRORLEVEL% NEQ 0 (
    echo ❌ ETL failed, stopping pipeline
    exit /b 1
)

echo.
echo 🤖 Step 2: ML Training
call "%~dp0\run_train.bat"
if %ERRORLEVEL% NEQ 0 (
    echo ❌ ML Training failed, stopping pipeline
    exit /b 1
)

echo.
echo 🎉 Complete pipeline finished successfully!
echo.
echo Next steps:
echo - Run dashboard: run_dashboard.bat
echo - Check models folder for trained models
echo - Check data/processed for processed datasets

pause
