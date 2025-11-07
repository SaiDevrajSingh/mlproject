@echo off
echo ========================================
echo CSK IPL Prediction - ETL Pipeline
echo ========================================

echo Starting ETL pipeline...
python -c "from src.pipelines.etl_pipeline import run_csk_etl_pipeline; import sys; results = run_csk_etl_pipeline(); print('ETL Pipeline completed successfully!')"

if %ERRORLEVEL% EQU 0 (
    echo ✅ ETL pipeline completed successfully!
) else (
    echo ❌ ETL pipeline failed!
    exit /b 1
)

pause
