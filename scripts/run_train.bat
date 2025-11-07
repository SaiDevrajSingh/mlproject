@echo off
echo ========================================
echo CSK IPL Prediction - ML Training
echo ========================================

echo Starting ML training pipeline...
python -c "from src.pipelines.ml_pipeline import run_complete_ml_pipeline; results = run_complete_ml_pipeline(); print('ML Training completed successfully!')"

if %ERRORLEVEL% EQU 0 (
    echo ✅ ML training completed successfully!
) else (
    echo ❌ ML training failed!
    exit /b 1
)

pause
