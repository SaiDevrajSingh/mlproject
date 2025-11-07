# 🚀 **QUICK START GUIDE**

## ⚡ **Immediate Fix for Current Issues**

### **1. PowerShell Execution Fix**
In PowerShell, use `.\` prefix for local scripts:
```powershell
# Instead of: run_dashboard.bat
# Use:
.\run_dashboard.bat

# Or better, use Python:
python run_dashboard.py
```

### **2. Test the Pipeline First**
```powershell
# Test if everything works
python test_pipeline.py
```

### **3. Launch Dashboard**
```powershell
# Simple way
python run_dashboard.py

# Or direct streamlit
python -m streamlit run dashboards/streamlit_app.py
```

## 🔧 **If You Get Errors**

### **Missing Data Error**
```powershell
# Make sure IPL.csv is in the right place
# It should be at: data/raw/IPL.csv
```

### **Module Import Error**
```powershell
# Make sure you're in the project directory
cd "C:\Users\Devraj Singh\Desktop\ml_project"

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### **Pipeline Errors**
```powershell
# Run the test script to diagnose issues
python test_pipeline.py
```

## 🎯 **Recommended Workflow**

### **Step 1: Setup**
```powershell
# Navigate to project
cd "C:\Users\Devraj Singh\Desktop\ml_project"

# Activate environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### **Step 2: Test**
```powershell
# Test the pipeline
python test_pipeline.py
```

### **Step 3: Launch**
```powershell
# Launch dashboard
python run_dashboard.py
```

## 🆘 **Troubleshooting**

### **Problem: "DataExtractor has no attribute 'extract_csk_matches'"**
**Solution**: ✅ **FIXED** - Updated the extract.py file

### **Problem: "File not found: csk_match_level_data.csv"**
**Solution**: Run `python test_pipeline.py` first to create the data

### **Problem: "run_dashboard.bat not recognized"**
**Solution**: Use `.\run_dashboard.bat` or `python run_dashboard.py`

### **Problem: Import errors**
**Solution**: Make sure you're in the project directory and virtual environment is activated

## 🎉 **Success Indicators**

When everything works, you should see:
- ✅ Data extraction successful
- ✅ Data transformation complete
- ✅ Model training successful
- ✅ Dashboard launches at http://localhost:8501

## 📞 **Quick Commands Reference**

```powershell
# Essential commands
python test_pipeline.py           # Test everything
python run_dashboard.py           # Launch dashboard
python -m streamlit run dashboards/streamlit_app.py  # Direct streamlit

# Alternative batch files (use .\ prefix)
.\run_dashboard.bat               # Launch dashboard
.\scripts\run_etl.bat            # Run ETL only
.\scripts\run_train.bat          # Train models only
```

---

**🏏 Follow these steps and your CSK prediction app will be running smoothly!**
