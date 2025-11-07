#!/usr/bin/env python3
"""
Simple script to launch the CSK IPL Prediction dashboard
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit dashboard"""
    print("=" * 50)
    print("CSK IPL Prediction - Dashboard")
    print("=" * 50)
    print()
    print("🚀 Starting Streamlit dashboard...")
    print("🌐 Dashboard will open at: http://localhost:8501")
    print()
    
    try:
        # Change to project directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "dashboards/streamlit_app.py"
        ], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching dashboard: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
