#!/usr/bin/env python3
"""
Quick Start Script for Streamlit Dashboard
Runs the integrated image matcher dashboard with one command
"""

import os
import sys
import subprocess
import webbrowser
from pathlib import Path

def run_streamlit_app():
    """Run the Streamlit dashboard"""
    
    # Get the directory of this script
    current_dir = Path(__file__).parent.absolute()
    dashboard_file = current_dir / "streamlit_dashboard.py"
    
    # Check if dashboard exists
    if not dashboard_file.exists():
        print(f"❌ Dashboard file not found: {dashboard_file}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("🔍 INTEGRATED IMAGE MATCHER - STREAMLIT DASHBOARD")
    print("="*70)
    print(f"\n📁 Dashboard Location: {dashboard_file}")
    print(f"📁 Working Directory: {current_dir}")
    print("\n✨ Starting dashboard...\n")
    
    # Start Streamlit app
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(dashboard_file),
                "--logger.level=info"
            ],
            cwd=current_dir
        )
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error running dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_streamlit_app()
