"""
One Script to Run Everything - Single Window Only
Run: python run_all.py
"""

import subprocess
import os
import sys
import time
import webbrowser
import requests

PROJECT_PATH = r"D:\Project\fraud_detection_complete"


def show_banner():
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                     FRAUD DETECTION SYSTEM v2.0                               ║
║                   Real-time ML/AI Fraud Detection                             ║
║                   Multi-Model Support (6 Models)                              ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)


def run_api():
    """Run API in background"""
    os.chdir(PROJECT_PATH)
    cmd = 'start cmd /k "title API Server && cd /d {} && .venv\\Scripts\\activate && python application/bank_api.py"'.format(PROJECT_PATH)
    subprocess.Popen(cmd, shell=True)
    return True


def wait_for_api(max_retries=15):
    """Wait for API to be ready"""
    print("   ⏳ Waiting for API to be ready (up to 30 seconds)...")
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                print("   ✅ API is ready!")
                return True
        except:
            pass
        time.sleep(2)
        print(f"   ⏳ Still waiting... ({i + 1}/{max_retries})")

    print("   ⚠️ API did not start automatically.")
    print("   💡 Please check if the API terminal opened successfully.")
    return False


def run_multi_model_dashboard():
    """Run Multi-Model Dashboard WITHOUT auto-opening browser"""
    os.chdir(PROJECT_PATH)
    # Using dashboard_multi_model.py with headless mode to prevent auto-browser
    cmd = 'start cmd /k "title Multi-Model Dashboard && cd /d {} && .venv\\Scripts\\activate && streamlit run application/dashboard_multi_model.py --server.port 8501 --server.headless true --browser.gatherUsageStats false"'.format(PROJECT_PATH)
    subprocess.Popen(cmd, shell=True)


def main():
    show_banner()

    print("=" * 70)
    print("🔬 STARTING FRAUD DETECTION SYSTEM (MULTI-MODEL)")
    print("=" * 70)

    os.chdir(PROJECT_PATH)

    # Check if multi-model dashboard exists
    multi_model_path = os.path.join(PROJECT_PATH, "application", "dashboard_multi_model.py")
    if not os.path.exists(multi_model_path):
        print(f"\n❌ Multi-Model Dashboard not found at: {multi_model_path}")
        print("   Please create dashboard_multi_model.py first")
        sys.exit(1)

    # Start API in new window
    print("\n▶️ Starting API Server in new window...")
    run_api()

    # Wait for API to be ready
    api_ready = wait_for_api()

    if api_ready:
        print("\n✅ API Server is running on http://localhost:8000")
    else:
        print("\n⚠️ API Server may still be starting...")

    # Start Multi-Model Dashboard (no auto-browser)
    print("\n▶️ Starting Multi-Model Dashboard (no auto-browser)...")
    run_multi_model_dashboard()
    time.sleep(5)

    # Open ONLY ONE browser window
    print("\n🌐 Opening dashboard in your browser...")
    webbrowser.open("http://localhost:8501")

    print("\n" + "=" * 70)
    print("📊 MULTI-MODEL FRAUD DETECTION SYSTEM IS RUNNING!")
    print("=" * 70)
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │  📡 API Server:  http://localhost:8000                         │
    │  🎨 Dashboard:   http://localhost:8501                         │
    └─────────────────────────────────────────────────────────────────┘

    🤖 AVAILABLE MODELS:
    ┌─────────────────────────────────────────────────────────────────┐
    │  🔴 XGBoost            - 99.96% accuracy                       │
    │  🟢 Random Forest      - 99.94% accuracy                       │
    │  🔵 Logistic Regression - 97.23% accuracy                      │
    │  🟠 SVM                - 96.89% accuracy                       │
    │  🟣 Neural Network     - 99.94% accuracy                       │
    │  ⚪ Rule-Based         - 85.00% accuracy                       │
    └─────────────────────────────────────────────────────────────────┘

    💡 Instructions:
       1. Select any model from the dropdown menu
       2. Click "LIVE" button to start real-time monitoring
       3. Or use "TEST" for manual transaction testing
       4. Compare model performance in the comparison table

    🛑 To stop: Close both terminal windows
    """)

    print("\n" + "=" * 70)
    print("⏰ Press Enter to exit this script (services will continue running)")
    print("=" * 70)

    input()


if __name__ == "__main__":
    main()