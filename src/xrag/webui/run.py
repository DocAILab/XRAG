import os
import subprocess
def run_web_ui():
    # 启动 Streamlit 应用
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    subprocess.run(["streamlit", "run", app_path])