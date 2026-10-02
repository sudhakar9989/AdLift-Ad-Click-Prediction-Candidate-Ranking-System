"""Launch the AdLift Streamlit interactive dashboard."""

import os
import sys


def main() -> None:
    dashboard_path = os.path.join("src", "adlift", "dashboard.py")
    if not os.path.exists(dashboard_path):
        print(f"Error: Dashboard script not found at {dashboard_path}")
        sys.exit(1)

    print("Launching AdLift interactive dashboard...")
    print(f"Running: streamlit run {dashboard_path} --server.port 8501 --server.headless false")

    import streamlit.web.cli as stcli
    sys.argv = ["streamlit", "run", dashboard_path, "--server.port", "8501", "--server.headless", "true"]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
