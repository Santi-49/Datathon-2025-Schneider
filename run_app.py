"""
Quick Launch Script for Sales Opportunity Explainability System
Run this script to check dependencies and launch the Streamlit app
"""

import sys
import subprocess
from pathlib import Path
import sys


def check_file_exists(filepath: str) -> bool:
    """
    Check if a file exists at the given path.

    Args:
        filepath (str): Path to the file.

    Returns:
        bool: True if file exists, False otherwise.
    """
    return Path(filepath).exists()


def main():
    """
    Main entry point for the quick launch script.
    Checks dependencies/files and launches the Streamlit app.
    """
    print("=" * 60)
    print("Sales Opportunity Explainability System - Quick Launch")
    print("=" * 60)
    print()

    # Check if prediction files exist
    print("Checking required files...")

    predictions_csv = check_file_exists("data/predictions_with_shap.csv")
    predictions_json = check_file_exists("predictions_detailed.json")
    model_file = check_file_exists("catboost_model.joblib")

    print(f"  predictions_with_shap.csv: {'✅' if predictions_csv else '❌'}")
    print(f"  predictions_detailed.json: {'✅' if predictions_json else '❌'}")
    print(f"  catboost_model.joblib: {'✅' if model_file else '❌'}")
    print()

    if not (predictions_csv and predictions_json):
        print("⚠️  Required prediction files not found!")
        print()
        print("Would you like to run temp.py to generate them? (y/n): ", end="")
        response = input().strip().lower()

        if response == "y":
            print()
            print("Running temp.py to train model and generate predictions...")
            print("-" * 60)
            try:
                subprocess.run([sys.executable, "temp.py"], check=True)
                print("-" * 60)
                print("✅ Model training and prediction generation complete!")
                print()
            except subprocess.CalledProcessError:
                print("❌ Error running temp.py")
                return
        else:
            print("Cannot launch app without prediction files. Exiting.")
            sys.exit(1)
            return

    # Launch Streamlit app
    print("🚀 Launching Streamlit app...")
    print()
    print("The app will open in your default browser at: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    print()

    try:
        subprocess.run(["streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n\nApp stopped by user.")
    except FileNotFoundError:
        print("❌ Streamlit not found. Install it with: pip install streamlit")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
