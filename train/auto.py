import subprocess
import sys
import time
import os
import pandas as pd

scripts_to_run = [
    # === Group 1: ORL Dataset ===
    'l2_orl.py',
    'mu_orl.py',
    'l1_orl.py',
    'hc_orl.py',
    'stack_orl.py',

    # === Group 2: Yale Dataset ===
    'l2_yale.py',
    'mu_yale.py',
    'l1_yale.py',
    'hc_yale.py',
    'stack_yale.py'
]


def run_script(script_name):
    print(f"\n{'=' * 60}")
    print(f"Now Training: {script_name}")
    print(f"{'=' * 60}\n")
    start_time = time.time()
    try:
        subprocess.run([sys.executable, script_name], check=True)
        print(f"\n {script_name} finished.")
    except subprocess.CalledProcessError as e:
        print(f"\n {script_name} error. Code: {e.returncode}")
    except Exception as e:
        print(f"\nUnknown error: {e}")


def merge_results():
    print(f"\n{'=' * 60}")
    print(f" Merging results together...")
    print(f"{'=' * 60}")

    raw_files = []
    summary_files = []

    for file in os.listdir('.'):
        if file.endswith('_Raw.xlsx'):
            raw_files.append(file)
        elif file.endswith('_Summary.xlsx'):
            summary_files.append(file)

    if raw_files:
        print(f"Merging {len(raw_files)} Raw files...")
        all_raw_df = pd.concat([pd.read_excel(f) for f in raw_files], ignore_index=True)
        all_raw_df.to_excel('ALL_ALGO_RAW.xlsx', index=False)
        print(">>> Merging 'ALL_ALGO_RAW.xlsx' generated.")

    if summary_files:
        print(f"Merging {len(summary_files)} Summary files...")
        all_summary_df = pd.concat([pd.read_excel(f) for f in summary_files], ignore_index=True)
        all_summary_df.to_excel('ALL_ALGO_SUMMARY.xlsx', index=False)
        print(">>> Merging 'ALL_ALGO_SUMMARY.xlsx' generated.")


if __name__ == "__main__":
    total_start = time.time()
    print(">>> Start Batch Training...\n")

    for script in scripts_to_run:
        run_script(script)

    merge_results()

    total_end = time.time()
    total_duration = total_end - total_start
    print(f"\n{'=' * 60}")
    print(f"All finished. Used: {total_duration / 60:.2f} minutes.")
    print(f"{'=' * 60}")