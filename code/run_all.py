"""Run every analysis and regenerate all figures, in dependency order.

The experimental-fire demonstration (annaburroo/annaburroo_demo.py) needs the Annaburroo dataset
(doi:10.25919/ycd3-w209) extracted to ../data_external/annaburroo/raw; it is skipped if the folder is absent
and make_figures.py then uses the outputs already in output/.
"""
import os
import runpy
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["shape_law_profiles.py", "positional_error_identifiability.py", "identifiability_extensions.py",
           "guard_benchmark.py", "refit_factorial.py", "graded_benchmark.py", "misspecification_checks.py",
           "benchmark_tables.py", "mismatch_scenarios.py", "fire_behaviour.py", "forecast_scenarios.py", "annaburroo/annaburroo_demo.py",
           "make_figures.py"]
RAW = os.path.join(HERE, "..", "data_external", "annaburroo", "raw")

if __name__ == "__main__":
    for script in SCRIPTS:
        if script.startswith("annaburroo") and not os.path.isdir(RAW):
            print(f"== {script} skipped: dataset not found at {os.path.abspath(RAW)}", flush=True)
            continue
        t0 = time.time()
        print(f"== {script}", flush=True)
        sys.argv = [script]
        runpy.run_path(os.path.join(HERE, script), run_name="__main__")
        print(f"   done in {time.time() - t0:.0f} s", flush=True)
