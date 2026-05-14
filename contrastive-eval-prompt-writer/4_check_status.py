# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import json
import logging
import argparse
import re
import sys
from google import genai
from google.genai import types

# Centralized path configuration using ABSOLUTE paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATHS = {
    "logs_dir": os.path.join(BASE_DIR, "logs"),
    "log_file_template": os.path.join(BASE_DIR, "logs", "run_contrastive_{version}.log")
}

def setup_logging(version):
    """Sets up logging to console and a version-specific log file."""
    os.makedirs(PATHS["logs_dir"], exist_ok=True)
    log_file = PATHS["log_file_template"].format(version=version)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode='a', encoding='utf-8')
        ]
    )


def check_score_trend(log_version):
    """Checks if the score difference is increasing by reading the log file."""
    log_file = PATHS["log_file_template"].format(version=log_version)
    if not os.path.exists(log_file):
        logging.warning(f"Log file {log_file} not found for trend analysis.")
        return

    diffs = []
    # Pattern to match: [v0] Average Scores - Good: 3.50, Poor: 1.20, Diff: 2.30
    pattern = re.compile(r'\[v\d+\] Average Scores - .*? Diff: ([\d\.-]+)')
    
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                diffs.append(float(match.group(1)))
    
    if len(diffs) < 2:
        return

    logging.info(f"Score Difference Trend: {diffs}")
    
    # Check last 3 entries for stagnation or decrease
    if len(diffs) >= 3:
        last_three = diffs[-3:]
        # If decreasing or plateauing
        # last_three[0] -> last_three[1] -> last_three[2]
        is_stagnant = all(last_three[i] <= last_three[i-1] + 0.01 for i in range(1, len(last_three)))
        
        if is_stagnant:
            logging.warning("!!! WARNING: Score difference has plateaued or decreased for 3 consecutive steps. Optimization may be saturating.")
        elif last_three[-1] < last_three[-2]:
            logging.warning(f"!!! WARNING: Score difference decreased in the last step ({last_three[-2]:.2f} -> {last_three[-1]:.2f}).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, required=True, help="Current version being processed (e.g. v0)")
    parser.add_argument("--new_version", type=str, required=True, help="Newly generated version (e.g. v1)")
    parser.add_argument("--log_version", type=str, help="Version tag for the log file (defaults to --version)")
    
    args = parser.parse_args()
    log_ver = args.log_version if args.log_version else args.version
    setup_logging(log_ver)
        
    # Check score trend
    check_score_trend(log_ver)
