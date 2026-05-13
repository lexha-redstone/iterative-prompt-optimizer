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

# python run_iterations.py --start_version v1 --start_step 4
# python run_iterations.py --start_version v1 --num_iterations 3
# python run_iterations.py --start_version v0 --num_iterations 4 --start_step 3
# python run_iterations.py --start_version v0 --num_iterations 10

import argparse
import glob
import logging
import os
import re
import subprocess
import sys


def setup_logging():
  """Sets up logging for the orchestrator."""
  os.makedirs("logs", exist_ok=True)
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - [ORCHESTRATOR] - %(levelname)s - %(message)s",
      handlers=[
          logging.FileHandler("logs/iteration_flow.log"),
          logging.StreamHandler(sys.stdout),
      ],
  )


def get_next_version(current_version):
  """Increments version string (e.g., v0 -> v1)."""
  match = re.search(r"v(\d+)", current_version)
  if match:
    version_num = int(match.group(1))
    return f"v{version_num + 1}"
  return current_version + "_optimized"


def run_script(script_name, args_list):
  """Runs a python script as a subprocess and waits for completion."""
  cmd = [sys.executable, script_name] + args_list
  logging.info(f"Running: {' '.join(cmd)}")

  process = subprocess.Popen(
      cmd,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      bufsize=1,
      universal_newlines=True,
  )

  for line in process.stdout:
    print(f"[{script_name}] {line.strip()}")

  process.wait()

  if process.returncode != 0:
    logging.error(
        f"Script {script_name} failed with exit code {process.returncode}"
    )
    return False
  return True


def cleanup_temp_files():
  """Removes temporary input files from the inputs directory."""
  temp_files = glob.glob("inputs/tmp_*")
  if temp_files:
    logging.info(f"Cleaning up {len(temp_files)} temporary files in inputs/")
    for f in temp_files:
      try:
        os.remove(f)
      except OSError as e:
        logging.error(f"Error deleting file {f}: {e}")


def main():
  parser = argparse.ArgumentParser(
      description="Iterative Prompt Optimization Orchestrator"
  )
  parser.add_argument(
      "--start_version",
      type=str,
      default="v0",
      help="Initial version (e.g., v0)",
  )
  parser.add_argument(
      "--num_iterations",
      type=int,
      default=1,
      help="Number of optimization loops to run",
  )
  parser.add_argument(
      "--start_step",
      type=int,
      default=1,
      choices=[1, 2, 3, 4],
      help=(
          "Step to start from in the first iteration: 1: Inference, 2: Eval, 3:"
          " Critic, 4: Optimize"
      ),
  )

  args = parser.parse_args()
  setup_logging()

  # Generate session-specific eval log path
  from datetime import datetime

  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  session_eval_log = f"logs/eval_logs_{timestamp}.jsonl"
  logging.info(f"Session evaluation log will be saved to: {session_eval_log}")

  current_version = args.start_version

  logging.info(
      f"Starting optimization loop starting from version"
      f" '{current_version}' step {args.start_step}"
  )

  for i in range(args.num_iterations):
    logging.info(
        f"=== Starting Iteration {i+1}/{args.num_iterations} (Version:"
        f" {current_version}) ==="
    )

    # In the first iteration, we might start from a middle step.
    # In subsequent iterations, we always start from step 1.
    effective_start_step = args.start_step if i == 0 else 1

    # Step 1: Batch Inference
    if effective_start_step <= 1:
      logging.info(f"Step 1: Batch Inference ({current_version})")
      if not run_script(
          "1_batch_inference.py",
          ["--version", current_version],
      ):
        break

    # Step 2: Batch Evaluation
    if effective_start_step <= 2:
      logging.info(f"Step 2: Batch Evaluation ({current_version})")
      if not run_script(
          "2_batch_evaluate.py",
          ["--version", current_version],
      ):
        break

    # Step 3: Critic
    if effective_start_step <= 3:
      logging.info(f"Step 3: Critic ({current_version})")
      if not run_script(
          "3_critic.py",
          [
              "--version",
              current_version,
              "--eval_log",
              session_eval_log,
          ],
      ):
        break

    # Step 4: Optimize
    if effective_start_step <= 4:
      logging.info(f"Step 4: Optimize ({current_version})")
      if not run_script(
          "4_optimize.py",
          [
              "--version",
              current_version,
              "--eval_log",
              session_eval_log,
          ],
      ):
        break

    # Prepare for next iteration
    cleanup_temp_files()
    current_version = get_next_version(current_version)
    logging.info(
        f"Iteration {i+1} complete. Next version will be: {current_version}"
    )

  logging.info("All iterations completed.")


if __name__ == "__main__":
  main()
