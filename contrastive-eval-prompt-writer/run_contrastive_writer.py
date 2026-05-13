# python run_contrastive_writer.py --version v0 --num_iterations 10 --start_step 2
import subprocess
import argparse
import logging
import os
import re
import sys
import glob

def setup_logging(version):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"run_contrastive_{version}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def run_command(command):
    logging.info(f"Running command: {' '.join(command)}")
    # Use capture_output=True to get stdout and stderr
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Print sub-process output to console
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
        
    if result.returncode != 0:
        logging.error(f"Command failed with return code {result.returncode}")
        exit(1)
    # Combine stdout and stderr for processing (e.g., extracting scores)
    return result.stdout + result.stderr

def get_next_version(v_string):
    """Increments version string (e.g., v0 -> v1)."""
    match = re.search(r'v(\d+)', v_string)
    if match:
        version_num = int(match.group(1))
        return f"v{version_num + 1}"
    return v_string + "_next"

def cleanup_temp_files(directory="inputs"):
    logging.info(f"Cleaning up temporary files in {directory}...")
    files = glob.glob(os.path.join(directory, "tmp_*"))
    for f in files:
        try:
            os.remove(f)
            logging.info(f"Deleted: {f}")
        except OSError as e:
            logging.error(f"Error deleting {f}: {e.strerror}")

def main():
    parser = argparse.ArgumentParser(description="Run the contrastive EVAL prompt optimization pipeline with iterations.")
    parser.add_argument("--version", type=str, required=True, help="Starting version tag (e.g., v0)")
    parser.add_argument("--num_iterations", type=int, default=1, help="Number of optimization loops to run")
    parser.add_argument("--start_step", type=int, default=1, help="Step to start from in the first iteration (0, 1, 2, or 3)")
    
    args = parser.parse_args()
    setup_logging(args.version)
    
    current_version = args.version
    
    # Step 0: Initiation (Only run once at the very beginning)
    if current_version == "v0" and args.start_step <= 0:
        logging.info("--- Step 0: Initiation (Generating Initial Prompts) ---")
        run_command(["python", "0_initiate.py"])
    elif current_version == "v0" and args.start_step == 1:
        # Check if prompts exist, if not, initiate anyway to be safe
        if not os.path.exists("eval_prompts/v0-judge.txt"):
            logging.info("--- Step 0: Initiation (Auto-running because v0-judge.txt is missing) ---")
            run_command(["python", "0_initiate.py"])

    for i in range(args.num_iterations):
        logging.info(f"\n{'='*20} ITERATION {i+1} (Version: {current_version}) {'='*20}")
        
        # Step 1: Batch Evaluate
        if i == 0 and args.start_step > 1:
            logging.info(f"--- Skipping Step 1: Batch Evaluation ({current_version}) ---")
        else:
            logging.info(f"--- Step 1: Batch Evaluation ({current_version}) ---")
            run_command([
                "python", "1_batch_evaluate.py",
                "--version", current_version,
                "--log_version", args.version
            ])
        
        # Step 2: Critic
        if i == 0 and args.start_step > 2:
            logging.info(f"--- Skipping Step 2: Critic Analysis ({current_version}) ---")
        else:
            logging.info(f"--- Step 2: Critic Analysis ({current_version}) ---")
            critic_output = run_command([
                "python", "2_critic.py",
                "--version", current_version,
                "--log_version", args.version
            ])
            
            # Extract and log average scores
            for line in critic_output.splitlines():
                if "Average Scores -" in line:
                    idx = line.find("Average Scores -")
                    if idx != -1:
                        logging.info(f"[{current_version}] {line[idx:]}")
        
        # Step 3: Optimize Evaluation Prompt
        logging.info(f"--- Step 3: EVAL Prompt Optimization ({current_version}) ---")
        run_command([
            "python", "3_optimize.py",
            "--version", current_version,
            "--log_version", args.version
        ])

        # Step 4: Check Status and Align Prompts
        next_version = get_next_version(current_version)
        logging.info(f"--- Step 4: Check Status and Align Prompts ({current_version} -> {next_version}) ---")
        run_command([
            "python", "4_check_status.py",
            "--version", current_version,
            "--new_version", next_version,
            "--log_version", args.version
        ])
        
        # Cleanup temporary input files after the iteration
        cleanup_temp_files()
        
        # Update version for next iteration
        # next_version = get_next_version(current_version) # already calculated above
        
        # Check if the next version's prompt file was actually created
        next_prompt_path = f"eval_prompts/{next_version}-judge.txt"
        if i < args.num_iterations - 1:
            if os.path.exists(next_prompt_path):
                logging.info(f"Moving to next iteration with {next_version}")
                current_version = next_version
            else:
                logging.error(f"Expected next prompt {next_prompt_path} not found. Stopping.")
                break
    
    logging.info(f"\n{'='*20} All {args.num_iterations} Iterations Complete {'='*20}")

if __name__ == "__main__":
    main()
