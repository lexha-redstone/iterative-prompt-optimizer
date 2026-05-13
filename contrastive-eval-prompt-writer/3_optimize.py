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


# python 3_optimize.py --version v0
import os
import json
import logging
import argparse
import re
import sys
from google import genai
from google.genai import types

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def setup_logging(version):
    """Sets up logging to console and a version-specific log file."""
    base_dir = "."
    os.makedirs(f"{base_dir}/logs", exist_ok=True)
    log_file = f"{base_dir}/logs/run_contrastive_{version}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode='a', encoding='utf-8')
        ]
    )

def read_file(path):
    """Reads a file and returns its content as a string."""
    if not os.path.exists(path):
        logging.error(f"File not found: {path}")
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def gather_critic_results(critic_dir):
    """Reads all critic predictions, calculates stats, and concatenates findings."""
    if not os.path.exists(critic_dir):
        logging.warning(f"Critic directory not found: {critic_dir}")
        return "No critic feedback available.", {}

    results = []
    score_diffs = []
    
    for filename in os.listdir(critic_dir):
        if not filename.endswith(".jsonl"):
            continue
            
        with open(os.path.join(critic_dir, filename), "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    item = json.loads(line)
                    critic_key = item.get("key", "unknown")
                    key = critic_key.replace("critic-", "") if critic_key.startswith("critic-") else critic_key

                    response_parts = item["response"]["candidates"][0]["content"]["parts"]
                    text = ""
                    for p in response_parts:
                        if "text" in p:
                            text += p["text"]
                    
                    # Try to extract structured data for the meta-prompt
                    try:
                        json_str = text
                        if "```json" in text:
                            json_str = re.search(r'```json\n(.*?)\n```', text, re.DOTALL).group(1)
                        elif "```" in text:
                            json_str = re.search(r'```\n(.*?)\n```', text, re.DOTALL).group(1)
                        
                        data = json.loads(json_str)
                        diff = float(data.get("score_difference", 0))
                        score_diffs.append(diff)
                        
                        structured_text = f"#### Score Difference: {diff:.2f}\n"
                        structured_text += f"**Vision Gap:** {data.get('vision_gap_analysis', 'N/A')}\n"
                        structured_text += "**Judge Vulnerabilities:**\n" + "\n".join([f"- {x}" for x in data.get("judge_vulnerabilities", [])])
                        structured_text += "\n**Judge Optimization Rules:**\n" + "\n".join([f"- {x}" for x in data.get("judge_optimization_rules", [])])
                        
                        results.append(f"### Sample {key}:\n{structured_text}")
                    except:
                        results.append(f"### Sample {key}:\n{text}")
                except Exception:
                    continue
    
    stats = {
        "avg_diff": sum(score_diffs) / len(score_diffs) if score_diffs else 0,
        "sample_count": len(score_diffs)
    }
    
    return "\n\n".join(results), stats

def optimize_eval_prompt(version, log_version=None):
    """Main function to run the evaluation prompt optimization."""
    setup_logging(log_version if log_version else version)
    client = genai.Client(vertexai=True, project=config.PROJECT_ID, location=config.LOCATION)
    
    base_dir = "."
    critic_results_dir = f"{base_dir}/results/critic/{version}"
    
    meta_prompt_template = read_file(f"{base_dir}/eval_prompts/optimizer_contrastive.txt")
    current_eval_prompt_path = f"{base_dir}/eval_prompts/{version}-judge.txt"
    current_eval_prompt = read_file(current_eval_prompt_path)
    
    critic_results_text, stats = gather_critic_results(critic_results_dir)
    
    if not current_eval_prompt or not meta_prompt_template:
        logging.error("Required prompt files are missing. Aborting.")
        sys.exit(1)

    # Prepare context for the meta-prompt
    structured_context = f"""
## CURRENT PERFORMANCE METRICS
- Average Score Difference (Good - Poor): {stats.get('avg_diff', 0):.2f}
- Number of Samples Analyzed: {stats.get('sample_count', 0)}

## TASK OBJECTIVE
Optimize the EVALUATION PROMPT (Judge instructions) to MAXIMIZE the score difference between Good and Poor samples.
A higher "Average Score Difference" indicates a more effective evaluation prompt. 
If the current difference is low, you must make the criteria more punitive for "Poor" quality traits and more rewarding for "Good" quality traits.

## CONTRASTIVE ANALYSIS (GOOD vs POOR SAMPLES)
{critic_results_text}

## CURRENT EVALUATION PROMPT (To be optimized)
{current_eval_prompt}
"""
    
    final_meta_prompt = meta_prompt_template.format(
        structured_context=structured_context
    )

    # Log the final meta prompt for debugging
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "meta_prompt_logs.txt")
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(
            f"\n{'='*50}\nVERSION: {version}\n{'='*50}\n\n{final_meta_prompt}\n\n"
        )

    logging.info(f"Calling Gemini API for evaluation prompt optimization...")
    response = client.models.generate_content(
        model=config.OPTIMIZATION_MODEL,
        contents=final_meta_prompt
    )
    
    optimized_content = response.text
    
    # Extract the prompt from code blocks
    prompt_match = re.search(r'```(?:markdown|text|txt)?\n(.*?)\n```', optimized_content, re.DOTALL)
    if prompt_match:
        new_prompt_text = prompt_match.group(1).strip()
    else:
        new_prompt_text = optimized_content.strip()

    # Save the new version of EVAL prompt
    match = re.search(r'v(\d+)', version)
    v_num = int(match.group(1)) if match else 0
    next_version = f"v{v_num + 1}"
    
    output_path = f"{base_dir}/eval_prompts/{next_version}-judge.txt"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(new_prompt_text)
    
    logging.info(f"Optimization complete. New EVALUATION prompt saved to: {output_path}")
    print(f"\n--- Optimization Summary ---\n{optimized_content}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, required=True, help="Current version (e.g. v0)")
    parser.add_argument("--log_version", type=str, help="Version tag for the log file (defaults to --version)")
    
    args = parser.parse_args()
    log_ver = args.log_version if args.log_version else args.version
    try:
        optimize_eval_prompt(args.version, log_ver)
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)
