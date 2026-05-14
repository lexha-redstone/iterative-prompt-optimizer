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

# python 4_optimize.py --version v0
import argparse
import json
import logging
import os
import re
from config import PROJECT_ID, LOCATION, OPTIMIZATION_MODEL
from google import genai
from google.genai import types

# Centralized path configuration for this script
PATHS = {
    "logs_dir": "logs",
    "prompts_dir": "prompts",
    "generated_dir": "generated",
    "eval_logs_jsonl": os.path.join("logs", "eval_logs.jsonl"),
    "meta_prompt_template": os.path.join("prompts", "optimizer_contrastive.txt"),
    "meta_prompt_logs": os.path.join("logs", "meta_prompt_logs.txt"),
    # Templates for dynamic paths
    "current_prompt_template": os.path.join("prompts", "inference_{version}.txt"),
    "critic_results_template": os.path.join("generated", "critic", "{version}", "predictions.jsonl"),
    "output_path_template": os.path.join("prompts", "inference_{next_version}.txt")
}

def setup_logging():
  """Sets up logging to console."""
  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
  )


def read_file(path):
  """Reads a file and returns its content as a string."""
  if not os.path.exists(path):
    logging.error(f"File not found: {path}")
    return ""
  with open(path, "r", encoding="utf-8") as f:
    return f.read()


def get_next_version(current_version):
  """Increments version string (e.g., v0 -> v1)."""
  match = re.search(r"v(\d+)", current_version)
  if match:
    version_num = int(match.group(1))
    return f"v{version_num + 1}"
  return current_version + "_optimized"


def calculate_score_summary_v2(version, eval_log=None):
  """Reads evaluation results from eval_logs.jsonl and returns a summary of field_averages."""
  log_file = (
      eval_log
      if eval_log and os.path.exists(eval_log)
      else PATHS["eval_logs_jsonl"]
  )
  if not os.path.exists(log_file):
    logging.warning(f"Log file not found: {log_file}")
    return "No evaluation data available."

  target_log = None
  try:
    with open(log_file, "r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if not line:
          continue

        if "}{" in line:
          parts = line.replace("}{", "}|||{").split("|||")
        else:
          parts = [line]

        for part in parts:
          try:
            item = json.loads(part)
            if item.get("version") == version:
              target_log = item
          except Exception:
            continue
  except Exception as e:
    logging.error(f"Error reading log file: {e}")
    return "Error reading evaluation data."

  if not target_log:
    logging.warning(
        f"No log entry found for version {version} in"
        f" {log_file}"
    )
    return f"No evaluation data found for version {version}."

  field_averages = target_log.get("field_averages", {})
  overall_average = target_log.get("overall_average", 0)

  summary_lines = []
  # List all available fields alphabetically, excluding overall score
  for field in sorted(field_averages.keys()):
    if field.lower() not in ["overall_score", "overallscore"]:
      summary_lines.append(f"- {field}: {field_averages[field]:.2f}")

  # Use 'overallScore' to maintain compatibility with existing meta-prompts
  summary_lines.append(f"- overallScore: {overall_average:.2f}")

  # Add Cap Information
  if "cap_rate" in target_log:
    summary_lines.append(f"- capRate: {target_log['cap_rate']:.2f}%")
    if target_log.get("cap_reasons"):
      summary_lines.append("#### Critical Failure Reasons (Score Caps):")
      for reason in target_log["cap_reasons"]:
        summary_lines.append(f"  [!] {reason}")

  return "\n".join(summary_lines)


def gather_critic_results(critic_file):
  """Reads critic predictions and concatenates the findings, including Good DNA to preserve."""
  if not os.path.exists(critic_file):
    logging.warning(f"Critic file not found: {critic_file}")
    return "No critic feedback available."

  results = []
  with open(critic_file, "r", encoding="utf-8") as f:
    for line in f:
      if not line.strip():
        continue
      try:
        item = json.loads(line)
        response_parts = item["response"]["candidates"][0]["content"]["parts"]
        text = ""
        for p in response_parts:
          if "text" in p:
            text += p["text"]

        # Try to parse JSON for better structure
        try:
          json_str = text
          if "```json" in text:
            json_str = re.search(r"```json\n(.*?)\n```", text, re.DOTALL).group(
                1
            )
          elif "```" in text:
            json_str = re.search(r"```\n(.*?)\n```", text, re.DOTALL).group(1)

          data = json.loads(json_str)
          structured_text = "#### [!] Good DNA to Preserve:\n" + "\n".join(
              [f"- {x}" for x in data.get("good_dna_to_preserve", [])]
          )
          if "vision_gap_analysis" in data:
              structured_text += f"\n\n#### Vision Gap Analysis:\n{data['vision_gap_analysis']}"
              
          structured_text += "\n\n#### Vulnerabilities:\n" + "\n".join(
              [f"- {x}" for x in data.get("prompt_vulnerabilities", [])]
          )
          structured_text += "\n\n#### Optimization Rules:\n" + "\n".join(
              [f"- {x}" for x in data.get("prompt_optimization_rules", [])]
          )
          results.append(f"### Sample {item.get('key')}:\n{structured_text}")
        except:
          results.append(f"### Sample {item.get('key')}:\n{text}")
      except Exception:
        continue

  return "\n\n".join(results)


def get_best_prompt_info(eval_log=None):
  """Reads all evaluation results and returns the best version name, its score, content, and score summary."""
  log_file = (
      eval_log
      if eval_log and os.path.exists(eval_log)
      else PATHS["eval_logs_jsonl"]
  )
  if not os.path.exists(log_file):
    logging.warning(f"Log file not found for best prompt info: {log_file}")
    return None, 0, "", ""

  history = []
  try:
    with open(log_file, "r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        if "}{" in line:
          parts = line.replace("}{", "}|||{").split("|||")
        else:
          parts = [line]
        for part in parts:
          try:
            item = json.loads(part)
            history.append({
                "version": item.get("version"),
                "score": item.get("overall_average", 0),
            })
          except Exception:
            continue
  except Exception as e:
    logging.error(f"Error reading log file for best prompt: {e}")
    return None, 0, "", ""

  if not history:
    return None, 0, ""

  # Sort by score to find the best version
  sorted_history = sorted(history, key=lambda x: x["score"], reverse=True)
  best_entry = sorted_history[0]

  best_version = best_entry["version"]
  best_score = best_entry["score"]

  # Read the actual prompt file and its score summary
  best_prompt_path = PATHS["current_prompt_template"].format(version=best_version)
  best_prompt_content = read_file(best_prompt_path)
  best_score_summary = calculate_score_summary_v2(
      best_version, eval_log=eval_log
  )

  return best_version, best_score, best_prompt_content, best_score_summary


def compare_prompts(client, best_info, current_info):
  """Calls Gemini to compare the best performing prompt vs the current one.

  Strongly focuses on identifying 'Lost Strengths' to prevent regression.
  """
  best_version, best_score, best_prompt, best_summary = best_info
  current_version, current_score, current_prompt, current_summary = current_info

  prompt = f"""
You are an expert AI prompt analyst. Compare these two versions of an image generation prompt for "Infographics".
The goal is to identify why {best_version} performed better and ensure its "Winning DNA" is not lost.

### Best Performing Version: {best_version} (Score: {best_score:.4f})
#### Detailed Scores:
{best_summary}
#### Prompt Content:
{best_prompt}

### Current Version: {current_version} (Score: {current_score:.4f})
#### Detailed Scores:
{current_summary}
#### Prompt Content:
{current_prompt}

### TASK:
1. **Regression Analysis**: Identify exactly what was present in {best_version} (e.g., specific phrasing, persona details, layout templates) that was removed or weakened in {current_version}.
2. **Geometric Anchor Check**: Look for specific pixel values (e.g., "120px margins"), margin rules, or structural dimensions that were lost or generalized.
3. **Impact Assessment**: Match the score drops to these changes (e.g., "Score for [Criterion] dropped because the explicit grid instruction was replaced by a vague 'symmetrical' rule").
4. **Preservation List**: List the "Must-Restore" elements from {best_version}.
5. **Good Points of Current**: Identify if {current_version} improved anything at all (to keep those improvements).

Output in a clear, structured Markdown format.
"""
  logging.info(
      f"Comparing {best_version} vs {current_version} for regression check..."
  )
  response = client.models.generate_content(
      model=OPTIMIZATION_MODEL, contents=prompt
  )
  return response.text


def format_optimization_context(client, raw_context):
  """Calls Gemini to normalize and structure all inputs into a consistent, easy-to-read format for the Meta Prompt."""
  prompt = f"""
You are a context organizer for a Prompt Engineering Meta-process. 
Below is a collection of raw data including current prompt, evaluation scores, critic results, and comparison analysis.

### RAW DATA:
{raw_context}

### TASK:
Organize this information into a single, highly structured, and consistent Markdown document. 
- Use clear headings and bullet points.
- Ensure all technical terms are consistent (especially evaluation criteria and specific constraints).
- Summarize long critic results into actionable patterns.
- Highlight the "Best vs Current" comparison and any "Cap" triggers (1.0 or 0.0 scores) clearly.
- This document will be the primary context for a Meta-Prompt Optimizer.

DO NOT optimize the prompt yet. ONLY format and structure the data.
"""
  logging.info(
      "Normalizing and formatting optimization context using Gemini..."
  )
  response = client.models.generate_content(
      model=OPTIMIZATION_MODEL, contents=prompt
  )
  return response.text


def optimize_prompt(version, eval_log=None):
  """Main function to run the optimization loop."""
  setup_logging()
  client = genai.Client(
      vertexai=True, project=PROJECT_ID, location=LOCATION
  )

  current_prompt_path = PATHS["current_prompt_template"].format(version=version)
  critic_results_path = PATHS["critic_results_template"].format(version=version)
  meta_prompt_path = PATHS["meta_prompt_template"]

  logging.info(f"Gathering data for version: {version}")
  current_prompt = read_file(current_prompt_path)
  meta_prompt_template = read_file(meta_prompt_path)

  if not current_prompt or not meta_prompt_template:
    logging.error("Required prompt files are missing. Aborting.")
    return

  evaluation_scores = calculate_score_summary_v2(
      version, eval_log=eval_log
  )
  critic_results = gather_critic_results(critic_results_path)

  # 1. Trajectory & Best Prompt Analysis
  best_info = get_best_prompt_info(eval_log=eval_log)
  if best_info and best_info[0]:
      best_v, best_s, best_p, best_sum = best_info
  else:
      best_v, best_s, best_p, best_sum = None, 0, "", ""

  # Extract current score as float for comparison
  current_score = 0
  match = re.search(r"overallScore: ([\d.]+)", evaluation_scores)
  if match:
    current_score = float(match.group(1))

  comparison_analysis = ""
  if best_v and best_v != version:
    comparison_analysis = compare_prompts(
        client,
        (best_v, best_s, best_p, best_sum),
        (version, current_score, current_prompt, evaluation_scores),
    )
  else:
    comparison_analysis = (
        "Current version is the best performing version so far or no previous"
        " history found."
    )

  # 2. Format Context for Meta Prompt
  raw_context = f"""
## 1. CURRENT PROMPT ({version})
{current_prompt}

## 2. EVALUATION SCORES ({version})
{evaluation_scores}

## 3. CRITIC FEEDBACK ({version})
{critic_results}

## 4. COMPARISON ANALYSIS (Best vs Current)
{comparison_analysis}
"""

  structured_context = format_optimization_context(client, raw_context)

  # 3. Final Optimization Call
  final_meta_prompt = meta_prompt_template.format(
      structured_context=structured_context,
      current_prompt=current_prompt,
      evaluation_scores=evaluation_scores,
      critic_results=critic_results,
      comparison_analysis=comparison_analysis,
  )

  # Log the final meta prompt for debugging
  os.makedirs(PATHS["logs_dir"], exist_ok=True)
  log_file_path = PATHS["meta_prompt_logs"]
  with open(log_file_path, "a", encoding="utf-8") as f:
    f.write(
        f"\n{'='*50}\nVERSION: {version}\n{'='*50}\n\n{final_meta_prompt}\n\n"
    )

  logging.info(f"Calling Gemini API for final prompt optimization...")
  response = client.models.generate_content(
      model=OPTIMIZATION_MODEL, contents=final_meta_prompt
  )

  optimized_content = response.text

  # Extract the actual prompt from the response
  prompt_match = re.search(
      r"```(?:markdown|text|txt)?\n(.*?)\n```", optimized_content, re.DOTALL
  )
  if prompt_match:
    new_prompt_text = prompt_match.group(1).strip()
  else:
    # Fallback if no code block is found
    new_prompt_text = optimized_content.strip()

  next_version = get_next_version(version)
  output_path = PATHS["output_path_template"].format(next_version=next_version)

  with open(output_path, "w", encoding="utf-8") as f:
    f.write(new_prompt_text)

  logging.info(f"Optimization complete. New prompt saved to: {output_path}")
  print(f"\n--- Optimization Analysis for {next_version} ---\n")
  # Print only the analysis part to the console
  analysis_part = optimized_content.split("```")[0]
  print(analysis_part)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description=(
          "Optimize image generation prompts using evaluation and critic"
          " results."
      )
  )
  parser.add_argument(
      "--version",
      type=str,
      required=True,
      help="Current prompt version (e.g., v0)",
  )
  parser.add_argument(
      "--eval_log",
      type=str,
      default=None,
      help="Path to session-specific evaluation log",
  )

  args = parser.parse_args()
  optimize_prompt(args.version, eval_log=args.eval_log)
