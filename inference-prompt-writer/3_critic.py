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

import argparse
import base64
from datetime import datetime
import json
import logging
import os
import random
from pathlib import Path
import time
import config
from google import genai
from google.cloud import storage
from google.genai import types


def setup_logging():
  """Sets up logging to console."""
  os.makedirs("logs", exist_ok=True)
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - %(levelname)s - %(message)s",
      handlers=[logging.StreamHandler()],
  )


def read_file(path):
  """Reads a file and returns its content as a string."""
  with open(path, "r", encoding="utf-8") as f:
    return f.read()


def upload_blob(bucket, local_path, destination_blob_name):
  """Uploads a file to GCS."""
  blob = bucket.blob(destination_blob_name)
  blob.upload_from_filename(local_path)
  logging.info(
      f"Uploaded {local_path} to gs://{bucket.name}/{destination_blob_name}"
  )

def encode_image_file(path):
  """Encodes an image file to base64."""
  with open(path, "rb") as f:
    return base64.b64encode(f.read()).decode("utf-8")

def parse_eval_results(eval_file, create_file):
  """Parses evaluation results and creation data.
  Recalculates overall_score as the arithmetic mean of all found numeric scores.
  """
  logging.info(f"Parsing evaluation results from {eval_file}")

  create_data = {}
  if os.path.exists(create_file):
    with open(create_file, "r", encoding="utf-8") as f:
      for line in f:
        if not line.strip():
          continue
        item = json.loads(line)
        key = item.get("key")

        user_prompt = "N/A"
        try:
          contents = item["request"]["contents"]
          for content in reversed(contents):
            if content.get("role") == "user":
              for part in content.get("parts", []):
                if "text" in part and part["text"]:
                  user_prompt = part["text"]
                  break
              if user_prompt != "N/A":
                break
        except (KeyError, IndexError, TypeError):
          pass

        create_data[key] = {"user_prompt": user_prompt}

  samples = []
  with open(eval_file, "r", encoding="utf-8") as f:
    for line in f:
      if not line.strip():
        continue
      item = json.loads(line)
      eval_key = item.get("key")
      original_key = (
          eval_key.replace("eval-", "")
          if eval_key.startswith("eval-")
          else eval_key
      )

      image_b64 = None
      try:
        parts = item["request"]["contents"][0]["parts"]
        for part in parts:
          inline_data = part.get("inlineData")
          if inline_data and isinstance(inline_data, dict):
            image_b64 = inline_data.get("data")
            if image_b64:
              break
      except:
        pass

      try:
        response_parts = item["response"]["candidates"][0]["content"]["parts"]
        response_text = ""
        for p in response_parts:
          if "text" in p:
            response_text += p["text"]

        if response_text.startswith("```json"):
          response_text = response_text.strip("```json").strip("```").strip()
        elif response_text.startswith("```"):
          response_text = response_text.strip("```").strip()

        eval_json = json.loads(response_text)
        if isinstance(eval_json, list):
          eval_json = eval_json[0]

        # --- DYNAMIC SCORE RECALCULATION (Arithmetic Mean of all found scores) ---
        scores = {}
        # Try to find a nested dictionary first, fallback to the top-level object
        eval_dict = eval_json.get("evaluation", eval_json.get("scores"))
        if eval_dict is None or not isinstance(eval_dict, dict):
            eval_dict = eval_json
        
        numeric_scores = []
        # Support both {"c1_score": 5.0} and {"c1": {"score": 5.0}}
        for k, v in eval_dict.items():
            score_val = None
            if isinstance(v, dict) and "score" in v:
                score_val = v["score"]
            elif isinstance(v, (int, float)):
                # Exclude keys that are likely not category scores if they don't contain 'score' or 'C\d'
                # but in a flat structure like v7-judge, we have C1_score, C2_score etc.
                if "score" in k.lower() or any(c in k for c in ["C1", "C2", "C3", "C4", "C5"]):
                    score_val = v
            
            if score_val is not None:
                try:
                    val = float(score_val)
                    numeric_scores.append(val)
                    scores[k] = v
                except (ValueError, TypeError):
                    continue
        
        if numeric_scores:
            recalculated_overall = sum(numeric_scores) / len(numeric_scores)
        else:
            recalculated_overall = 0.0
            
        scores["overall_score"] = recalculated_overall
        # --------------------------------------------------------------------------

        samples.append({
            "key": original_key,
            "user_prompt": (
                create_data.get(original_key, {}).get("user_prompt", "N/A")
            ),
            "image_b64": image_b64,
            "eval_result": response_text,
            "scores": scores,
            "cap_applied": eval_json.get("cap_applied", False),
            "cap_reason": eval_json.get("cap_reason", ""),
            "critical_failures": eval_json.get("critical_failures", []),
        })

      except (KeyError, IndexError, json.JSONDecodeError):
        continue
  return samples


def print_and_log_score_summary_v2(samples, version, eval_log=None):
  """Calculates total and average scores per field and logs them.
  Recalculated scores are used.
  """

  if not samples:
    logging.warning("No samples to summarize.")
    return

  all_fields = set()
  for s in samples:
    all_fields.update(s["scores"].keys())

  all_fields = sorted(list(all_fields))
  field_totals = {field: 0.0 for field in all_fields}
  field_counts = {field: 0 for field in all_fields}

  cap_count = 0
  cap_reasons = []

  for s in samples:
    if s.get("cap_applied"):
      cap_count += 1
      if s.get("cap_reason"):
        cap_reasons.append(s["cap_reason"])

    for field in all_fields:
      if field in s["scores"]:
        score_obj = s["scores"][field]
        if isinstance(score_obj, dict):
          score_val = score_obj.get("score", 0)
        else:
          score_val = score_obj
        field_totals[field] += float(score_val)
        field_counts[field] += 1

  field_averages = {
      field: (
          round(field_totals[field] / field_counts[field], 3)
          if field_counts[field] > 0
          else 0
      )
      for field in all_fields
  }

  overall_avg = field_averages.get("overall_score", 0)
  cap_rate = (cap_count / len(samples)) * 100 if samples else 0

  print("\n" + "=" * 80)
  print(f"SCORE SUMMARY (v2 - Recalculated) - Version: {version}")
  print("-" * 80)
  print(f"{'Field':<55} | {'Total':<10} | {'Average':<10}")
  print("-" * 80)
  for field in all_fields:
    if field.lower() in ["overall_score", "overallscore"]:
      continue
    print(
        f"{field:<55} | {field_totals[field]:<10.2f} |"
        f" {field_averages[field]:<10.2f}"
    )
  print("-" * 80)
  print(f"{'OVERALL AVERAGE (Arithmetic)':<55} | {'':<10} | {overall_avg:<10.2f}")
  print(
      f"{'CAP RATE':<55} | {'':<10} | {cap_rate:<10.2f}%"
      f" ({cap_count}/{len(samples)})"
  )
  print("=" * 80 + "\n")

  log_entry = {
      "version": version,
      "timestamp": datetime.now().isoformat(),
      "sample_count": len(samples),
      "field_totals": field_totals,
      "field_averages": field_averages,
      "overall_average": overall_avg,
      "cap_rate": cap_rate,
      "cap_count": cap_count,
      "cap_reasons": cap_reasons[:10],
      "is_v2": True,
  }
  os.makedirs("logs", exist_ok=True)
  with open("logs/eval_logs.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(log_entry) + "\n")

  if eval_log:
    with open(eval_log, "a", encoding="utf-8") as f:
      f.write(json.dumps(log_entry) + "\n")


def get_contrastive_samples(samples, top_num=2, bottom_num=3):
  """Finds best and worst samples for contrastive analysis."""
  if not samples:
    return [], []

  def get_overall_score(x):
    return float(x["scores"].get("overall_score", 0))

  sorted_samples = sorted(samples, key=get_overall_score, reverse=True)
  top_samples = sorted_samples[:top_num] if top_num > 0 else []
  bottom_samples = sorted_samples[-bottom_num:] if bottom_num > 0 else []

  return top_samples, bottom_samples


def run_critic_batch(version, eval_log=None):
  """Orchestrates the critic batch job with Golden Standard images."""
  setup_logging()
  logging.info(f"Starting contrastive critic (Golden Standard) for version: {version}.")

  eval_file = f"generated/evaluate/{version}/predictions.jsonl"
  create_file = f"generated/create/{version}/predictions.jsonl"
  critic_prompt_file = config.CRITIC_PROMPT
  output_local_base = f"generated/critic/{version}"
  os.makedirs(output_local_base, exist_ok=True)

  if not os.path.exists(critic_prompt_file):
    logging.error(f"Critic prompt file not found: {critic_prompt_file}")
    return

  samples = parse_eval_results(eval_file, create_file)
  if not samples:
    logging.error("No samples found to analyze.")
    return

  print_and_log_score_summary_v2(samples, version, eval_log=eval_log)

  # Get Poor samples (Bottom 3)
  _, bottom_samples = get_contrastive_samples(samples, top_num=0, bottom_num=3)

  # --- PREPARE FALLBACK (Batch Top 1) ---
  top_samples, _ = get_contrastive_samples(samples, top_num=1, bottom_num=0)
  fallback_b64 = top_samples[0]["image_b64"] if top_samples else None

  golden_dir = config.GOLDEN_STANDARD_DIR
  critic_template = read_file(critic_prompt_file)
  timestamp = int(time.time())
  input_jsonl_name = f"inputs/tmp_batch_input_critic_{version}_{timestamp}.jsonl"

  with open(input_jsonl_name, "w", encoding="utf-8") as f:
    for s in bottom_samples:
      # --- 1:1 MATCHING GOLDEN STANDARD ---
      current_golden_b64 = None
      ref_context = ""
      
      if golden_dir and os.path.exists(golden_dir):
          # 1. Exact matching filename
          specific_golden_path = os.path.join(golden_dir, s["key"])
          if os.path.exists(specific_golden_path):
              current_golden_b64 = encode_image_file(specific_golden_path)
              ref_context = (
                  "### REFERENCE TYPE: GOLDEN GROUND TRUTH (Exact Match)\n"
                  "This is a PERFECT execution for the EXACT same user prompt. Use this as a direct comparison "
                  "to identify structural and aesthetic failures in the target image."
              )
              logging.info(f"Matched Golden Standard for {s['key']}")
          else:
              # 2. Random file in golden_dir
              files = [f for f in os.listdir(golden_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
              if files:
                  selected_f = random.choice(files)
                  current_golden_b64 = encode_image_file(os.path.join(golden_dir, selected_f))
                  ref_context = (
                      "### REFERENCE TYPE: STYLE REFERENCE (Same Art Direction, Different Content)\n"
                      "This image has different content but represents the PERFECT visual execution and "
                      "art direction for this mode. Analyze why the target image fails to match this aesthetic quality."
                  )
                  logging.warning(f"Specific Golden Standard NOT found for {s['key']}. Using random fallback: {selected_f}")

      if not current_golden_b64:
          # 3. Batch Top 1
          logging.warning(f"No Golden Standard available in directory. Using Batch Top fallback for {s['key']}.")
          current_golden_b64 = fallback_b64
          ref_context = (
              "### REFERENCE TYPE: RELATIVE BENCHMARK (Best of Current Batch)\n"
              "This is the best performing image from the current generation batch. It may not be perfect, "
              "but it is significantly better than the target. Analyze the gap between this benchmark and the target failure."
          )

      if not current_golden_b64:
          logging.error(f"Critical: No reference image (Golden or Top) found for {s['key']}. Skipping.")
          continue

      prompt_text = critic_template.replace("&USER_PROMPT&", s["user_prompt"])
      prompt_text = prompt_text.replace("&EVAL_RESULT&", s["eval_result"])

      instruction_prefix = (
          f"{ref_context}\n\n"
          "### TARGET FOR ANALYSIS\n"
      )

      parts = []
      # Reference Image first
      parts.append({
          "inlineData": {"mimeType": "image/png", "data": current_golden_b64}
      })
      
      # Target Poor Image
      parts.append({
          "inlineData": {"mimeType": "image/png", "data": s["image_b64"]}
      })
      
      # Instruction text last
      parts.append({"text": instruction_prefix + prompt_text})

      request_obj = {
          "key": f"critic-{s['key']}",
          "request": {
              "contents": [{"role": "user", "parts": parts}],
              "generationConfig": {"response_mime_type": "application/json"},
          },
      }
      f.write(json.dumps(request_obj) + "\n")

  client = genai.Client(
      vertexai=True, project=config.PROJECT_ID, location=config.LOCATION
  )
  storage_client = storage.Client(project=config.PROJECT_ID)
  bucket = storage_client.bucket(config.GCS_BUCKET_NAME)

  gcs_input_uri = f"gs://{config.GCS_BUCKET_NAME}/ATL/{input_jsonl_name}"
  upload_blob(bucket, input_jsonl_name, f"ATL/{input_jsonl_name}")

  gcs_output_dir = f"gs://{config.GCS_BUCKET_NAME}/ATL/outputs/critic_{version}_{timestamp}/"

  logging.info(f"Starting batch job for critic...")
  job = client.batches.create(
      model=config.CRITIC_MODEL,
      src=gcs_input_uri,
      config=types.CreateBatchJobConfig(
          dest=gcs_output_dir,
          display_name=f"Critic_{version}_{timestamp}",
      ),
  )

  while True:
    job = client.batches.get(name=job.name)
    state = job.state.name if hasattr(job.state, "name") else str(job.state)
    if state == "JOB_STATE_SUCCEEDED":
      break
    elif state in ["FAILED", "CANCELLED"]:
      logging.error(f"Batch job failed: {state}")
      return
    else:
      time.sleep(30)

  blobs = bucket.list_blobs(prefix=f"ATL/outputs/critic_{version}_{timestamp}/")
  for blob in blobs:
    if "predictions.jsonl" in blob.name:
      local_output_path = os.path.join(output_local_base, "predictions.jsonl")
      blob.download_to_filename(local_output_path)
      break


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--version", type=str, required=True)
  parser.add_argument("--eval_log", type=str, default=None)
  args = parser.parse_args()
  run_critic_batch(args.version, eval_log=args.eval_log)
