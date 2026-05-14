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


# python 2_critic.py --version v0
import os
import json
import time
import logging
import base64
import argparse
import re
from datetime import datetime
from pathlib import Path
from google import genai
from google.genai import types
from google.cloud import storage
import sys
from config import (
    PROJECT_ID, LOCATION, GCS_BUCKET_NAME, CRITIC_MODEL
)

# Centralized path configuration using ABSOLUTE paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATHS = {
    "logs_dir": os.path.join(BASE_DIR, "logs"),
    "inputs_dir": os.path.join(BASE_DIR, "inputs"),
    "eval_prompts_dir": os.path.join(BASE_DIR, "eval_prompts"),
    "results_dir": os.path.join(BASE_DIR, "results"),
    "log_file_template": os.path.join(BASE_DIR, "logs", "run_contrastive_{version}.log"),
    "judge_eval_logs": os.path.join(BASE_DIR, "logs", "judge_eval_logs.jsonl"),
    "sample_inputs": os.path.join(BASE_DIR, "inputs", "sample_inputs.json"),
    "critic_template": os.path.join(BASE_DIR, "eval_prompts", "critic_contrastive.txt"),
    "good_samples_dir": os.path.join(BASE_DIR, "samples", "good"),
    "poor_samples_dir": os.path.join(BASE_DIR, "samples", "poor"),
    "batch_jobs_log": os.path.join(BASE_DIR, "logs", "batch_jobs.log"),
    # Templates for dynamic paths
    "best_prompt_template": os.path.join(BASE_DIR, "eval_prompts", "{best_version}-judge.txt"),
    "eval_results_dir_template": os.path.join(BASE_DIR, "results", "evaluate", "{version}"),
    "input_jsonl_template": os.path.join(BASE_DIR, "inputs", "tmp_batch_input_critic_contrastive_{version}_{timestamp}.jsonl"),
    "output_local_base_template": os.path.join(BASE_DIR, "results", "critic", "{version}")
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

def read_file(path):
    """Reads a file and returns its content as a string."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def encode_image(image_path):
    """Encodes an image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_eval_results(eval_dir):
    """Loads all evaluation results from the directory."""
    results = {}
    if not os.path.exists(eval_dir):
        logging.warning(f"Evaluation directory {eval_dir} does not exist.")
        return results
        
    for filename in os.listdir(eval_dir):
        if not filename.endswith(".jsonl"):
            continue
        with open(os.path.join(eval_dir, filename), "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    eval_key = item.get("key")
                    if not eval_key:
                        continue
                    
                    # Remove 'eval-' prefix to get the original key (category-filename)
                    key = eval_key.replace("eval-", "") if eval_key.startswith("eval-") else eval_key

                    # Extract response text
                    parts = item["response"]["candidates"][0]["content"]["parts"]
                    text = ""
                    for p in parts:
                        if "text" in p:
                            text += p["text"]
                    
                    if text.startswith("```json"):
                        text = text.strip("```json").strip("```").strip()
                    elif text.startswith("```"):
                        text = text.strip("```").strip()
                    
                    eval_data = json.loads(text)
                    if isinstance(eval_data, list):
                        eval_data = eval_data[0]
                    
                    # Calculate overall_score as the average of all detected category scores (C1_score, C2_score, etc.)
                    category_scores = []
                    for k, v in eval_data.items():
                        # Match keys like C1_score, C2_score, C10_score, etc.
                        if re.match(r'^C\d+_score$', k):
                            try:
                                category_scores.append(float(v))
                            except (ValueError, TypeError):
                                pass
                    
                    if category_scores:
                        eval_data["overall_score"] = sum(category_scores) / len(category_scores)
                    else:
                        # Fallback to existing overall_score or 0 if no category scores found
                        eval_data["overall_score"] = eval_data.get("overall_score", eval_data.get("overallScore", 0))
                    
                    results[key] = eval_data
                except (KeyError, IndexError, json.JSONDecodeError, TypeError):
                    continue
                    
    return results

def log_score_results(version, avg_good, avg_poor):
    """Logs the evaluation summary scores to a central JSONL file."""
    log_file = PATHS["judge_eval_logs"]
    os.makedirs(PATHS["logs_dir"], exist_ok=True)
    
    log_entry = {
        "version": version,
        "avg_good": round(avg_good, 4),
        "avg_poor": round(avg_poor, 4),
        "avg_diff": round(avg_good - avg_poor, 4),
        "timestamp": datetime.now().isoformat()
    }
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
    logging.info(f"Recorded scores to {log_file}")

def get_best_judge_prompt_info():
    """Reads judge_eval_logs.jsonl and returns the best version and its prompt content."""
    log_file = PATHS["judge_eval_logs"]
    if not os.path.exists(log_file):
        return None, 0, ""

    history = []
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    history.append({
                        "version": item.get("version"),
                        "score": item.get("avg_diff", 0),
                    })
                except Exception:
                    continue
    except Exception as e:
        logging.error(f"Error reading log file: {e}")
        return None, 0, ""

    if not history:
        return None, 0, ""

    # Sort by score to find the best version
    sorted_history = sorted(history, key=lambda x: x["score"], reverse=True)
    best_entry = sorted_history[0]

    best_version = best_entry["version"]
    best_score = best_entry["score"]

    # Read the actual prompt file
    best_prompt_path = PATHS["best_prompt_template"].format(best_version=best_version)
    if os.path.exists(best_prompt_path):
        with open(best_prompt_path, "r", encoding="utf-8") as f:
            best_prompt_content = f.read()
    else:
        best_prompt_content = ""

    return best_version, best_score, best_prompt_content

def prepare_critic_batch_input(version, eval_results, input_prompts_file, critic_template_file, input_jsonl_name):
    """
    Pairs good and poor samples and prepares the critic batch input.
    """
    logging.info("Preparing critic batch input...")
    
    with open(input_prompts_file, "r", encoding="utf-8") as f:
        user_inputs = json.load(f)
        
    critic_template = read_file(critic_template_file)
    
    # Get best judge prompt info for reference
    best_v, best_s, best_p = get_best_judge_prompt_info()
    best_judge_context = ""
    if best_v:
        best_judge_context = (
            f"### REFERENCE: BEST PERFORMING JUDGE PROMPT ({best_v})\n"
            f"This judge prompt achieved the highest score gap ({best_s:.2f}) in previous iterations. "
            "Use it to understand what kind of instructions and rubrics are effective.\n\n"
            f"{best_p}\n\n"
        )

    good_dir = PATHS["good_samples_dir"]
    poor_dir = PATHS["poor_samples_dir"]
    
    good_scores = []
    poor_scores = []
    
    count = 0
    with open(input_jsonl_name, "w", encoding="utf-8") as outfile:
        for filename, user_input_text in user_inputs.items():
            good_key = f"good-{filename}"
            poor_key = f"poor-{filename}"
            
            if good_key not in eval_results or poor_key not in eval_results:
                continue
            
            good_eval = eval_results[good_key]
            poor_eval = eval_results[poor_key]
            
            good_score = float(good_eval.get("overall_score", 0))
            poor_score = float(poor_eval.get("overall_score", 0))
            
            good_scores.append(good_score)
            poor_scores.append(poor_score)
            
            # Prepare Critic Prompt
            score_diff = good_score - poor_score
            critic_prompt = critic_template.replace("&USER_INPUT&", user_input_text)
            critic_prompt = critic_prompt.replace("&GOOD_EVAL&", json.dumps(good_eval, indent=2))
            critic_prompt = critic_prompt.replace("&POOR_EVAL&", json.dumps(poor_eval, indent=2))
            critic_prompt = critic_prompt.replace("&GOOD_SCORE&", f"{good_score:.2f}")
            critic_prompt = critic_prompt.replace("&POOR_SCORE&", f"{poor_score:.2f}")
            critic_prompt = critic_prompt.replace("&SCORE_DIFF&", f"{score_diff:.2f}")
            
            # Inject best judge context if available
            if best_v:
                if "&BEST_JUDGE_PROMPT&" in critic_prompt:
                    critic_prompt = critic_prompt.replace("&BEST_JUDGE_PROMPT&", best_judge_context)
                else:
                    critic_prompt = best_judge_context + critic_prompt

            good_img_path = os.path.join(good_dir, filename)
            poor_img_path = os.path.join(poor_dir, filename)
            
            if not os.path.exists(good_img_path) or not os.path.exists(poor_img_path):
                continue
                
            good_img_b64 = encode_image(good_img_path)
            poor_img_b64 = encode_image(poor_img_path)
            
            critic_key = f"critic-{filename}"
            request_obj = {
                "key": critic_key,
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "inlineData": {
                                        "mimeType": "image/png",
                                        "data": good_img_b64
                                    }
                                },
                                {
                                    "inlineData": {
                                        "mimeType": "image/png",
                                        "data": poor_img_b64
                                    }
                                },
                                {"text": critic_prompt}
                            ]
                        }
                    ],
                    "generationConfig": {
                        "response_mime_type": "application/json"
                    }
                }
            }
            outfile.write(json.dumps(request_obj) + "\n")
            count += 1
    
    if count == 0:
        logging.error(f"No valid images and user inputs found to critique in {good_dir} and {poor_dir}. Check eval results and images.")
        sys.exit(1)
            
    if good_scores and poor_scores:
        avg_good = sum(good_scores) / len(good_scores)
        avg_poor = sum(poor_scores) / len(poor_scores)
        logging.info(f"[{version}] Average Scores - Good: {avg_good:.2f}, Poor: {avg_poor:.2f}, Diff: {avg_good - avg_poor:.2f}")
        
        # Record to score log
        log_score_results(version, avg_good, avg_poor)
    
    logging.info(f"Created {count} critic requests in {input_jsonl_name}")

def upload_blob(bucket, local_path, destination_blob_name):
    """Uploads a file to GCS."""
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)
    logging.info(f"Uploaded {local_path} to gs://{GCS_BUCKET_NAME}/{destination_blob_name}")

def run_batch_critic(version):
    """Orchestrates the batch critic job submission."""
    # Initialization
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(GCS_BUCKET_NAME)

    os.makedirs(PATHS["inputs_dir"], exist_ok=True)
    os.makedirs(PATHS["logs_dir"], exist_ok=True)

    # Configs
    eval_results_dir = PATHS["eval_results_dir_template"].format(version=version)
    input_prompts_file = PATHS["sample_inputs"]
    critic_template_file = PATHS["critic_template"]
    
    eval_results = load_eval_results(eval_results_dir)
    
    timestamp = int(time.time())
    input_jsonl_name = PATHS["input_jsonl_template"].format(version=version, timestamp=timestamp)
    gcs_input_uri = f"gs://{GCS_BUCKET_NAME}/PhotoWidget/contrastive/{os.path.basename(input_jsonl_name)}"
    gcs_output_dir = f"gs://{GCS_BUCKET_NAME}/PhotoWidget/outputs/critic_contrastive_{version}_{timestamp}/"
    log_file = PATHS["batch_jobs_log"]

    # 1. Prepare local JSONL
    prepare_critic_batch_input(version, eval_results, input_prompts_file, critic_template_file, input_jsonl_name)
    
    # 2. Upload to GCS
    upload_blob(bucket, input_jsonl_name, f"PhotoWidget/contrastive/{os.path.basename(input_jsonl_name)}")
    
    # 3. Submit Batch Job
    logging.info(f"Submitting critic batch job for model {CRITIC_MODEL}...")
    batch_job = client.batches.create(
        model=CRITIC_MODEL,
        src=gcs_input_uri,
        config=types.CreateBatchJobConfig(
            dest=gcs_output_dir,
            display_name=f"Critic_Contrastive_{version}_{timestamp}"
        )
    )
    
    job_id = batch_job.name
    logging.info(f"Critic batch job submitted. ID: {job_id}")
    
    # 4. Record to logs
    log_entry = {
        "type": "critic_contrastive",
        "input_uri": gcs_input_uri,
        "output_uri": gcs_output_dir,
        "job_id": job_id,
        "timestamp": datetime.now().isoformat()
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    return batch_job, gcs_output_dir

def wait_for_job(client, batch_job):
    """Polls the batch job status until completion."""
    logging.info(f"Waiting for batch job {batch_job.name} to complete...")
    while True:
        job = client.batches.get(name=batch_job.name)
        state = job.state.name if hasattr(job.state, 'name') else str(job.state)
        logging.info(f"Current state: {state}")
        
        if "SUCCEEDED" in state:
            logging.info("Batch job completed successfully.")
            return True
        elif any(fail_state in state for fail_state in ["FAILED", "CANCELLED", "EXPIRED"]):
            logging.error(f"Batch job ended with state: {state}")
            return False
        
        time.sleep(60)


def download_results(storage_client, gcs_uri, local_base):
    """Downloads the batch critic results from GCS."""
    logging.info(f"Downloading results from {gcs_uri} to {local_base}")
    Path(local_base).mkdir(parents=True, exist_ok=True)
    
    if gcs_uri.startswith("gs://"):
        uri_path = gcs_uri[5:]
        bucket_name, prefix = uri_path.split("/", 1)
    else:
        bucket_name = GCS_BUCKET_NAME
        prefix = gcs_uri

    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    
    found = False
    for blob in blobs:
        if ".jsonl" in blob.name or "prediction" in blob.name:
            found = True
            filename = os.path.basename(blob.name)
            local_path = os.path.join(local_base, filename)
            blob.download_to_filename(local_path)
            logging.info(f"Downloaded {filename} to {local_path}")
    
    if not found:
        logging.warning("No result files found in the output GCS directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--log_version", type=str, help="Version tag for the log file (defaults to --version)")
    args = parser.parse_args()

    log_ver = args.log_version if args.log_version else args.version
    setup_logging(log_ver)
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    storage_client = storage.Client(project=PROJECT_ID)
    output_local_base = PATHS["output_local_base_template"].format(version=args.version)

    try:
        job, gcs_output_dir = run_batch_critic(args.version)
        if wait_for_job(client, job):
            download_results(storage_client, gcs_output_dir, output_local_base)
        else:
            logging.error("Batch critic job failed.")
            sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)
