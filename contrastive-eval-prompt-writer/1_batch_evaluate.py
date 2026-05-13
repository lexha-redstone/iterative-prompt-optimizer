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


# gcloud auth application-default login

import os
import json
import time
import logging
import base64
import argparse
from datetime import datetime
from pathlib import Path
from google import genai
from google.genai import types
from google.cloud import storage
import sys
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
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def encode_image(image_path):
    """Encodes an image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def prepare_eval_batch_input(version, input_prompts_file, eval_prompt_template_file, input_jsonl_name):
    """
    Creates a .jsonl file for the evaluation batch job by pairing good/poor samples 
    with their corresponding user inputs.
    """
    logging.info(f"Preparing evaluation batch input...")
    
    with open(input_prompts_file, "r", encoding="utf-8") as f:
        user_inputs = json.load(f)
        
    eval_template = read_file(eval_prompt_template_file)
    
    base_dir = "."
    good_dir = os.path.join(base_dir, "samples", "good")
    poor_dir = os.path.join(base_dir, "samples", "poor")
    
    count = 0
    with open(input_jsonl_name, "w", encoding="utf-8") as outfile:
        # Process both good and poor samples
        for category, directory in [("good", good_dir), ("poor", poor_dir)]:
            if not os.path.exists(directory):
                logging.warning(f"Directory {directory} not found. Skipping.")
                continue
                
            for filename in os.listdir(directory):
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                # The filename (e.g., img-0.png) is the key in user_inputs
                if filename not in user_inputs:
                    logging.warning(f"No user input found for {filename}. Skipping.")
                    continue
                
                user_input_text = user_inputs[filename]
                image_path = os.path.join(directory, filename)
                image_b64 = encode_image(image_path)
                
                eval_prompt = eval_template.replace("&USER_INPUT&", user_input_text)
                
                # Unique key for batch job: eval-{category}-{filename}
                eval_key = f"eval-{category}-{filename}"
                
                request_obj = {
                    "key": eval_key,
                    "request": {
                        "contents": [
                            {
                                "role": "user",
                                "parts": [
                                    {"text": eval_prompt},
                                    {
                                        "inlineData": {
                                            "mimeType": "image/png",
                                            "data": image_b64
                                        }
                                    }
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
        logging.error(f"No valid images and user inputs found to evaluate in {good_dir} and {poor_dir}. Check sample_inputs.json and images.")
        sys.exit(1)
            
    logging.info(f"Created {count} evaluation requests in {input_jsonl_name}")

def upload_blob(bucket, local_path, destination_blob_name):
    """Uploads a file to GCS."""
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)
    logging.info(f"Uploaded {local_path} to gs://{config.GCS_BUCKET_NAME}/{destination_blob_name}")

def run_batch_evaluation(version):
    """Orchestrates the batch evaluation job submission."""
    # Initialization
    client = genai.Client(vertexai=True, project=config.PROJECT_ID, location=config.LOCATION)
    storage_client = storage.Client(project=config.PROJECT_ID)
    bucket = storage_client.bucket(config.GCS_BUCKET_NAME)

    base_dir = "."
    os.makedirs(f"{base_dir}/inputs", exist_ok=True)
    os.makedirs(f"{base_dir}/logs", exist_ok=True)

    # Configs
    input_prompts_file = "./inputs/sample_inputs.json"

    # Use version for the eval prompt filename
    eval_prompt_template_file = f"{base_dir}/eval_prompts/{version}-judge.txt"
    
    timestamp = int(time.time())
    input_jsonl_name = f"{base_dir}/inputs/tmp_batch_input_eval_contrastive_{version}_{timestamp}.jsonl"
    gcs_input_uri = f"gs://{config.GCS_BUCKET_NAME}/PhotoWidget/contrastive/{os.path.basename(input_jsonl_name)}"
    gcs_output_dir = f"gs://{config.GCS_BUCKET_NAME}/PhotoWidget/outputs/eval_contrastive_{version}_{timestamp}/"
    log_file = f"{base_dir}/logs/batch_jobs.log"

    # 1. Prepare local JSONL
    prepare_eval_batch_input(version, input_prompts_file, eval_prompt_template_file, input_jsonl_name)
    
    # 2. Upload to GCS
    upload_blob(bucket, input_jsonl_name, f"PhotoWidget/contrastive/{os.path.basename(input_jsonl_name)}")
    
    # 3. Submit Batch Job
    logging.info(f"Submitting evaluation batch job for model {config.EVALUATION_MODEL}...")
    batch_job = client.batches.create(
        model=config.EVALUATION_MODEL,
        src=gcs_input_uri,
        config=types.CreateBatchJobConfig(
            dest=gcs_output_dir,
            display_name=f"Eval_Contrastive_{version}_{timestamp}"
        )
    )
    
    job_id = batch_job.name
    logging.info(f"Evaluation batch job submitted. ID: {job_id}")
    
    # 4. Record to logs
    log_entry = {
        "type": "evaluation_contrastive",
        "input_uri": gcs_input_uri,
        "output_uri": gcs_output_dir,
        "eval_prompt_template": eval_prompt_template_file,
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
    """Downloads the batch evaluation results from GCS."""
    logging.info(f"Downloading results from {gcs_uri} to {local_base}")
    Path(local_base).mkdir(parents=True, exist_ok=True)
    
    if gcs_uri.startswith("gs://"):
        uri_path = gcs_uri[5:]
        bucket_name, prefix = uri_path.split("/", 1)
    else:
        bucket_name = config.GCS_BUCKET_NAME
        prefix = gcs_uri

    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    
    found = False
    for blob in blobs:
        # Vertex AI Batch outputs usually contain 'prediction' or end with '.jsonl'
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
    client = genai.Client(vertexai=True, project=config.PROJECT_ID, location=config.LOCATION)
    storage_client = storage.Client(project=config.PROJECT_ID)
    output_local_base = f"results/evaluate/{args.version}"

    try:
        job, gcs_output_dir = run_batch_evaluation(args.version)
        if wait_for_job(client, job):
            download_results(storage_client, gcs_output_dir, output_local_base)
        else:
            logging.error("Batch evaluation job failed.")
            sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)
