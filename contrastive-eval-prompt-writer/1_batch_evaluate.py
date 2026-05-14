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
from config import (
    PROJECT_ID, LOCATION, GCS_BUCKET_NAME, EVALUATION_MODEL
)

# Centralized path configuration using ABSOLUTE paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATHS = {
    "logs_dir": os.path.join(BASE_DIR, "logs"),
    "inputs_dir": os.path.join(BASE_DIR, "inputs"),
    "eval_prompts_dir": os.path.join(BASE_DIR, "eval_prompts"),
    "results_dir": os.path.join(BASE_DIR, "results"),
    "log_file_template": os.path.join(BASE_DIR, "logs", "run_contrastive_{version}.log"),
    "sample_inputs": os.path.join(BASE_DIR, "inputs", "sample_inputs.json"),
    "good_dir": os.path.join(BASE_DIR, "samples", "good"),
    "poor_dir": os.path.join(BASE_DIR, "samples", "poor"),
    "batch_jobs_log": os.path.join(BASE_DIR, "logs", "batch_jobs.log"),
    # Templates for dynamic paths
    "eval_prompt_template": os.path.join(BASE_DIR, "eval_prompts", "{version}-judge.txt"),
    "input_jsonl_template": os.path.join(BASE_DIR, "inputs", "tmp_batch_input_eval_contrastive_{version}_{timestamp}.jsonl"),
    "output_local_base_template": os.path.join(BASE_DIR, "results", "evaluate", "{version}")
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

def prepare_eval_batch_input(version, input_prompts_file, eval_prompt_template_file, input_jsonl_name):
    """
    Creates a .jsonl file for the evaluation batch job by pairing good/poor samples 
    with their corresponding user inputs.
    """
    logging.info(f"Preparing evaluation batch input...")
    
    with open(input_prompts_file, "r", encoding="utf-8") as f:
        user_inputs = json.load(f)
        
    eval_template = read_file(eval_prompt_template_file)
    
    good_dir = PATHS["good_dir"]
    poor_dir = PATHS["poor_dir"]
    
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
    logging.info(f"Uploaded {local_path} to gs://{GCS_BUCKET_NAME}/{destination_blob_name}")

def run_batch_evaluation(version):
    """Orchestrates the batch evaluation job submission."""
    # Initialization
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(GCS_BUCKET_NAME)

    os.makedirs(PATHS["inputs_dir"], exist_ok=True)
    os.makedirs(PATHS["logs_dir"], exist_ok=True)

    # Configs
    input_prompts_file = PATHS["sample_inputs"]

    # Use version for the eval prompt filename
    eval_prompt_template_file = PATHS["eval_prompt_template"].format(version=version)
    
    timestamp = int(time.time())
    input_jsonl_name = PATHS["input_jsonl_template"].format(version=version, timestamp=timestamp)
    gcs_input_uri = f"gs://{GCS_BUCKET_NAME}/PhotoWidget/contrastive/{os.path.basename(input_jsonl_name)}"
    gcs_output_dir = f"gs://{GCS_BUCKET_NAME}/PhotoWidget/outputs/eval_contrastive_{version}_{timestamp}/"
    log_file = PATHS["batch_jobs_log"]

    # 1. Prepare local JSONL
    prepare_eval_batch_input(version, input_prompts_file, eval_prompt_template_file, input_jsonl_name)
    
    # 2. Upload to GCS
    upload_blob(bucket, input_jsonl_name, f"PhotoWidget/contrastive/{os.path.basename(input_jsonl_name)}")
    
    # 3. Submit Batch Job
    logging.info(f"Submitting evaluation batch job for model {EVALUATION_MODEL}...")
    batch_job = client.batches.create(
        model=EVALUATION_MODEL,
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
        bucket_name = GCS_BUCKET_NAME
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
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    storage_client = storage.Client(project=PROJECT_ID)
    output_local_base = PATHS["output_local_base_template"].format(version=args.version)

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
