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
from pathlib import Path
import time
import config
from google import genai
from google.cloud import storage
from google.genai import types


def setup_logging():
  """Sets up logging to console only."""
  os.makedirs("logs", exist_ok=True)
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - %(levelname)s - %(message)s",
      handlers=[logging.StreamHandler()],
  )


def read_file(path):
  with open(path, "r", encoding="utf-8") as f:
    return f.read()


def read_json(path):
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)


def prepare_batch_input(
    version, system_prompt_file, user_inputs_file, input_jsonl_name
):
  """Reads local prompt and input files, and creates a .jsonl file for batch inference."""
  logging.info(
      f"Preparing batch input from {user_inputs_file} using"
      f" {system_prompt_file}"
  )

  system_instruction = read_file(system_prompt_file)
  user_inputs = read_json(user_inputs_file)

  with open(input_jsonl_name, "w", encoding="utf-8") as f:
    for img_name, content in user_inputs.items():
      request_obj = {
          "key": img_name,
          "request": {
              "contents": [
                  {
                      "role": "user",
                      "parts": [{"text": f"{system_instruction}"}],
                  },
                  {"role": "user", "parts": [{"text": f"{content}"}]},
              ],
              "generationConfig": {
                  "thinking_config": {"thinking_level": "minimal"},
                  "image_config": {"image_size": "1K"},
              },
          },
      }
      f.write(json.dumps(request_obj) + "\n")


def upload_blob(bucket, local_path, destination_blob_name):
  """Uploads a file to GCS."""
  blob = bucket.blob(destination_blob_name)
  blob.upload_from_filename(local_path)
  logging.info(
      f"Uploaded {local_path} to"
      f" gs://{config.GCS_BUCKET_NAME}/{destination_blob_name}"
  )


def run_batch_inference(version):
  """Orchestrates the batch inference job submission."""
  # Initialization
  client = genai.Client(
      vertexai=True, project=config.PROJECT_ID, location=config.LOCATION
  )
  storage_client = storage.Client(project=config.PROJECT_ID)
  bucket = storage_client.bucket(config.GCS_BUCKET_NAME)

  # Configs
  user_inputs_file = "inputs/extracted_inputs.json"
  system_prompt_file = f"prompts/inference_{version}.txt"
  if version == "v0":
    system_prompt_file = f"prompts/inference_v0.txt"
    
  timestamp = int(time.time())
  input_jsonl_name = (
      f"inputs/tmp_batch_input_{version}_{timestamp}.jsonl"
  )
  gcs_input_uri = f"gs://{config.GCS_BUCKET_NAME}/ATL/{input_jsonl_name}"
  gcs_output_dir = (
      f"gs://{config.GCS_BUCKET_NAME}/ATL/outputs/{version}_{timestamp}/"
  )

  log_file = "logs/batch_jobs.log"

  # 1. Prepare local JSONL
  prepare_batch_input(
      version, system_prompt_file, user_inputs_file, input_jsonl_name
  )

  # 2. Upload to GCS
  upload_blob(bucket, input_jsonl_name, f"ATL/{input_jsonl_name}")

  # 3. Submit Batch Job
  logging.info(f"Submitting batch job for model {config.INFERENCE_MODEL}...")
  batch_job = client.batches.create(
      model=config.INFERENCE_MODEL,
      src=gcs_input_uri,
      config=types.CreateBatchJobConfig(
          dest=gcs_output_dir,
          display_name=f"Batch_{version}_{timestamp}",
      ),
  )


  job_id = batch_job.name
  logging.info(f"Batch job submitted. ID: {job_id}")

  # 4. Record to logs
  log_entry = {
      "type": "inference",
      "input_uri": gcs_input_uri,
      "output_uri": gcs_output_dir,
      "system_prompt": system_prompt_file,
      "user_inputs": user_inputs_file,
      "job_id": job_id,
      "timestamp": datetime.now().isoformat(),
  }
  with open(log_file, "a", encoding="utf-8") as f:
    f.write(json.dumps(log_entry) + "\n")

  return batch_job, gcs_output_dir


def wait_for_job(client, batch_job):
  """Polls the batch job status until completion."""
  logging.info(f"Waiting for batch job {batch_job.name} to complete...")
  while True:
    job = client.batches.get(name=batch_job.name)
    state = job.state.name if hasattr(job.state, "name") else str(job.state)
    logging.info(f"Current state: {state}")

    if state == "JOB_STATE_SUCCEEDED":
      logging.info("Batch job completed successfully.")
      return True
    elif state in ["FAILED", "CANCELLED", "EXPIRED"]:
      logging.error(f"Batch job ended with state: {state}")
      return False

    time.sleep(60)


def download_results(storage_client, gcs_uri, local_base):
  """Downloads the batch prediction results from GCS."""
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
    if ".jsonl" in blob.name or "prediction" in blob.name:
      found = True
      filename = os.path.basename(blob.name)
      local_path = os.path.join(local_base, filename)
      blob.download_to_filename(local_path)
      logging.info(f"Downloaded {filename} to {local_path}")

  if not found:
    logging.warning("No result files found in the output GCS directory.")


def extract_images_from_local_jsonl(local_dir):
  """Parses .jsonl files in the local directory and extracts base64 images."""
  logging.info(f"Extracting images from .jsonl files in {local_dir}")
  count = 0
  for filename in os.listdir(local_dir):
    if filename.endswith(".jsonl"):
      jsonl_path = os.path.join(local_dir, filename)
      with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
          try:
            item = json.loads(line)
            key = item.get("key")
            if not key:
              continue

            response = item.get("response", {})
            candidates = response.get("candidates", [])

            image_saved = False
            for candidate in candidates:
              content = candidate.get("content", {})
              parts = content.get("parts", [])
              for part in parts:
                if "inlineData" in part:
                  b64_data = part["inlineData"].get("data")
                  if b64_data:
                    img_filename = key if "." in key else f"{key}.png"
                    img_path = os.path.join(local_dir, img_filename)
                    with open(img_path, "wb") as img_f:
                      img_f.write(base64.b64decode(b64_data))
                    count += 1
                    image_saved = True
                    break
              if image_saved:
                break
          except Exception as e:
            logging.error(f"Error parsing line {line_num} in {filename}: {e}")
  logging.info(f"Extraction complete. Total images saved: {count}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--version", type=str, required=True)
  args = parser.parse_args()

  setup_logging()
  client = genai.Client(
      vertexai=True, project=config.PROJECT_ID, location=config.LOCATION
  )
  storage_client = storage.Client(project=config.PROJECT_ID)
  output_local_base = f"generated/create/{args.version}"

  try:
    job, gcs_output_dir = run_batch_inference(args.version)
    if wait_for_job(client, job):
      download_results(storage_client, gcs_output_dir, output_local_base)
      extract_images_from_local_jsonl(output_local_base)
  except Exception as e:
    logging.error(f"An error occurred: {e}", exc_info=True)
