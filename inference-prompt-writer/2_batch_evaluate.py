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

# python 2_batch_evaluate.py --version v00
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


def prepare_eval_batch_input(
    version,
    input_predictions_file,
    eval_prompt_template_file,
    input_jsonl_name,
):
  """Reads the output from the creation batch job and creates a new .jsonl file

  for the evaluation batch job.
  """
  logging.info(
      f"Preparing evaluation batch input from {input_predictions_file}"
  )

  if not os.path.exists(input_predictions_file):
    raise FileNotFoundError(
        f"Source predictions file not found: {input_predictions_file}"
    )

  eval_template = read_file(eval_prompt_template_file)

  count = 0
  with open(input_predictions_file, "r", encoding="utf-8") as infile, open(
      input_jsonl_name, "w", encoding="utf-8"
  ) as outfile:

    for line in infile:
      if not line.strip():
        continue

      item = json.loads(line)
      key = item.get("key")

      try:
        text_input = item["request"]["contents"][1]["parts"][0]["text"]
      except (KeyError, IndexError):
        logging.warning(
            f"Could not extract text input for key {key}. Skipping."
        )
        continue

      image_b64 = None
      response = item.get("response", {})
      candidates = response.get("candidates", [])
      for candidate in candidates:
        parts = candidate.get("content", {}).get("parts", [])
        for part in parts:
          if "inlineData" in part:
            image_b64 = part["inlineData"].get("data")
            break
        if image_b64:
          break

      if not image_b64:
        logging.warning(f"Could not find image data for key {key}. Skipping.")
        continue

      eval_prompt = eval_template.replace("&USER_INPUT&", text_input)

      eval_key = f"eval-{key}"
      request_obj = {
          "key": eval_key,
          "request": {
              "contents": [{
                  "role": "user",
                  "parts": [
                      {"text": eval_prompt},
                      {
                          "inlineData": {
                              "mimeType": "image/png",
                              "data": image_b64,
                          }
                      },
                  ],
              }],
              "generationConfig": {"response_mime_type": "application/json"},
          },
      }
      outfile.write(json.dumps(request_obj) + "\n")
      count += 1

  logging.info(f"Created {count} evaluation requests in {input_jsonl_name}")


def upload_blob(bucket, local_path, destination_blob_name):
  """Uploads a file to GCS."""
  blob = bucket.blob(destination_blob_name)
  blob.upload_from_filename(local_path)
  logging.info(
      f"Uploaded {local_path} to"
      f" gs://{config.GCS_BUCKET_NAME}/{destination_blob_name}"
  )


def run_batch_evaluation(version):
  """Orchestrates the batch evaluation job submission."""
  # Initialization
  client = genai.Client(
      vertexai=True, project=config.PROJECT_ID, location=config.LOCATION
  )
  storage_client = storage.Client(project=config.PROJECT_ID)
  bucket = storage_client.bucket(config.GCS_BUCKET_NAME)

  # Configs
  input_predictions_file = (
      f"generated/create/{version}/predictions.jsonl"
  )
  eval_prompt_template_file = config.EVAL_PROMPT
  timestamp = int(time.time())
  input_jsonl_name = (
      f"inputs/tmp_batch_input_eval_{version}_{timestamp}.jsonl"
  )
  gcs_input_uri = f"gs://{config.GCS_BUCKET_NAME}/ATL/{input_jsonl_name}"
  gcs_output_dir = f"gs://{config.GCS_BUCKET_NAME}/ATL/outputs/eval_{version}_{timestamp}/"
  log_file = "logs/batch_jobs.log"

  # 1. Prepare local JSONL
  prepare_eval_batch_input(
      version,
      input_predictions_file,
      eval_prompt_template_file,
      input_jsonl_name,
  )

  # 2. Upload to GCS
  upload_blob(bucket, input_jsonl_name, f"ATL/{input_jsonl_name}")

  # 3. Submit Batch Job
  logging.info(
      f"Submitting evaluation batch job for model {config.EVALUATION_MODEL}..."
  )
  batch_job = client.batches.create(
      model=config.EVALUATION_MODEL,
      src=gcs_input_uri,
      config=types.CreateBatchJobConfig(
          dest=gcs_output_dir, display_name=f"Eval_{version}_{timestamp}"
      ),
  )


  job_id = batch_job.name
  logging.info(f"Evaluation batch job submitted. ID: {job_id}")

  # 4. Record to logs
  log_entry = {
      "type": "evaluation",
      "input_uri": gcs_input_uri,
      "output_uri": gcs_output_dir,
      "eval_prompt_template": eval_prompt_template_file,
      "source_predictions": input_predictions_file,
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
  args = parser.parse_args()

  setup_logging()
  client = genai.Client(
      vertexai=True, project=config.PROJECT_ID, location=config.LOCATION
  )
  storage_client = storage.Client(project=config.PROJECT_ID)
  output_local_base = f"generated/evaluate/{args.version}"

  try:
    job, gcs_output_dir = run_batch_evaluation(args.version)
    if wait_for_job(client, job):
      download_results(storage_client, gcs_output_dir, output_local_base)
  except Exception as e:
    logging.error(f"An error occurred: {e}", exc_info=True)
