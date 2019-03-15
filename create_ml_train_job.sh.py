readonly PROJECT=$(gcloud config list project --format "value(core.project)")
readonly JOB_ID="generative_${USER}_$(date +%Y%m%d_%H%M%S)"
readonly BUCKET="gs://memory-box-0"
readonly GCS_PATH="${BUCKET}/${USER}/${JOB_ID}"

echo
echo "Using job id: ${JOB_ID}"
set -e


gcloud ml-engine jobs submit training "${JOB_ID}" \
  --stream-logs \
  --runtime-version 1.5  \
  --module-name main_cloud_job \
  --package-path "./" \
  --staging-bucket "${BUCKET}" \
  --region us-east1 \
  --config "./config.yaml" \
  -- \
  --batch_size 64 \
  --data_dir "${BUCKET}" \
      