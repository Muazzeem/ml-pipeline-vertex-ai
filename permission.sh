# Set the project name
PROJECT=bitstrapped-gpu-testing

# Define the service account name
ACCOUNT="pipeline-service-account"

# Generate the email address for the service account using the project and account name
EMAIL="${ACCOUNT}@${PROJECT}.iam.gserviceaccount.com"

# Enable required Google Cloud services for the project
for SERVICE in "compute" "aiplatform" "cloudbuild"
do
  gcloud services enable ${SERVICE}.googleapis.com \
  --project=${PROJECT}
done

# Create a new IAM service account for the project
gcloud iam service-accounts create ${ACCOUNT} \
--project=${PROJECT}

# Add the service account to the project's IAM policy with 'roles/editor' role
gcloud projects add-iam-policy-binding ${PROJECT} \
--member=serviceAccount:${EMAIL} \
--role=roles/editor --condition=None
