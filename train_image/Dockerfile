FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-12.py310 AS base

RUN apt-get update && \
    apt-get install -y curl gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && \
    apt-get update -y && \
    apt-get install google-cloud-sdk -y

WORKDIR /subtlemedical

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY train .
ADD train/train.py .
ENTRYPOINT ["python", "train.py"]

