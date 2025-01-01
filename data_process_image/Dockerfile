FROM ubuntu:22.04
# FROM python:3.10-slim
RUN apt-get update && \
    apt-get install -y curl gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && \
    apt-get update -y && \
    apt-get install google-cloud-sdk -y
WORKDIR /data_procesor


COPY requirements.txt .

RUN apt-get update  && apt install -y python3-pip

RUN pip3 install --no-cache-dir -r requirements.txt

COPY prepare_dataset_IXI.py .
ENTRYPOINT ["python3", "prepare_dataset_IXI.py"]

