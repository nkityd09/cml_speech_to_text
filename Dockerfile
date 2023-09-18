# Dockerfile
# Specify an ML Runtime base image
FROM docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-jupyterlab-python3.9-cuda:2023.05.2-b7
# Install telnet in the new image
RUN apt-get update && apt-get install -y --no-install-recommends telnet && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y ffmpeg
# Upgrade packages in the base image
RUN apt-get update && apt-get upgrade -y && apt-get clean && rm -rf /var/lib/apt/lists/*
# Install the python package sklearn
RUN pip install --no-cache-dir torch transformers youtube-dl gradio sentence-transformers accelerate bitsandbytes
# Override Runtime label and environment variables metadata
ENV ML_RUNTIME_EDITION="S2T Edition" \
       	ML_RUNTIME_SHORT_VERSION="1.0" \
        ML_RUNTIME_MAINTENANCE_VERSION=1 \
        ML_RUNTIME_DESCRIPTION="CML Runtime for Speech to Text Summarization using Generative AI"
ENV ML_RUNTIME_FULL_VERSION="${ML_RUNTIME_SHORT_VERSION}.${ML_RUNTIME_MAINTENANCE_VERSION}"
LABEL com.cloudera.ml.runtime.edition=$ML_RUNTIME_EDITION \
        com.cloudera.ml.runtime.full-version=$ML_RUNTIME_FULL_VERSION \
        com.cloudera.ml.runtime.short-version=$ML_RUNTIME_SHORT_VERSION \
        com.cloudera.ml.runtime.maintenance-version=$ML_RUNTIME_MAINTENANCE_VERSION \
        com.cloudera.ml.runtime.description=$ML_RUNTIME_DESCRIPTION
