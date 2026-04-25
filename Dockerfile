FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /workspace/project

ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Seoul

RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

CMD ["/bin/bash"]
