FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && \
    apt-get install -y git curl && \
    apt-get clean

WORKDIR /app

COPY environment.yaml .

RUN apt-get update && apt-get install -y wget bzip2 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    /opt/conda/bin/conda init bash

ENV PATH="/opt/conda/bin:$PATH"

RUN conda update -n base -c defaults conda && \
    conda env create -f environment.yaml && \
    conda clean -a

SHELL ["conda", "run", "-n", "in-context-learning", "/bin/bash", "-c"]

COPY . .

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "in-context-learning", "python", "train_GLM.py"]
