FROM nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04

WORKDIR /app

ARG SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.13 \
    python3.13-venv \
    python3.13-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.13 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml .
COPY src ./src

RUN pip install --upgrade pip
RUN SETUPTOOLS_SCM_PRETEND_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION} pip install .

CMD ["python", "-m", "portfolio_risk_engine"]



#nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04 : 3.98 GB compressé en linux/amd64
#nvidia/cuda:13.1.1-cudnn-runtime-ubuntu24.04 : 1.69 GB compressé en linux/amd64

#FROM nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04 AS builder
#
#WORKDIR /app
#
#RUN apt-get update && apt-get install -y \
#    python3 \
#    python3-venv \
#    python3-pip \
#    && rm -rf /var/lib/apt/lists/*
#
#RUN python3 -m venv /opt/venv
#ENV PATH="/opt/venv/bin:$PATH"
#
#COPY pyproject.toml .
#COPY src ./src
#
#RUN pip install --upgrade pip
#RUN pip install .
#
#
#FROM nvidia/cuda:13.1.1-cudnn-runtime-ubuntu24.04
#
#WORKDIR /app
#
#RUN apt-get update && apt-get install -y \
#    python3 \
#    && rm -rf /var/lib/apt/lists/*
#
#COPY --from=builder /opt/venv /opt/venv
#COPY --from=builder /app/src /app/src
#COPY --from=builder /app/pyproject.toml /app/pyproject.toml
#
#ENV PATH="/opt/venv/bin:$PATH"
#
#CMD ["python", "-m", "main"]