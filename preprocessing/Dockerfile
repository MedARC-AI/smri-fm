FROM freesurfer/freesurfer:7.4.1

# Install uv and use it to manage a clean Python env separate from FreeSurfer's
# stripped Python 3.8 (which is missing _sqlite3 and other stdlib modules).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV TEMPLATEFLOW_HOME=/opt/templateflow

RUN uv python install 3.12 && \
    uv venv --python 3.12 /opt/venv && \
    uv pip install --python /opt/venv nibabel antspyx numpy templateflow \
        "tensorflow[and-cuda]" "tf-keras" surfa voxelmorph neurite

# Expose the venv's bundled CUDA libs (esp. nvjitlink for PTX JIT compilation
# of Blackwell sm_120 kernels) so TF can find them at runtime.
ENV LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cufft/lib:/opt/venv/lib/python3.12/site-packages/nvidia/curand/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cusolver/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cusparse/lib:/opt/venv/lib/python3.12/site-packages/nvidia/nccl/lib:/opt/venv/lib/python3.12/site-packages/nvidia/nvjitlink/lib:/usr/lib64:${LD_LIBRARY_PATH}"

COPY preprocessing/ /app/preprocessing/

WORKDIR /app

ENTRYPOINT ["/opt/venv/bin/python", "/app/preprocessing/pipeline.py"]
