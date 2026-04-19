FROM freesurfer/freesurfer:7.4.1

# Install uv and use it to manage a clean Python env separate from FreeSurfer's
# stripped Python 3.8 (which is missing _sqlite3 and other stdlib modules).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
# Bake TemplateFlow templates into the image so the container is self-contained.
ENV TEMPLATEFLOW_HOME=/opt/templateflow

RUN uv python install 3.12 && \
    uv venv --python 3.12 /opt/venv && \
    uv pip install --python /opt/venv \
        nibabel antspyx numpy templateflow \
        "tensorflow[and-cuda]" "tf-keras" surfa voxelmorph neurite && \
    python -c "import templateflow; templateflow.api.get('MNI152NLin2009cAsym', resolution=1, desc='brain', suffix='T1w')"

COPY preprocessing/ /app/preprocessing/

WORKDIR /app

ENTRYPOINT ["/opt/venv/bin/python", "/app/preprocessing/pipeline.py"]
