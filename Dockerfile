# Base image
FROM jupyter/base-notebook:latest

# ---------------------------
# Install system packages as root
# ---------------------------
USER root
RUN apt-get update && \
    apt-get install -y sudo git python3-venv python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Give jovyan passwordless sudo (optional)
RUN echo "jovyan ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# ---------------------------
# Set working directory
# ---------------------------
WORKDIR /home/jovyan/work

# Copy the contents of the repo (Dockerfile should be in repo root)
COPY . .

# ---------------------------
# Create Python virtual environment as root
# ---------------------------
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install -r requirements.txt && \
    /opt/venv/bin/pip install ipykernel && \
    /opt/venv/bin/python -m ipykernel install --user --name=venv --display-name "Python (venv)" && \
    # Remove old default kernel metadata to force venv
    jupyter kernelspec remove python3 -f && \
    # Install JupyterLab extension for ipywidgets
    /opt/venv/bin/jupyter labextension install @jupyter-widgets/jupyterlab-manager

# Fix permissions so jovyan can access everything
RUN chown -R jovyan:users /home/jovyan

# ---------------------------
# Switch back to jovyan user
# ---------------------------
USER jovyan
ENV PATH="/home/jovyan/work/venv/bin:$PATH"

# ---------------------------
# Start Jupyter Notebook
# ---------------------------
CMD ["start-notebook.sh", "--NotebookApp.token=''"]

