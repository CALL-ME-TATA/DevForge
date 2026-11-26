# Training image (Linux: ViZDoom IPC is ~3.5x faster than on Windows). Usage: see training/README.md
FROM python:3.13-slim
RUN pip install --no-cache-dir vizdoom==1.3.3 gymnasium==1.4.0 numpy==2.5.5 \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128
ENV OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
WORKDIR /w
