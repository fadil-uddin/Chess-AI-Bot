# -----------------------------------------------------------
# Dockerfile
#
# Container for running the Chess AI Web Application.
# It sets up a Python environment with Flask, PyTorch,
# and other dependencies needed for serving the model.
#
# Author: Fadil Uddin

# -------- Frontend build --------
FROM node:18 AS frontend
WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# -------- Backend --------
FROM python:3.10-slim AS backend
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY app.py .
COPY model.py .
COPY savedModels ./savedModels

# Copy built frontend into Flask static folder
COPY --from=frontend /app/frontend/build ./static

EXPOSE 5000
CMD ["python", "app.py"]
