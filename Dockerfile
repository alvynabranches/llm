FROM python:3.11.4

WORKDIR /app
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools
RUN pip install --no-cache-dir --upgrade transformers torch einops accelerate xformers fastapi uvicorn

COPY ws.py ws.py

CMD [ "uvicorn", "ws:app", "--host", "0.0.0.0", "--port", "8000" ]
