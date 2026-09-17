FROM python:3.12-slim

WORKDIR /app_mlops

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app_mlops

CMD ["uvicorn", "model_app.model_inference:app", "--host", "0.0.0.0", "--port", "8000"]