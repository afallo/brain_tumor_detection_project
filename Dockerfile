FROM python:3.10.6-slim

WORKDIR /prod

# We strip the requirements from useless packages like `ipykernel`, `matplotlib` etc...
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY brain brain

CMD uvicorn brain.api.fast:app --host 0.0.0.0 --port $PORT