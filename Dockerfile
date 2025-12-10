FROM python:3.10.6-buster

#WORKDIR /prod

# We strip the requirements from useless packages like `ipykernel`, `matplotlib` etc...
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY brain brain

CMD ["sh", "-c", "uvicorn brain.api.fast:app --host 0.0.0.0 --port ${PORT:-8080}"]