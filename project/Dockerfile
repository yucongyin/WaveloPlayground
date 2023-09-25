FROM python:3.11

WORKDIR /app

COPY requirements.txt /app/

RUN apt-get update && \
    pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 15001


ENTRYPOINT ["python","main.py"]