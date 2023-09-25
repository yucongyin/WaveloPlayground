FROM python:3.11

WORKDIR /app

COPY requirements.txt /app/

RUN apt-get update && \
    apt-get install -y python3-pil tesseract-ocr libtesseract-dev tesseract-ocr-eng tesseract-ocr-script-latn && \
    pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 18501

ENTRYPOINT ["streamlit", "run", "waveloGPT.py", "--server.port=18501", "--server.address=0.0.0.0"]