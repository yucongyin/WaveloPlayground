FROM python:3.10

WORKDIR /app

COPY requirements.txt /app/
RUN python -m pip install --upgrade pip
RUN apt-get update && apt-get -y install libzmq3-dev
RUN pip install cython
RUN pip download pyzmq
RUN pip install pyzmq

RUN apt-get update && \
    apt-get install -y python3-pil tesseract-ocr libtesseract-dev tesseract-ocr-eng tesseract-ocr-script-latn && \
    pip install --no-cache-dir -r requirements.txt 

COPY . /app

EXPOSE 18501

ENTRYPOINT ["streamlit", "run", "macGPT.py", "--server.port=18501", "--server.address=0.0.0.0"]