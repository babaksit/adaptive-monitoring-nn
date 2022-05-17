FROM python:3.8-bullseye
WORKDIR /home
RUN apt-get update

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY models models
COPY dataset dataset
COPY prometheus prometheus
COPY utils utils
COPY saved_models/lstm_300_600.pth.tar saved_models/lstm_300_600.pth.tar
COPY run_pipeline.sh .
COPY main.py .
RUN chmod +x run_pipeline.sh

ENTRYPOINT ["bash", "run_pipeline.sh"]
