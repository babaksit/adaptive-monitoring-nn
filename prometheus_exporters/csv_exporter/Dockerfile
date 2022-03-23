FROM python:3.8-bullseye
WORKDIR /home
RUN apt-get update

COPY requirements.txt .
COPY run_csv_exporter.sh .
COPY csv_exporter.py .
RUN chmod +x run_csv_exporter.sh
RUN pip3 install -r requirements.txt
ENTRYPOINT ["bash", "run_csv_exporter.sh"]
