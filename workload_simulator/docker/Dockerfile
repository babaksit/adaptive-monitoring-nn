FROM python:3.8-bullseye
WORKDIR /home
RUN apt-get update
#    && apt-get install -y wondershaper \
#    && apt-get install bmon -y

COPY requirements.txt .
COPY run_subscriber.sh .
COPY subscriber.py .
RUN chmod +x run_subscriber.sh
RUN pip3 install -r requirements.txt
ENTRYPOINT ["bash", "run_subscriber.sh"]
