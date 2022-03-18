FROM python:3.8-bullseye
WORKDIR /home
RUN apt-get update

COPY requirements.txt .
COPY run_node_exporter_proxy.sh .
COPY node_exporter_proxy.py .
RUN chmod +x run_node_exporter_proxy.sh
RUN pip3 install -r requirements.txt
ENTRYPOINT ["bash", "run_node_exporter_proxy.sh"]
