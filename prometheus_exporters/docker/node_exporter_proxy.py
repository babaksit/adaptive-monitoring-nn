import argparse
import logging

import requests

from flask import Flask, request

app = Flask(__name__)

args = None
node_exporter_metrics_url = None
keep_metrics = set()

def get_html(text):
    return '<pre style="word-wrap: break-word; white-space: pre-wrap;">' + text + "</pre>"

def is_in_keep_metrics(line):
    for keep_metric in keep_metrics:
        if keep_metric in line:
            return True
    return False

@app.route('/keep_metric')
def keep_metric():
    metric = request.args.get('metric')
    if metric:
        keep_metrics.add(metric)
        return "Added " + str(metric) + " to keep metrics list"
    return "Metric was None", 400

@app.route('/clear_keep_list')
def clear_keep_list():
    keep_metrics.clear()
    return "Cleared Keep list"

@app.route('/metrics')
def metrics():
    metrics = requests.get(node_exporter_metrics_url).text
    if not keep_metrics:
        return metrics
        # return get_html(metrics)

    lines = metrics.splitlines()
    res = []

    for line in lines:
        if is_in_keep_metrics(line):
            res.append(line)

    res = "\n".join(res)
    return res
    # return get_html(res)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-4s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    parser = argparse.ArgumentParser(description="Node exporter proxy to filter metrics")
    parser.add_argument('--node-exporter-host', type=str,
                        help='node-exporter host', default="127.0.0.1")
    parser.add_argument('--node-exporter-port', type=int,
                        help='node-exporter port', default=9100)
    parser.add_argument('--host', type=str,
                        help='host', default="127.0.0.1")
    parser.add_argument('--port', type=int,
                        help='port', default=9393)

    args = parser.parse_args()
    logging.info(args)
    node_exporter_metrics_url = "http://{}:{}/metrics".format(args.node_exporter_host, args.node_exporter_port)

    app.run(host=args.host, port=args.port)
