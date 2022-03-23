import argparse
import logging

import requests

from flask import Flask, request

app = Flask(__name__)

args = None
node_exporter_metrics_url = None
keep_metrics = set()
drop_metrics = set()


def get_html(text):
    return '<pre style="word-wrap: break-word; white-space: pre-wrap;">' + text + "</pre>"


def is_in(line, metric_list):
    for metric in metric_list:
        if metric in line:
            return True
    return False


def is_in_keep_metrics(line):
    return is_in(line, keep_metrics)


def is_in_drop_metrics(line):
    return is_in(line, drop_metrics)


@app.route('/drop_metric')
def drop_metric():
    metric = request.args.get('metric')
    if metric:
        drop_metrics.add(metric)
        return "Added " + str(metric) + " to drop metrics list"
    return "Metric was None", 400


@app.route('/keep_metric')
def keep_metric():
    metric = request.args.get('metric')
    if metric:
        keep_metrics.add(metric)
        return "Added " + str(metric) + " to keep metrics list"
    return "Metric was None", 400


@app.route('/clear_drop_list')
def clear_drop_list():
    drop_metrics.clear()
    return "Cleared drop list"


@app.route('/clear_keep_list')
def clear_keep_list():
    keep_metrics.clear()
    return "Cleared Keep list"


@app.route('/metrics')
def metrics():
    metric_list = requests.get(node_exporter_metrics_url).text
    if not keep_metrics and not drop_metrics:
        return metric_list

    if not keep_metrics:
        lines = metric_list.splitlines()
        res = []

        for line in lines:
            if not is_in_drop_metrics(line):
                res.append(line)

        res = "\n".join(res)
        return res

    lines = metric_list.splitlines()
    res = []

    for line in lines:
        if is_in_keep_metrics(line) and not is_in_drop_metrics(line):
            res.append(line)

    res = "\n".join(res)
    return res


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
