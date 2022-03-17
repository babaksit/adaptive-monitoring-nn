from prometheus_client.parser import text_string_to_metric_families

import requests

from flask import Flask, request

app = Flask(__name__)

dropped_metrics = set()


def get_html(text):
    return '<pre style="word-wrap: break-word; white-space: pre-wrap;">' + text + "</pre>"


def is_in_dropped_metrics(line):
    for dropped_metric in dropped_metrics:
        if dropped_metric in line:
            return True
    return False


@app.route('/drop_metric')
def drop_metric():
    metric = request.args.get('metric')
    if metric:
        dropped_metrics.add(metric)
        return "Added " + str(metric) + " to drop list"
    return "Metric was None", 400


@app.route('/add_metric')
def add_metric():
    metric = request.args.get('metric')
    if metric:
        dropped_metrics.discard(metric)
        return "Removed " + str(metric) + " from drop list"
    return "Metric was None", 400


@app.route('/metrics')
def metrics():
    metrics = requests.get("http://127.0.0.1:9100/metrics").text
    if not dropped_metrics:
        return get_html(metrics)

    lines = metrics.splitlines()
    res = []

    for line in lines:
        if is_in_dropped_metrics(line):
            continue
        res.append(line)

    res = "\n".join(res)
    return get_html(res)


if __name__ == '__main__':
    app.run()
