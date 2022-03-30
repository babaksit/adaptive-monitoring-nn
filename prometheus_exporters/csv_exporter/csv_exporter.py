import argparse
import logging
from typing import Dict

import time
import pandas as pd
from flask import Flask, request, Response
from threading import Thread

app = Flask(__name__)

args = None
node_exporter_metrics_url = None
keep_metrics = set()
drop_metrics = set()
csv_exporter = None
csv_exporter_thread = None
thread_running = False

@app.route('/start_csv_exporter')
def start_csv_exporter():
    global csv_exporter_thread
    global thread_running
    if thread_running:
        csv_exporter.restart = True
        return "Restarted csv_exporter"

    csv_exporter_thread = Thread(target=csv_exporter.run)
    csv_exporter_thread.start()
    thread_running = True
    return "started csv_exporter"


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
    return Response(csv_exporter.get_latest(), mimetype="text/plain")


class CSVExporter:
    gauges_col_val: Dict[str, Dict[str, float]]

    def __init__(self, csv_path, time_interval_seconds):
        self.gauges_col_val = {}
        self.col_gauges = {}
        self.time_interval_seconds = time_interval_seconds
        self.df = pd.read_csv(csv_path, index_col=0)
        self.init_gauges()
        self.stop_run = False
        self.restart = False

    def init_gauges(self):
        for column in self.df.columns:
            gauge_name = self.get_gauge_name(column)
            if gauge_name in self.gauges_col_val:
                self.gauges_col_val[gauge_name][column] = 0.0
            else:
                self.gauges_col_val[gauge_name] = {column: 0.0}
            self.col_gauges[column] = gauge_name

    @staticmethod
    def get_gauge_name(metric):
        return metric.split('{')[0]

    def build_metric(self, gauge, col_val: Dict[str, float]):
        metric_str = "# HELP " + gauge + " " + gauge + "\n" + \
                     "# TYPE " + gauge + " " + "gauge" + "\n"

        for col, val in col_val.items():
            metric_str = metric_str + col + " " + str(val) + "\n"
        return metric_str

    def get_latest(self):
        latest = []
        for gauge, col_val in self.gauges_col_val.items():
            if gauge in drop_metrics:
                continue
            if keep_metrics and gauge not in keep_metrics:
                continue
            latest.append(self.build_metric(gauge, col_val))

        if latest:
            return "".join(latest)
        return ""

    def run(self):
        for _, row in self.df.iterrows():
            now = time.time()
            for col in self.df.columns:
                gauge_name = self.col_gauges[col]
                self.gauges_col_val[gauge_name][col] = float(row[col])
            time_diff = self.time_interval_seconds - (time.time() - now)
            if time_diff > 0.00:
                time.sleep(time_diff)
            if self.restart:
                self.restart = False
            if self.stop_run:
                break


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)-4s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    parser = argparse.ArgumentParser(description="CSV File Exporter")
    parser.add_argument('--csv-path', type=str,
                        help='Path to csv file')
    parser.add_argument('--time-interval', type=int,
                        help='Time interval in seconds', default=1)
    parser.add_argument('--host', type=str,
                        help='host', default="0.0.0.0")
    parser.add_argument('--port', type=int,
                        help='port', default=9393)

    args = parser.parse_args()
    logging.info(args)
    csv_exporter = CSVExporter(csv_path=args.csv_path,
                               time_interval_seconds=args.time_interval)
    app.run(host=args.host, port=args.port)
    csv_exporter.stop_run = True
    csv_exporter_thread.join()
