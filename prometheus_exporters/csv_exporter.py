import argparse
import logging
from typing import Dict, Any
from prometheus_client import REGISTRY, PROCESS_COLLECTOR, PLATFORM_COLLECTOR
from prometheus_client import start_http_server, Gauge
import time
import pandas as pd


class CSVExporter:
    gauges: Dict[str, Gauge]

    def __init__(self, csv_path, time_interval_seconds):
        REGISTRY.unregister(PROCESS_COLLECTOR)
        REGISTRY.unregister(PLATFORM_COLLECTOR)
        REGISTRY.unregister(REGISTRY._names_to_collectors['python_gc_objects_collected_total'])
        self.time_interval_seconds = time_interval_seconds
        self.df = pd.read_csv(csv_path)
        self.gauges = {}
        self.init_gauges()

    def init_gauges(self):
        for column in self.df.columns:
            self.gauges[column] = Gauge(column, column)

    def run(self):
        for _, row in self.df.iterrows():
            for col in self.df.columns:
                self.gauges[col].set(row[col])
            time.sleep(self.time_interval_seconds)
        while True:
            time.sleep(1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)-4s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    parser = argparse.ArgumentParser(description="CSV File Exporter")
    parser.add_argument('--csv-path', type=str,
                        help='Path to csv file', default="cpu_memory_rate.csv")
    parser.add_argument('--time-interval', type=int,
                        help='Time interval in seconds', default=1)
    parser.add_argument('--port', type=int,
                        help='exporter port', default=9494)
    args = parser.parse_args()

    # Start up the server to expose the metrics.
    start_http_server(args.port)
    csv_exporter = CSVExporter(csv_path=args.csv_path,
                               time_interval_seconds=args.time_interval)

    csv_exporter.run()
