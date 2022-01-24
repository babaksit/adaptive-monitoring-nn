from prometheus_client import start_http_server, Gauge
import time


current_requests = Gauge("app_requests_current", "Current requests")


if __name__ == '__main__':
    # Start up the server to expose the metrics.
    start_http_server(8085)

    # Generate some requests.
    while True:
        current_requests.inc()
        time.sleep(1)


