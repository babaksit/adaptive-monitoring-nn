# import datetime
# import time
# import requests
#
# PROMETHEUS = 'http://localhost:9090/'
#
# response =requests.get('http://localhost:9090/api/v1/query_range?query=rabbitmq_module_up&start=2021-12-31T18:00:00Z&end=2021-12-31T18:00:10Z&step=1s')
# ms = response.json()['data']['result']
#
# for m in ms:
#     print(m['values'])  #[[1640973600, '0'], [1640973601, '0']]
import time
from datetime import datetime
from datetime import timedelta

from prometheus_pandas import query
from prometheus_api_client import PrometheusConnect, MetricsList, Metric, MetricSnapshotDataFrame, MetricRangeDataFrame
from prometheus_api_client.utils import parse_datetime
prom = PrometheusConnect(url="http://127.0.0.1:9090/", disable_ssl=True)
all_metrics_name = prom.all_metrics()
filter_metrics_name = [s for s in all_metrics_name if s.startswith("rabbitmq")]

date_format_str = '%Y-%m-%dT%H:%M:%SZ'


for metric_name in filter_metrics_name:
    start_time_str = "2021-12-31T18:00:00Z"
    start_time = datetime.strptime(start_time_str, date_format_str)
    for i in range(16):
        end_time = start_time + timedelta(hours=3)
        start_time_str = start_time.strftime(date_format_str)
        end_time_str = end_time.strftime(date_format_str)
        p = query.Prometheus('http://localhost:9090')
        res = p.query_range(metric_name, start_time_str, end_time_str, '1s')

        save_dir = "/home/bsi/thesis/Adaptive_Monitoring_NN/data/prometheus/2021-12-31 18:00:00 to 2022-01-02 18:00:00/"
        save_path = save_dir + metric_name + ".csv"
        if i == 0:
            res.to_csv(save_path)
        else:
            res.to_csv(save_path,mode='a', header=False)
        start_time = end_time


