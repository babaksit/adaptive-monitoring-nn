import logging

import requests
from typing import List


class ExporterApi:
    def __init__(self, col_query_dict, clear_keep_list_url,
                 keep_metric_url, drop_metric_url, start_csv_exporter_url):

        self.start_csv_exporter_url = start_csv_exporter_url
        self.drop_metric_url = drop_metric_url
        self.keep_metric_url = keep_metric_url
        self.clear_keep_list_url = clear_keep_list_url
        self.col_query_dict = col_query_dict

    def __keep_all_metrics(self):
        #  we should fetch all metrics
        req = requests.get(self.clear_keep_list_url)
        if req.status_code != 200:
            logging.error("couldn't clear_keep_list")
            return False
        return True

    def start_csv_exporter(self):
        req = requests.get(self.start_csv_exporter_url)
        logging.debug("starting csv_exporter")
        if req.status_code != 200:
            logging.error("couldn't start csv_exporter")
            return False
        logging.debug("started csv_exporter")
        return True

    def drop_metrics(self, metrics):

        success = True
        for metric in metrics:
            req = requests.get(self.drop_metric_url + metric)
            if req.status_code != 200:
                logging.error("couldn't keep metric " + metric)
                success = False
        return success

    def keep_metrics(self, cols: List[str] = None,
                     keep_all: bool = False):
        if keep_all:
            return self.__keep_all_metrics()
        success = True
        for col in cols:
            for metric in self.col_query_dict[col]["metrics"]:
                req = requests.get(self.keep_metric_url + metric)
                if req.status_code != 200:
                    logging.error("couldn't keep metric " + metric)
                    success = False
        return success
