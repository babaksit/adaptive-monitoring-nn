import argparse
import json
import logging
import os
import sys
import threading
import time
from typing import List
from rabbitmq.message_handler.connection_handler import ConnectionHandler
from rabbitmq.message_handler.producer import Producer
from rabbitmq.message_handler.consumer import Consumer
from dataset.dataset_loader import DatasetLoader
import pandas as pd

from rabbitmq.message_handler.publisher import Publisher
from rabbitmq.message_handler.subscriber import Subscriber


class Scenario:
    """
    Scenario class that runs a scenario for given configs

    """

    producers: List[Producer]
    publishers: List[Publisher]
    subscribers: List[Subscriber]
    consumers: List[Consumer]
    threads: List[threading.Thread]
    configs_loaded: bool

    def __init__(self, scenario_config_path: str, connection_config_path: str):
        """
        Initialize

        Parameters
        ----------
        scenario_config_path : path to scenario config file
        connection_config_path : path to connection config file
        """
        self.scenario_config_path = scenario_config_path
        self.connection_config_path = connection_config_path
        self.scenario_config = None
        self.producers = []
        self.publishers = []
        self.subscribers = []
        self.consumers = []
        self.threads = []
        self.configs_loaded = False

    def load_config(self) -> bool:
        """
        Load config files and create corresponding variables

        Returns
        -------

        """
        with open(self.scenario_config_path) as f:
            self.scenario_config = json.load(f)
        if self.scenario_config["type"] == "pub_sub":
            if not self.__create_publishers() \
                    or not self.__create_subscribers():
                return False

        elif not self.__create_producers() \
                or not self.__create_consumers():
            return False

        self.configs_loaded = True

        return True

    def __create_publishers(self) -> bool:
        """
        Create publishers

        Returns
        -------

        """
        logging.info("Creating publishers")

        # TODO test for multiple publishers
        for i in range(self.scenario_config['num_publishers']):
            ch = ConnectionHandler()
            success = ch.connect(self.connection_config_path)
            if not success:
                return False
            channel = ch.create_pub_sub_channel()
            routing_key = ''
            p = Publisher(ch, channel, self.scenario_config['exchange'], routing_key)
            self.publishers.append(p)
        logging.info("Created publishers")
        return True

    def __create_subscribers(self) -> bool:
        """
        Create subscribers

        Returns
        -------

        """
        logging.info("Creating subscribers")
        for i in range(self.scenario_config['num_subscribers']):
            ch = ConnectionHandler()
            success = ch.connect(self.connection_config_path)
            if not success:
                return False
            channel = ch.create_channel()
            queue_name = self.scenario_config['queue_name'] + str(i)
            ch.declare_sub_queue(channel, queue_name)
            s = Subscriber(ch, channel, "logs", queue_name)
            self.subscribers.append(s)
        logging.info("Created subscribers")
        return True

    def __create_producers(self) -> bool:
        """
        Create producers

        Returns
        -------

        """

        # TODO test for multiple producers
        for i in range(self.scenario_config['num_producers']):
            ch = ConnectionHandler()
            success = ch.connect(self.connection_config_path)
            if not success:
                return False
            channel = ch.create_channel()
            routing_key = self.scenario_config['queue_name']
            ch.declare_queue(channel, routing_key)
            p = Producer(ch, channel, self.scenario_config['exchange'], routing_key)
            self.producers.append(p)

        logging.DEBUG("Created producers")
        return True

    def __create_consumers(self):
        """
        Create consumers

        Returns
        -------

        """

        # TODO test for multiple producers
        for i in range(self.scenario_config['num_consumers']):
            ch = ConnectionHandler()
            success = ch.connect(self.connection_config_path)
            if not success:
                return False
            channel = ch.create_channel()
            routing_key = self.scenario_config['queue_name']
            ch.declare_queue(channel, routing_key)
            c = Consumer(ch, channel, self.scenario_config['exchange'], routing_key)
            self.consumers.append(c)
        return True

    def run_consumers(self):
        """
        Running consumers in threads

        Returns
        -------

        """
        for consumer in self.consumers:
            thread = threading.Thread(target=consumer.start)
            thread.start()
            self.threads.append(thread)

    def run_subscribers(self):
        """
        Running subscribers in threads

        Returns
        -------

        """
        for subscriber in self.subscribers:
            thread = threading.Thread(target=subscriber.start)
            thread.start()
            self.threads.append(thread)

    def run_publishers(self):
        """
        Run publishers and simulates the time series dataset by waiting between sending each
        message based on duration time of dataframe loaded from load_timeseries function

        Returns
        -------

        """

        # if self.scenario_config['type'] == "multiple_message":
        df = DatasetLoader.load_multiple_msg_df(self.scenario_config['dataset_path'],
                                                self.scenario_config['time_column_name'])
        cl = self.scenario_config['value_column_name']

        for publisher in self.publishers:
            for index, row in df.iterrows():
                next_call = time.time()
                if not pd.isna(index):
                    for i in range(row[cl]):
                        publisher.publish_str_msg(str(row[cl]))
                    publisher.publish_str_msg("$")
                    sleep_time = next_call + 1.0 - time.time()
                    if sleep_time > 0.000000:
                        time.sleep(sleep_time)
                    else:
                        err_msg = "sleep time less than zero : " + str(sleep_time)
                        logging.warning(err_msg)
                        # logging.warning("sleep time less than zero : " + str(sleep_time))
        # else:
        #     df = DatasetLoader.load_timeseries(self.scenario_config['dataset_path'],
        #                                        self.scenario_config['time_column_name'],
        #                                        True)
        #     cl = self.scenario_config['value_column_name']
        #     for publisher in self.publishers:
        #         for index, row in df.iterrows():
        #             if not pd.isna(index):
        #                 publisher.publish_str_msg(str(row[cl]))
        #                 time.sleep(float(index))

    def run_producers(self):
        """
        Run producers and simulates the time series dataset by waiting between sending each
        message based on duration time of dataframe loaded from load_timeseries function

        Returns
        -------

        """

        if self.scenario_config['type'] == "multiple_message":
            df = DatasetLoader.load_multiple_msg_df(self.scenario_config['dataset_path'],
                                                    self.scenario_config['time_column_name'])
            cl = self.scenario_config['value_column_name']

            for producer in self.producers:
                for index, row in df.iterrows():
                    next_call = time.time()
                    if not pd.isna(index):
                        for i in range(row[cl]):
                            producer.publish_str_msg(str(row[cl]))
                        producer.publish_str_msg("-1")
                        sleep_time = next_call + 1.0 - time.time()
                        if sleep_time > 0.000000:
                            time.sleep(sleep_time)
                        else:
                            err_msg = "sleep time less than zero : " + str(sleep_time)
                            logging.warning(err_msg)
                            # logging.warning("sleep time less than zero : " + str(sleep_time))
        else:
            df = DatasetLoader.load_timeseries(self.scenario_config['dataset_path'],
                                               self.scenario_config['time_column_name'],
                                               True)
            cl = self.scenario_config['value_column_name']
            for producer in self.producers:
                for index, row in df.iterrows():
                    if not pd.isna(index):
                        producer.publish_str_msg(str(row[cl]))
                        time.sleep(float(index))

    # TODO remove this function
    def run(self) -> bool:
        """
        Run the scenario, it creates a thread for each consumer

        Returns
        -------
        True if running the scenario was successful else False
        """
        if not self.configs_loaded:
            if not self.__load_config():
                logging.error("Could not load the configs")
                return False
        self.run_consumers()
        self.run_producers()
        self.close()
        return True

    def close(self):
        """
        Close connections and threads

        Returns
        -------

        """

        #     consumer.connection_handler.close_connection()
        for producer in self.producers:
            producer.connection_handler.close_connection()
        # TODO close consumers
        # for consumer in self.consumers:
        #     consumer.stop()
        logging.info("closing threads")
        for thread in self.threads:
            thread.join()
        logging.info("scenario closed")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)-4s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    parser = argparse.ArgumentParser(description="Train a time series network")
    parser.add_argument('--scenario-config', type=str,
                        help='Path to the scenario config file', default="configs/scenario.json")
    parser.add_argument('--connection-config', type=str,
                        help='Path to the connection config file', default="configs/connection.json")
    args = parser.parse_args()

    sc = Scenario(args.scenario_config, args.connection_config)
    try:
        sc.run()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
