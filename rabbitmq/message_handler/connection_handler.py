import pika
import json
import logging


class ConnectionHandler:
    connection: pika.BlockingConnection

    def __init__(self):
        self.connection = None
        self.config = None

    def connect(self, config_file: str) -> bool:
        """
        Connect to the RabbitMQ server using the connection parameters
        in the given config file

        Parameters
        ----------
        config_file : path to the config file

        Returns
        -------
        True if the connection was successful else False
        """

        with open(config_file) as f:
            config = json.load(f)
        self.config = config
        credentials = pika.PlainCredentials(config['user'], config['pass'])
        parameters = pika.ConnectionParameters(config['host'],
                                               config['port'],
                                               config['virtual_host'],
                                               credentials)
        try:
            self.connection = pika.BlockingConnection(parameters)
        except Exception as e:
            logging.error("Exception occurred when trying "
                          "to connect to the RabbitMQ server: " + str(e))
            return False

        return True

    def create_channel(self):
        """
        Create a channel

        Returns
        -------
        BlockingChannel:
                    Created channel
        """
        if not self.connection:
            logging.error("connection not found,"
                          "In order to create a channel, "
                          "a connection should be established first")
            return None

        return self.connection.channel()

    def create_pub_sub_channel(self):
        if not self.connection:
            logging.error("connection not found,"
                          "In order to create a channel, "
                          "a connection should be established first")
            return None
        if not self.config:
            logging.error("config file is None")
            return None
        channel = self.connection.channel()
        channel.exchange_declare(exchange="logs",
                                 exchange_type="fanout")
        return channel

    @staticmethod
    def declare_queue(channel, name: str):
        """
        Declare a queue

        Parameters
        ----------
        channel: BlockingChannel
                channel that queue would be declared on
        name : name of the queue

        Returns
        -------
        BlockingChannel:
                        channel with declared queue
        """
        # logging.warning(channel)
        # if channel.is_closed():
        #     logging.error("channel is not open")
        #     return channel

        channel.queue_declare(queue=name)
        channel.queue_bind(exchange='logs', queue=name)

        return channel

    @staticmethod
    def declare_sub_queue(channel, name: str):
        """
        Declare a queue for subscriber

        Parameters
        ----------
        channel: BlockingChannel
                channel that queue would be declared on
        name : name of the queue

        Returns
        -------
        BlockingChannel:
                        channel with declared queue
        """
        # logging.warning(channel)
        # if channel.is_closed():
        #     logging.error("channel is not open")
        #     return channel

        channel.queue_declare(queue=name)
        channel.queue_bind(exchange='logs', queue=name)

        return channel

    def close_connection(self) -> bool:
        """
        Close the connection

        Returns
        -------
        True if it was successful else False
        """
        # if self.connection.is_open():
        self.connection.close()
        return True
        # return False
