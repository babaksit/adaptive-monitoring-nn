import datetime
import logging
import time

from rabbitmq.message_handler.connection_handler import ConnectionHandler
from pika.adapters.blocking_connection import BlockingChannel


class Subscriber:
    """
    Subscriber class to subscribe messages

    """

    def __init__(self, connection_handler: ConnectionHandler, channel: BlockingChannel,
                 exchange: str, queue_name: str):
        """
        Initialize

        Parameters
        ----------
        connection_handler : handle connection to RabbitMQ server
        channel :  A BlockingChannel which msgs will be published on
        exchange: Exchange name
        queue_name: Queue name which could be the queue name declared in @declare_queue function

        """
        self.connection_handler = connection_handler
        self.channel = channel
        self.exchange = exchange
        self.queue_name = queue_name
        self.last_saved_time = -1
        self.last_val = -1

    def callback(self, ch, method, properties, body):
        """
        Callback function to receive new messages
        Parameters
        ----------
        ch :
        method :
        properties :
        body :

        Returns
        -------

        """

        # logging.info("Received %r %s", body, self.queue_name)
        # logging.info(str(ch))

        now = time.time()
        if (now - self.last_saved_time) > 10:
            self.last_saved_time = now
            logging.info("Received %r" % body)

    def start(self):
        """
        Start subscribing

        Returns
        -------

        """
        self.channel.basic_consume(queue=self.queue_name, on_message_callback=self.callback,
                                   auto_ack=True)
        self.channel.start_consuming()

    def stop(self):
        """
        Stop consuming

        Returns
        -------

        """
        self.channel.stop_consuming()
