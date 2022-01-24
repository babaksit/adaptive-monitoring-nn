import logging

from connection_handler import ConnectionHandler
from pika.adapters.blocking_connection import BlockingChannel


class Consumer:
    """
    Producer class to publish messages

    """

    def __init__(self, connection_handler: ConnectionHandler, channel: BlockingChannel,
                 exchange: str, routing_key: str):
        """
        Initialize

        Parameters
        ----------
        connection_handler : handle connection to RabbitMQ server
        channel :  A BlockingChannel which msgs will be published on
        exchange: Exchange name
        routing_key: Routing Key which could be the queue name declared in @declare_queue function

        """
        self.connection_handler = connection_handler
        self.channel = channel
        self.exchange = exchange
        self.routing_key = routing_key

    @staticmethod
    def callback(ch, method, properties, body):
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
        # pass
        logging.info(" [x] Received %r" % body)

    def start(self):
        """
        Start consuming

        Returns
        -------

        """
        self.channel.basic_consume(queue=self.routing_key, on_message_callback=self.callback,
                                   auto_ack=True)
        self.channel.start_consuming()

    def stop(self):
        """
        Stop consuming

        Returns
        -------

        """
        self.channel.stop_consuming()
