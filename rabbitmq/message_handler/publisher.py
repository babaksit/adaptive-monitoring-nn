import logging
from rabbitmq.message_handler.connection_handler import ConnectionHandler


class Publisher:
    """
    Producer class to publish messages

    """

    def __init__(self, connection_handler: ConnectionHandler, channel,
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

    def publish_str_msg(self, msg: str) -> bool:
        """
        Publish
        Parameters
        ----------
        msg : The string message to publish

        Returns
        -------
        True if publishing the msg was successful else False
        """

        # if not self.channel.is_open():
        #     logging.error("channel is not open to publish the message")
        #     return False

        self.channel.basic_publish(exchange=self.exchange,
                                   routing_key=self.routing_key, body=msg)
        # logging.info("Sent: " + msg)
        return True
