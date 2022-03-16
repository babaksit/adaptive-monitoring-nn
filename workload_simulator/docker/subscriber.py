import argparse
import logging
import mmap
import os
import sys
import threading
import time
from pathlib import Path

import pika


class Subscriber:

    def __init__(self, cpu_scale, memory_scale, file_path):
        self.last_saved_time = -1
        self.stop_thread = False
        self.new_data = False
        # 1 KiB data
        self.one_kb = "A" * 1024
        self.data = 0
        self.temp_mem = []
        self.cpu_scale = cpu_scale
        self.memory_scale = memory_scale
        #creating unique file name for each pod
        self.file_name = file_path + str(time.time_ns()) + ".txt"
        fle = Path(self.file_name)
        fle.touch(exist_ok=True)
        self.tmp_mmap = None
        self.init_mmap()

    def init_mmap(self):
        self.tmp_mmap = mmap.mmap(-1, 1024 * 10)
        s = 'b' * 1024 * 10
        self.tmp_mmap.write(s.encode())
        f = os.open(self.file_name, os.O_CREAT | os.O_DIRECT | os.O_TRUNC | os.O_RDWR)
        os.write(f, self.tmp_mmap)
        os.close(f)

    def log_alive(self):
        now = time.time()
        if self.last_saved_time == -1:
            self.last_saved_time = now - 60
        diff = now - self.last_saved_time
        if diff >= 60:
            self.last_saved_time = now
            logging.info("Received %r" % self.data)

    def callback(self, ch, method, properties, body):
        self.new_data = True
        if not body.isdigit():
            # logging.warning("Received non digit message : " + str(body))
            return
        try:
            self.data = int(body)
        except ValueError as e:
            logging.error(e)

    def dummy_func(self):
        now = time.time()
        try:
            self.temp_mem = self.one_kb * self.data * self.memory_scale
            for _ in range(self.data * self.cpu_scale):
                f = os.open(self.file_name, os.O_DIRECT | os.O_RDWR)
                f = os.fdopen(f, 'rb+')
                f.readinto(self.tmp_mmap)
                f.close()
                f = os.open(self.file_name, os.O_CREAT | os.O_DIRECT | os.O_TRUNC | os.O_RDWR)
                os.write(f, self.tmp_mmap)
                os.close(f)
            sleep_time = now + 1.0 - time.time()
        except Exception as e:
            logging.error(str(e))

        if sleep_time > 0.000000:
            time.sleep(sleep_time)
        else:
            err_msg = "sleep time less than zero : " + str(sleep_time) \
                      + " for data: " + str(self.data)
            logging.warning(err_msg)

    def workload_simulator(self):
        while not self.stop_thread:
            self.log_alive()
            if self.new_data:
                # self.new_data = False
                self.dummy_func()


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-4s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    parser = argparse.ArgumentParser(description="Standalone Subscriber")
    parser.add_argument('--rabbitmq-host', type=str,
                        help='rabbitmq-host', default="127.0.0.1")
    parser.add_argument('--rabbitmq-user', type=str,
                        help='rabbitmq User', default="test")
    parser.add_argument('--rabbitmq-pass', type=str,
                        help='rabbitmq Pass', default="test")
    parser.add_argument('--rabbitmq-port', type=int,
                        help='rabbitmq port', default=5672)
    parser.add_argument('--rabbitmq-vhost', type=str,
                        help='rabbitmq host', default="/")
    parser.add_argument('--cpu-scale', type=int,
                        help='Workload Scale', default=10)
    parser.add_argument('--memory-scale', type=int,
                        help='Workload Scale', default=100000)
    parser.add_argument('--file-path', type=str,
                        help='Where to read and write file', default=".")

    args = parser.parse_args()

    logging.info(args)

    credentials = pika.PlainCredentials(args.rabbitmq_user, args.rabbitmq_pass)
    parameters = pika.ConnectionParameters(args.rabbitmq_host,
                                           args.rabbitmq_port,
                                           args.rabbitmq_vhost,
                                           credentials)

    try:
        connection = pika.BlockingConnection(parameters)
    except Exception as e:
        logging.error("Exception occurred when trying "
                      "to connect to the RabbitMQ server: " + str(e))
        return False

    channel = connection.channel()

    channel.exchange_declare(exchange='logs', exchange_type='fanout')

    result = channel.queue_declare(queue='', exclusive=True)
    queue_name = result.method.queue

    channel.queue_bind(exchange='logs', queue=queue_name)

    print(' [*] Waiting for logs. To exit press CTRL+C')

    sub = Subscriber(args.cpu_scale, args.memory_scale, args.file_path)

    thread = threading.Thread(target=sub.workload_simulator)
    thread.start()

    channel.basic_consume(
        queue=queue_name, on_message_callback=sub.callback, auto_ack=True)

    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        sub.stop_thread = True
        channel.stop_consuming()
        connection.close()
        thread.join()
        sys.exit(0)


if __name__ == '__main__':
    main()
