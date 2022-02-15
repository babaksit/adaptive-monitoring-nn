import argparse
import logging
import os
import sys

from rabbitmq.message_handler.scenario import Scenario

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-4s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
parser = argparse.ArgumentParser(description="Train a time series network")
parser.add_argument('--scenario-config', type=str,
                    help='Path to the scenario config file', default="../configs/pub_sub_scenario.json")
parser.add_argument('--connection-config', type=str,
                    help='Path to the connection config file', default="../configs/connection.json")
args = parser.parse_args()

sc = Scenario(args.scenario_config, args.connection_config)


if not sc.load_config():
    logging.error("Could not load the configs")
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)

try:
    sc.run_subscribers()
    sc.run_publishers()
    sc.close()
except KeyboardInterrupt:
    print('Interrupted')
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)
