import requests
import logging
from kubernetes import client, config


def change_fetching_state(
                          enable: bool,
                          prometheus_url="http://127.0.0.1:9090/",
                          group="monitoring.coreos.com",
                          version="v1",
                          namespace="default",
                          plural="servicemonitors",
                          name="csv-exporter",
                          ) -> bool:
    logging.debug("Changing fetching state to:" + str(enable))
    config.load_config()
    api_custom = client.CustomObjectsApi()
    try:
        conf = api_custom.get_namespaced_custom_object(
            group=group,
            version=version,
            namespace=namespace,
            plural=plural,
            name=name,
        )
    except Exception as e:
        logging.error("Could not get the monitoring service: " + str(e))
        return False

    if enable:
        conf['metadata']['labels']['release'] = "prometheus"
    else:
        # setting to a random string, so the prometheus does not scrape it
        conf['metadata']['labels']['release'] = "disable"

    try:
        api_custom.patch_namespaced_custom_object(
            group=group,
            version=version,
            namespace=namespace,
            plural=plural,
            name=name,
            body=conf)
    except Exception as e:
        logging.error("Could not patch the monitoring service: " + str(e))
        return False

    r = requests.post(prometheus_url + '-/reload')
    if r.status_code != 200:
        logging.error("Reloading the prometheus config was not successful: " + str(r.status_code))
        return False

    logging.debug("Changed fetching state to:" + str(enable))

    return True


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)-4s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    change_fetching_state(False)
