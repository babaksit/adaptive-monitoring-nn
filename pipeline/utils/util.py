import time
import logging
import progressbar

def progressbar_sleep(seconds: int):
    logging.debug("Sleeping for: " + str(seconds) + " seconds")
    for i in progressbar.progressbar(range(seconds)):
        time.sleep(1)
