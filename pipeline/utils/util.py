import time
import logging
import progressbar


def day_of_week(idxs):
    days = ((idxs.second + idxs.minute * 60 + idxs.hour * 60 * 60 + idxs.dayofweek * 24 * 60 * 60) // (24 * 60)) % 7
    encoders = []
    for day in days:
        if day == 0:
            encoders.append(1)
        elif day == 1 or day == 2 or day == 3 or day == 4:
            encoders.append(2)
        elif day == 5 or day == 6:
            encoders.append(3)
    return encoders


def minute_of_day(idxs):
    minute_of_day = ((idxs.second + idxs.minute * 60 + idxs.hour * 60 * 60 + idxs.dayofweek * 24 * 60 * 60) % (24 * 60))
    return minute_of_day


def progressbar_sleep(seconds: int):
    logging.debug("Sleeping for: " + str(seconds) + " seconds")
    for i in progressbar.progressbar(range(seconds)):
        time.sleep(1)

if __name__ == '__main__':
    progressbar_sleep(5)