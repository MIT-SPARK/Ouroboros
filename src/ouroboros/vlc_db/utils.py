import datetime

EPOCH = datetime.datetime.utcfromtimestamp(0)


def epoch_ns_from_datetime(dt):
    return (dt - EPOCH).total_seconds() * int(1e9)
