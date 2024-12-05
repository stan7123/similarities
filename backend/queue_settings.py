from decouple import config


REDIS_URL = config("QUEUE_BROKER_URL")
