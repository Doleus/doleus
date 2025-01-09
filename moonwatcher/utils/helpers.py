import datetime

import pytz


def get_current_timestamp() -> str:
    tz = pytz.timezone("Europe/Berlin")
    timestamp = datetime.datetime.now(tz=tz).isoformat()
    return timestamp
