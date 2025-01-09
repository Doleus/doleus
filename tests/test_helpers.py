from moonwatcher.utils.helpers import get_current_timestamp


def test_get_current_timestamp():
    timestamp = get_current_timestamp()

    assert isinstance(timestamp, str)
    assert len(timestamp) == 32
    assert timestamp[4] == "-"
    assert timestamp[7] == "-"
    assert timestamp[10] == "T"
    assert timestamp[13] == ":"
    assert timestamp[16] == ":"
    assert timestamp[19] == "."
    assert timestamp[26] == "+"
    # If summertime, the timezone is +02:00
    # If wintertime, the timezone is +01:00
    assert timestamp[-6:] in ["+02:00", "+01:00"]
