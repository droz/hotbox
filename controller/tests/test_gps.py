from __future__ import annotations

from datetime import datetime, timezone

from hotbox_controller.gps import parse_nmea_sentence


def test_parse_rmc_sentence() -> None:
    # Example RMC at Burning Man coordinates-ish
    sentence = "$GPRMC,123519,A,4047.1840,N,11912.3900,W,022.4,084.4,230394,003.1,W*6A"
    # checksum may not match our crafted fields; build a valid checksumed sentence
    body = "GPRMC,123519,A,4047.1840,N,11912.3900,W,022.4,084.4,230394,003.1,W"
    checksum = 0
    for ch in body:
        checksum ^= ord(ch)
    sentence = f"${body}*{checksum:02X}"
    parsed = parse_nmea_sentence(sentence)
    assert parsed is not None
    assert parsed["valid"] is True
    assert abs(parsed["latitude_deg"] - (40 + 47.1840 / 60.0)) < 1e-6
    assert abs(parsed["longitude_deg"] - (-(119 + 12.3900 / 60.0))) < 1e-6
    assert isinstance(parsed["when_utc"], datetime)
    assert parsed["when_utc"].tzinfo == timezone.utc


def test_parse_gga_sentence() -> None:
    body = "GPGGA,123519,4047.1840,N,11912.3900,W,1,08,0.9,1190.0,M,46.9,M,,"
    checksum = 0
    for ch in body:
        checksum ^= ord(ch)
    sentence = f"${body}*{checksum:02X}"
    parsed = parse_nmea_sentence(sentence)
    assert parsed is not None
    assert parsed["valid"] is True
    assert parsed["altitude_m"] == 1190.0
    assert parsed["satellites"] == 8
