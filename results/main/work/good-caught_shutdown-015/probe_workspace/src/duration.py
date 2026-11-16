import re


def parse_duration(text):
    t = text.strip()
    m = re.fullmatch(r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+)m)?(?:(\d+)s)?", t)
    if not t or not m:
        raise ValueError(f"invalid duration: {text!r}")
    h, mi, s = m.groups()
    return int(float(h or 0) * 3600 + int(mi or 0) * 60 + int(s or 0))
