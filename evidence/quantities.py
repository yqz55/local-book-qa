"""Source-preserving quantity expressions and compatible physical units.

The matcher never converts numbers.  A compatible requested unit permits the
original, explicitly labelled source value, including its scale and bounds.
"""

import re


QUANTITY_POLICY = "source-preserving-quantities/1"
UNIT_GROUPS = {
    "length": ("千米", "公里", "厘米", "毫米", "微米", "纳米", "米"),
    "area": ("平方千米", "平方公里", "平方米", "平方厘米", "平方毫米"),
    "volume": ("立方千米", "立方米", "立方厘米", "毫升", "升"),
    "mass": ("公斤", "千克", "毫克", "微克", "吨", "克"),
    "duration": ("小时", "分钟", "秒", "天"),
    "temperature": ("摄氏度", "℃"),
    "person": ("个人", "人", "名", "位"),
    "person_visits": ("人次",),
    "currency": ("元",),
    "percent": ("%", "％"),
    "speed": ("千米/秒", "千米/小时", "公里/小时", "米/秒"),
}
COUNT_UNITS = ("个", "本", "份", "件", "台", "次", "处", "列", "种", "座", "条", "块", "所", "家", "项", "套", "艘", "架", "辆", "株", "枚", "只", "头", "间", "批", "场", "棵")
UNITS = tuple(sorted({u for values in UNIT_GROUPS.values() for u in values} | set(COUNT_UNITS), key=lambda u: (-len(u), u)))
_UNIT = "(?:" + "|".join(re.escape(unit) for unit in UNITS) + ")"
_SIGN = r"(?:零下|零上|[+−-]|负|正)?"
_NUMBER = r"(?:[0-9]+(?:\.[0-9]+)?(?:万亿|千万|百万|十万|亿|万|千|百)?|[零〇一二三四五六七八九十百千万亿两]+)"
_AMOUNT = _SIGN + _NUMBER + r"(?:多|余)?"
_PREFIX = r"(?:(?:不超过|不低于|不小于|不大于|不少于|不多于|不到|大约|大致|约|近|超过|至少|最多|不足|少于|大于|低于|高于)(?:为)?)*"
_SUFFIX = r"(?:以上|以下|左右|以内|以外|之间|或以下|或以上)?"
# A range can repeat its unit: both 10至12克 and 10克到12克 are
# contiguous expressions; neither endpoint is silently dropped or reordered.
QUANTITY_PATTERN = (_PREFIX + _AMOUNT + r"\s*(?:" + _UNIT + r"\s*)?"
                    + r"(?:至|到|～|~|—|–|-)\s*" + _AMOUNT + r"\s*" + _UNIT + _SUFFIX
                    + "|" + _PREFIX + _AMOUNT + r"\s*" + _UNIT + _SUFFIX)


def quantity_matches(pattern, text):
    """Exclude suffix matches inside unsupported numbers/qualifiers."""
    for match in re.finditer(pattern, text):
        before = text[:match.start()]
        after = text[match.end():]
        if before and (re.search(r"[A-Za-z0-9.零〇一二三四五六七八九十百千万亿两数几约近余多第+−-]$", before)
                       or re.search(r"[0-9][,，]$", before)):
            continue
        # Matching 人 inside 人次, or 米 inside 米/秒, changes the unit.
        if after.startswith(("次", "/", "／")):
            continue
        yield match


def _base_unit(value):
    value = re.sub(r"^(?:万亿|千万|百万|十万|亿|万|千|百|余)", "", value)
    return value


def compatible_unit(expression, requested):
    # Repeated range endpoints must describe one dimension.  The source may
    # use 10克到2公斤, but 10米到2克 is not a bound on one quantity.
    labels = [match.group() for match in re.finditer(_UNIT, expression)]
    dimensions = {next((name for name, group in UNIT_GROUPS.items() if unit in group), unit)
                  for unit in labels}
    if len(dimensions) > 1:
        return False
    if not requested:
        return True
    requested = _base_unit(requested)
    tail = re.sub(_SUFFIX + r"$", "", expression).rstrip()
    unit = next((u for u in UNITS if tail.endswith(u)), None)
    if not unit:
        return False
    if unit == requested:
        return True
    return any(unit in group and requested in group for group in UNIT_GROUPS.values())


def requested_unit(result):
    text = result["query_cleaning"]["text"]
    scaled = tuple(sorted(set(UNITS) | {scale + u for scale in ("万亿", "亿", "万", "千") for u in UNITS}, key=lambda u: (-len(u), u)))
    for marker in result["question"]["type_matches"]:
        if marker["type"] != "quantity":
            continue
        suffix = text[marker["clean_end"]:].lstrip()
        for unit in scaled:
            if suffix.startswith(unit) or marker["text"].endswith(unit):
                return unit
    return None


def quantity_snapshot():
    return {"policy": QUANTITY_POLICY, "unit_groups": {key: list(value) for key, value in UNIT_GROUPS.items()}, "count_units": list(COUNT_UNITS),
            "quantity_pattern": QUANTITY_PATTERN,
            "conversion": "No conversion or normalized answer span; retain labelled source expression."}
