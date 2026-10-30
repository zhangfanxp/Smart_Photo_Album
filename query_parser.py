"""Small, conservative Chinese query-to-filters parser for the local photo album.

Does not call a language model. All hard filters are supported by indexed JSON;
unsupported semantics stay in the text query for vector ranking.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

Condition = tuple[str, str, object]

_NUM = r"(?:\d+|[一二两三四五六七八九十]+)"
_NUM_PATTERN = re.compile(rf"^{_NUM}$")
_CHINESE = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
            "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}

_PERSON_WORD = r"(?:小男孩|男孩子|男童|小女孩|女童|中年男人|中年男性|中年男子|儿童|小孩|孩子|幼儿|女性|女人|成年女性|男人|男性|男子|人物|人|行人)"
_NUMBERED = re.compile(rf"(?P<qual>只有|仅有|仅|恰好|正好|总共|一共|至少|最少|不少于|至多|最多|不超过|超过|多于|有)?(?P<num>{_NUM})(?:个|名|位|只)?(?P<name>{_PERSON_WORD})")

_NEGATIVE = [
    ("male_child_count", "child_gender_complete", re.compile(r"(?:没有|没|不含|不包含|无|看不到)(?:任何|一个|一名|一位)?(?:小男孩|男孩子|男童)")),
    ("female_child_count", "child_gender_complete", re.compile(r"(?:没有|没|不含|不包含|无|看不到)(?:任何|一个|一名|一位)?(?:小女孩|女童)")),
    ("child_count", "person_age_complete", re.compile(r"(?:没有|没|不含|不包含|无|看不到)(?:任何|一个|一名|一位)?(?:孩子|小孩|儿童|幼儿)")),
    ("adult_female_count", "adult_gender_complete", re.compile(r"(?:没有|没|不含|不包含|无|看不到)(?:任何|一个|一名|一位)?(?:成年女性|女性|女人)")),
    ("people_count", "people_count_exact", re.compile(r"(?:空无一人|无人(?!机)|没人|(?:没有|没|不含|不包含|无|看不到)(?:任何|一个|一名|一位)?(?:人物|人员|行人|路人|人)(?!工智能))")),
    ("animals_count", "animals_count_exact", re.compile(r"(?:没有|没|不含|不包含|无|看不到)(?:任何|一只|一个)?(?:动物|小动物)")),
]
_TIME = [("night", re.compile(r"夜晚|夜间|深夜|黑夜|晚上")),
         ("daytime", re.compile(r"白天|日间")),
         ("dawn", re.compile(r"黎明|清晨|破晓")),
         ("dusk", re.compile(r"黄昏|傍晚"))]
_SKY_BLUE = re.compile(r"蓝天|蓝色天空|(?:蔚蓝|碧蓝|湛蓝)(?:的)?天空")
_NO_SKY_BLUE = re.compile(r"(?:没有|没|不含|不包含|无|看不到)(?:任何)?(?:蓝天|蓝色天空|(?:蔚蓝|碧蓝|湛蓝)(?:的)?天空)")
_RED_FLAG = re.compile(r"五星红旗|红旗|(?:红色|鲜红色?)(?:的)?旗(?:帜|子)?|红色党旗")
_NO_RED_FLAG = re.compile(r"(?:没有|没|不含|不包含|无|看不到)(?:任何|一个|一面|一杆)?(?:五星红旗|红旗|(?:红色|鲜红色?)(?:的)?旗(?:帜|子)?|红色党旗)")
_RED_HANDPRINT = re.compile(r"红(?:色)?手(?:掌)?印|手(?:掌)?印(?:是|为)?红(?:色)?")
_NO_RED_HANDPRINT = re.compile(r"(?:没有|没|不含|不包含|无|看不到)(?:任何|一个|一枚)?(?:红(?:色)?手(?:掌)?印|手(?:掌)?印(?:是|为)?红(?:色)?)")
_CLOTHING_TERM = re.compile(r"衣服|衣着|衣物|衣|穿着|穿|服装|上衣|外套|T恤|衬衫|裤子|裙子|卫衣|毛衣|夹克|连衣裙", re.I)
_CLOTHING_AFTER_COLOR = re.compile(r"^(?:的)?(?:衣服|衣着|衣物|上衣|外套|T恤|衬衫|裤子|裙子|卫衣|毛衣|夹克|连衣裙|衣)")
_CLOTHING_BEFORE_COLOR = re.compile(r"(?:衣服|衣着|衣物|上衣|外套|T恤|衬衫|裤子|裙子|卫衣|毛衣|夹克|连衣裙|衣|穿着|身穿|穿)(?:.{0,4})$")
_CLOTHING_COLORS = {
    "green": re.compile(r"绿(?:色)?"),
    "orange": re.compile(r"橙(?:色)?|橘(?:色)?"),
    "pink": re.compile(r"粉(?:红)?(?:色)?"),
    "purple": re.compile(r"紫(?:红)?(?:色)?"),
    "red": re.compile(r"红(?:色)?"),
    "yellow": re.compile(r"黄(?:色)?"),
    "blue": re.compile(r"蓝(?:色)?"),
    "black": re.compile(r"黑(?:色)?"),
    "white": re.compile(r"白(?:色)?"),
    "gray": re.compile(r"灰(?:色)?|银灰"),
    "brown": re.compile(r"棕(?:色)?|褐(?:色)?"),
}
_STOP = re.compile(r"(?:请帮我|请帮忙|给我|帮我|查找|找到|找出|搜索|检索|照片|图片|相片|图像|拍摄的|拍的|有关|关于)")


@dataclass(frozen=True)
class QueryPlan:
    original_query: str
    semantic_query: str
    conditions: tuple[Condition, ...]
    unsupported_logic: bool = False

    @property
    def metadata_only(self) -> bool:
        return bool(self.conditions) and not self.semantic_query


def cn_num(value: str) -> int | None:
    if not _NUM_PATTERN.fullmatch(value):
        return None
    if value.isdecimal():
        return int(value)
    if value in _CHINESE:
        return _CHINESE[value]
    if len(value) == 2 and value.startswith("十"):
        return 10 + _CHINESE.get(value[1], 0)
    if len(value) == 2 and value.endswith("十"):
        return _CHINESE.get(value[0], 0) * 10
    if len(value) == 3 and value[1] == "十":
        return _CHINESE.get(value[0], 0) * 10 + _CHINESE.get(value[2], 0)
    return None


def _field_for_name(name: str) -> str | None:
    if name in ("小男孩", "男孩子", "男童"):
        return "male_child_count"
    if name in ("小女孩", "女童"):
        return "female_child_count"
    if name in ("孩子", "小孩", "儿童", "幼儿"):
        return "child_count"
    if name in ("成年女性", "女性", "女人"):
        return "adult_female_count"
    if name in ("中年男人", "中年男性", "中年男子"):
        return "middle_aged_male_count"
    if name in ("人物", "人", "行人"):
        return "people_count"
    # Bare man/woman is not a precise age category; avoid hard-filtering '男人'.
    return None


def parse_query(query: str) -> QueryPlan:
    """Return constraints and a residual *positive* semantic query.

    Negations / exact counts are stripped from vector text so embeddings don't
    accidentally rank photos *containing* excluded subjects more highly.
    """
    q = re.sub(r"\s+", "", query.strip())
    conditions: list[Condition] = []
    used = []
    for field, proof, pattern in _NEGATIVE:
        for m in pattern.finditer(q):
            # Require proof that a zero count is meaningful. Subject-specific
            # zero counts also demand complete age/gender labels, not just 0.
            conditions.extend([(proof, "eq", True), (field, "eq", 0)])
            used.append((m.start(), m.end()))

    for m in _NO_SKY_BLUE.finditer(q):
        conditions.append(("sky_blue", "eq", False))
        used.append((m.start(), m.end()))

    # A positive red-flag query must have explicit visual flag evidence in the
    # indexed description; semantic similarity alone also recalls red slogans
    # and political scenes without any flag. Mask negated phrases so
    # "没有红旗" is never interpreted as a request for images with red flags.
    red_flag_query = list(q)
    for m in _NO_RED_FLAG.finditer(q):
        red_flag_query[m.start():m.end()] = " " * (m.end() - m.start())
    for m in _RED_FLAG.finditer("".join(red_flag_query)):
        conditions.append(("red_flag_visible", "eq", True))

    red_handprint_query = list(q)
    for m in _NO_RED_HANDPRINT.finditer(q):
        red_handprint_query[m.start():m.end()] = " " * (m.end() - m.start())
    for m in _RED_HANDPRINT.finditer("".join(red_handprint_query)):
        conditions.append(("red_handprint_visible", "eq", True))
        used.append(m.span())

    # Mask matches to ensure "没有一个小男孩" isn't also read as "一个小男孩".
    mask = list(q)
    for start, end in used:
        mask[start:end] = " " * (end-start)
    positive = "".join(mask)

    if _SKY_BLUE.search(positive):
        # A positive visual attribute is a required metadata condition as well
        # as a semantic ranking term. Keep it in the residual query below.
        conditions.append(("sky_blue", "eq", True))

    for m in _NUMBERED.finditer(positive):
        num = cn_num(m.group("num"))
        field = _field_for_name(m.group("name"))
        if num is None or field is None:
            continue
        qual = m.group("qual") or ""
        # "只有一个小男孩" means nobody else is in the frame; but
        # "只有一个女性和两个孩子" does not mean only one person total.
        extra_people = bool(re.search(r"(?:和|与|及|、).{0,5}(?:孩子|小孩|儿童|男孩|女孩|女性|男人|人)", positive[m.end():]))
        if qual in ("只有", "仅有", "仅", "恰好", "正好", "总共", "一共") and not extra_people:
            if field != "people_count":
                conditions.append(("people_count_exact", "eq", True))
                conditions.append(("people_count", "eq", num))
            else:
                conditions.extend([("people_count_exact", "eq", True), (field, "eq", num)])
        elif qual in ("至多", "最多", "不超过"):
            # Upper bounds need a confirmed number of people when applied to people.
            if field == "people_count":
                conditions.append(("people_count_exact", "eq", True))
            conditions.append((field, "le", num))
        elif qual in ("超过", "多于"):
            conditions.append((field, "ge", num + 1))
        else:
            conditions.append((field, "ge", num))
        # Remove only the numeral/quantifier, preserve category for semantic ranking.
        start, end = m.span()
        prefix_len = len(m.group(0)) - len(m.group("name"))
        used.append((start, start + prefix_len))

    # An unnumbered subject term should be a presence condition, provided it is
    # not inside any negation. Longest expressions first.
    covered = [False] * len(q)
    for start, end in used:
        for k in range(start, min(end, len(covered))):
            covered[k] = True
    unnumbered = [
        ("middle_aged_male_count", r"中年男人|中年男性|中年男子"),
        ("male_child_count", r"小男孩|男孩子|男童"),
        ("female_child_count", r"小女孩|女童"),
        ("adult_female_count", r"成年女性|女性|女人"),
        ("child_count", r"小孩|孩子|儿童|幼儿"),
    ]
    for field, expression in unnumbered:
        for m in re.finditer(expression, q):
            if any(covered[m.start():m.end()]):
                continue
            if not any(c[0] == field for c in conditions):
                conditions.append((field, "ge", 1))

    # Clothing colors are explicit visual attributes on each person. Require
    # the requested color on one person's clothing instead of relying on the
    # embedding model to distinguish colors. Age stays soft because VLM age
    # estimates can be uncertain; an explicit gender can tie the clothing to
    # the requested person in group photos.
    color_text = list(q)
    clothing_colors = []
    for color, pattern in _CLOTHING_COLORS.items():
        masked = "".join(color_text)
        matches = list(pattern.finditer(masked))
        if not matches:
            continue
        for match in matches:
            before = q[max(0, match.start() - 8):match.start()]
            after = q[match.end():min(len(q), match.end() + 8)]
            tied_to_clothing = bool(
                _CLOTHING_AFTER_COLOR.match(after)
                or _CLOTHING_BEFORE_COLOR.search(before)
            )
            if not tied_to_clothing:
                continue
            clothing_colors.append(color)
            color_text[match.start():match.end()] = " " * (match.end() - match.start())
    if clothing_colors and _CLOTHING_TERM.search(q):
        has_male = bool(re.search(r"小男孩|男孩子|男童|男孩|男性|男人|男子", q))
        has_female = bool(re.search(r"小女孩|女孩子|女童|女孩|女性|女人|女子", q))
        gender = "male" if has_male and not has_female else "female" if has_female and not has_male else None
        person_spec = (gender, tuple(clothing_colors))
        conditions.append(("person_match", "matches", person_spec))

    for time, pattern in _TIME:
        m = pattern.search(q)
        if m and not any(covered[m.start():m.end()]):
            conditions.append(("time_of_day", "eq", time))
            for one in pattern.finditer(q):
                used.append(one.span())
            break

    # Strip excluded phrases and numeric qualifiers, preserving '海边' etc.
    stripped = list(q)
    for start, end in used:
        stripped[start:end] = [" "] * (end-start)
    semantic = "".join(stripped).replace(" ", "")
    semantic = _STOP.sub("", semantic)
    semantic = re.sub(r"[,，;；]+", "", semantic)
    semantic = re.sub(r"^[的有在和与、。]+|[的有在和与、。]+$", "", semantic)
    # Exclude fillers if a query consists exclusively of hard constraints.
    if semantic in {"", "的", "全部", "所有", "所有的"}:
        semantic = ""

    return QueryPlan(
        original_query=query,
        semantic_query=semantic or ("" if conditions else query),
        conditions=tuple(dict.fromkeys(conditions)),
        unsupported_logic=bool(re.search(r"或者|或是|要么|\bOR\b", q, re.I)),
    )


def meets_filters(fields: dict, conditions: tuple[Condition, ...] | list[Condition]) -> bool:
    for key, op, expected in conditions:
        if key == "person_match":
            if op != "matches" or not isinstance(expected, tuple) or len(expected) != 2:
                return False
            persons = fields.get("person_details", [])
            if not isinstance(persons, list):
                return False
            gender, colors = expected
            matched_person = False
            for person in persons:
                if not isinstance(person, dict):
                    continue
                if gender is not None and person.get("gender_presentation") != gender:
                    continue
                clothing = str(person.get("clothing") or "")
                if isinstance(colors, tuple) and all(
                    color in _CLOTHING_COLORS and _CLOTHING_COLORS[color].search(clothing)
                    for color in colors
                ):
                    matched_person = True
                    break
            if not matched_person:
                return False
            continue
        got = fields.get(key)
        if op == "eq":
            if got != expected:
                return False
        elif op in {"ge", "le"}:
            if type(got) not in (int, float):
                return False
            if op == "ge" and got < expected:
                return False
            if op == "le" and got > expected:
                return False
        else:
            raise ValueError(f"Unsupported operator: {op}")
    return True
