"""Corpus-driven person-name discovery with explicit context and ambiguity rules.

No person names, book titles, or name-pair allowlists are used by this module.
Surname, title, grammar, and action vocabularies are linguistic rule resources.
"""

from collections import Counter, defaultdict
import hashlib
import json
import re

from .ingest import SourceMap
from .tokenization import is_han

PERSON_POLICY = "corpus-person-rules/1"
HAN = r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323af]"
SURNAMES = """赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳酆鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田樊胡凌霍虞万支柯昝管卢莫经房裘缪干解应宗丁宣贲邓郁单杭洪包诸左石崔吉钮龚程嵇邢滑裴陆荣翁荀羊惠甄曲家封芮羿储靳汲邴糜松井段富巫乌焦巴弓牧隗山谷车侯宓蓬全郗班仰秋仲伊宫宁仇栾暴甘钭厉戎祖武符刘景詹束龙叶幸司韶郜黎蓟薄印宿白怀蒲邰从鄂索咸籍赖卓蔺屠蒙池乔阴胥能苍双闻莘党翟谭贡劳逄姬申扶堵冉宰郦雍郤璩桑桂濮牛寿通边扈燕冀郏浦尚农温别庄晏柴瞿阎充慕连茹习宦艾鱼容向古易慎戈廖庾终暨居衡步都耿满弘匡国文寇广禄阙东欧殳沃利蔚越夔隆师巩厍聂晁勾敖融冷訾辛阚那简饶空曾毋沙乜养鞠须丰巢关蒯相查后荆红游竺权逯盖益桓公晋楚闫岳仉佟商牟佘佴伯赏墨哈谯笪年爱阳佗言福趙錢孫鄭馮陳衛蔣韓楊許呂張嚴華鄒魯韋馬費薛賀湯羅畢鄔樂時齊顧黃蕭鄒賈婁顏鍾駱盧繆應鄧單榮陸劉龍葉雙聞藺賴喬陰譚貢酈壽邊溫莊閻魚廣祿闕歐聶饒簡蓋權愛陽"""
COMPOUND_SURNAMES = "欧阳 太史 端木 上官 司马 东方 独孤 南宫 万俟 闻人 夏侯 诸葛 尉迟 公羊 赫连 澹台 皇甫 宗政 濮阳 公冶 太叔 申屠 公孙 慕容 仲孙 钟离 长孙 宇文 司徒 鲜于 司空 闾丘 子车 亓官 司寇 巫马 公西 颛孙 壤驷 公良 漆雕 乐正 宰父 谷梁 拓跋 夹谷 轩辕 令狐 段干 百里 呼延 东郭 南门 羊舌 微生 梁丘 左丘 东门 西门 歐陽 司馬 東方 獨孤 南宮 聞人 諸葛 尉遲 赫連 皇甫 公孫 鍾離 長孫 鮮于 閭丘 司寇 巫馬 顓孫 穀梁 夾谷 軒轅 東郭 南門 東門 西門".split()
SPEECH = "说道 问道 答道 笑道 叹道 哭道 回道 应道 叫道 骂道 喊道 写道 嚷道 泣道 禀道 禀告 說道 問道 歎道 應道 罵道 喊道 寫道 稟道 稟告 说话 问话 答话 回话 說話 問話 答話 回話".split()
ACTIONS = "点头 摇头 起身 行礼 作揖 拱手 落泪 抽泣 叹气 叹口气 开口 起床 俯身 伸手 摆手 拍手 咳嗽 沉吟 告辞 吩咐 嘱咐 应声 安慰 责骂 咬牙 堆着笑 出生 去世 病逝 生于 享年 出嫁 成婚 娶妻 入宫 受封 升任 出任 就任 赴任 拜见 参拜 點頭 搖頭 行禮 落淚 歎氣 歎口氣 開口 擺手 告辭 囑咐 應聲 責罵 生於 入宮 拜見 參拜".split()
TITLES = "先生 女士 老师 教授 主任 经理 医生 队长 警官 博士 小姐 阿姨 姑娘 夫人 太太 老爷 大人 公子 老先生 老夫人 姨妈 姥姥 爷爷 奶奶 老太太 老太爷 叔叔 伯伯 父 母 哥 姐 师傅 师父 掌柜 老師 經理 醫生 隊長 老爺 姨媽 爺爺 老太爺 師傅 師父".split()
GENERIC = "他 她 它 你 我 您 咱 本人 自己 我们 你们 他们 她们 咱们 大家 众人 二人 两人 三人 一人 几人 那人 这人 有人 别人 旁人 各位 众位 个人 古人 人人 人家 姑娘 小姐 老爷 先生 夫人 太太 姐姐 妹妹 哥哥 弟弟 兄弟 父亲 母亲 儿子 女儿 孩子 丫头 下人 主人 客人 学生 老师 医生 记者 警察 群众 老太太 老先生 大小姐 大姑娘 小姑娘 大老爷 老婆子 小丫头 小媳妇 不知道 知道 明白 明了 不明白 原来 后来 从前 现在 当时 此时 于是 但是 所以 因为 果然 竟然 如此 这样 那样 只好 只得 不禁 一面 一边 正要 即刻 连忙 事情 问题 时间 地方 数据 科学 研究 自然语言处理 文档 词频 某某 大家伙 这个 那个 这些 那些 怎么 怎么样 什么 我們 你們 他們 她們 眾人 兩人 幾人 這人 別人 父親 母親 兒子 女兒 學生 老師 醫生 記者 警察 群眾 原來 後來 現在 當時 於是 問題 時間 地方 數據 研究".split()
LEFT_CUES = "只见 听见 看见 忽见 听了 告诉 问了 见了 望着 看着 叫 唤 问 请 让 找 同 与 和 对 向 跟 那 这 原来 此时 只見 聽見 看見 忽見 聽了 告訴 問了 見了 望著 看著 喚 問 請 讓 與 對 這 原來 此時".split()
AUXILIARY = set("便又也就却正忙仍还更先才再笑遂乃亦但即曾已忽连卻還連")
FUNCTION_LEADING = set("的地得了着把将被是便又也就却只都让叫与对跟那这一二三两几忙仍还更而他她它你我您咱众些因竟虽所做不刚为个人么听见看拉递指依据自怎里说要著將卻讓與對這兩幾還眾雖為個聽見遞據說")
AMBIGUOUS_SURNAME_PREFIX = set("向于和同从后连任闻过应时年边扶方幸别家满何尚空万明蒙曾诸从後連聞過應時邊滿萬諸")
BOUNDARIES = set("\n\r\t \u3000。！？!?；;，,：:“”‘’\"'（）()《》【】[]—…")

PERSON_RULES = {
    "N001": "由本书姓名规则发现或人工校正的名称优先整词匹配；同起点长名称优先，其余文字仍按书内词频和固定回退处理。",
    "N101": "从说话、人物动作等上下文枚举 2–4 字候选；边界有效且达到全书出现次数和独立上下文次数阈值时接受。",
    "N102": "单姓或复姓加 1–2 字名的结构，有人物上下文支持时接受；常见代词、普通称谓和语法片段拒绝。",
    "N103": "明确的名叫、姓名、又名、字、号等介绍句可提供名称或同一人物关系；保存介绍句的原文位置。",
    "N104": "已发现的二字人名之前，若在原文中实际出现单姓或复姓并有边界支持，可提出全名候选；介词歧义姓氏不凭拼接直接接受。",
    "N105": "姓氏加人物称谓可作为完整人物称呼保护；普通称谓本身不据此认定为某一个人。",
    "N106": "姓氏姓名结构有亲属关系或官职介绍的直接上下文、且在书中重复出现，可接受为人物名称候选。",
    "N107": "已发现全名的二字去姓称呼，有独立人物上下文和独立出现时可整词保护；是否为同一人物另行判断。",
    "N109": "普通语法词、只有词频而无人物上下文、或姓名后附带的语气/动作片段不自动作为人名。",
    "N201": "全名去掉已识别的姓得到二字简称；简称须有独立出现及人物上下文，且在本书中只有一个候选全名，才自动关联。",
    "N202": "同一个简称对应多个不同全名时记录歧义，不把人物直接合并；单字、普通称谓和普通词不自动作为简称。",
    "N203": "原文明示的又名、也叫、字或号关系可关联；相同称呼同时连接不相容的全名时保留歧义。",
}


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def default_config():
    return {"policy": PERSON_POLICY, "enabled": True, "min_count": 2, "min_contexts": 2,
            "max_name_chars": 4, "evidence_limit": 3,
            "surnames": "".join(sorted(set(SURNAMES))), "compound_surnames": sorted(set(COMPOUND_SURNAMES)),
            "speech_cues": sorted(set(SPEECH + '劝道 啐道 叱道 讥道 勸道 嘆道'.split())), "action_cues": sorted(set(ACTIONS)),
            "predicate_modifiers": '含笑 含泪 含 哭 嚷 悄悄 悄 早 疾 未 喜 嗔 咳 惊讶 正色 红着脸 乃 所 看 求 大'.split(),
            "titles": sorted(set(TITLES + '嫂子 太医 姨娘 大夫 大爷 妈妈 婆子 嬷嬷 奴才 圣人 道人 道士 道婆 仙姑 娘娘 皇上 皇帝 皇后 太后 天子 陛下 元妃 贵妃 公主 王爷 妃子 大舅 奶子 婶娘 姊妹 丫鬟 丫鬟们 媳妇 太监 门子 小厮 老尼 老嬷嬷'.split())),
            "generic_words": sorted(set(GENERIC + '家母 家父 父母 家人 家父母 家下人 家老爷 祖母 祖父 女人 妇人 姑子 女尼 仙女 女先儿 会子 四人 一个 两个 三个 不便 不大 不必 不敢 不知 不觉 不用 不过 不消 不妨 不料 不想 不曾 不如 不得 不能 不是 不要 不可 再不 刚才 刚要 半日 半晌 明儿 明日 今儿 今日 起来 起身 过来 进来 出来 回来 回头 听见 听得 喜欢 含笑 含泪 高声 厉声 悄悄 忽然 然后 既然 彼此 口里 心里 心下 心中 口中 嘴里 口内 耳边 眼里 眼中 肚里 内中 里边 外边 外头 上面 上头 地下 旁边 对面 面前 要紧 才知 人来 人回 人知 既如此 既这么 既这样 俗语 有话 旨意 灯谜 陪笑 笑劝 喜得 大怒 大概 诧异 随口 少胡 休胡 诗词 勉强 主上 胡说 公道 索性 莫若 管保 经上 常听人 终是要 书办 知县'.split())),
            "left_cues": sorted(set(LEFT_CUES)), "auxiliary_chars": "".join(sorted(AUXILIARY)),
            "function_leading": "".join(sorted(FUNCTION_LEADING)),
            "ambiguous_surname_prefixes": "".join(sorted(AMBIGUOUS_SURNAME_PREFIX))}


def _configuration(config):
    result = default_config() if config is None else dict(config)
    expected = default_config()
    if result.keys() != expected.keys() or result["policy"] != PERSON_POLICY:
        raise ValueError("不支持的人名发现规则配置。")
    if type(result["enabled"]) is not bool:
        raise ValueError("人名发现 enabled 必须为布尔值。")
    for key in ("min_count", "min_contexts", "max_name_chars", "evidence_limit"):
        if type(result[key]) is not int or not 1 <= result[key] <= 20:
            raise ValueError(f"人名规则参数 {key} 必须是 1–20 的整数。")
    if result["max_name_chars"] != 4:
        raise ValueError("本版人名上下文规则支持 2–4 字名称。")
    for key, value in expected.items():
        if isinstance(value, list) and (not isinstance(result[key], list) or any(not isinstance(v, str) or not v for v in result[key])):
            raise ValueError(f"人名规则资源 {key} 必须是非空字符串数组。")
        if isinstance(value, str) and not isinstance(result[key], str):
            raise ValueError(f"人名规则资源 {key} 必须是字符串。")
    return result


def _trie(words):
    trie = {}
    for word in sorted(words):
        node = trie
        for char in word:
            node = node.setdefault(char, {})
        node[""] = word
    return trie


def _matches(text, trie):
    for start, char in enumerate(text):
        if char not in trie:
            continue
        node, end = trie, start
        while end < len(text) and text[end] in node:
            node = node[text[end]]
            end += 1
            if "" in node:
                yield node[""], start, end


def discover_people(cleaned, raw, config=None):
    cfg = _configuration(config)
    text = cleaned["text"]
    mapping = SourceMap(cleaned["source_runs"], raw)
    surnames, compounds = set(cfg["surnames"]), sorted(cfg["compound_surnames"], key=lambda s: (-len(s), s))
    ambiguous_prefixes = set(cfg["ambiguous_surname_prefixes"])
    generic = set(cfg["generic_words"]) | set(cfg["titles"]) | set(cfg["action_cues"]) | set(cfg["speech_cues"]) | set(cfg["predicate_modifiers"])
    left_cues = sorted(cfg["left_cues"], key=lambda s: (-len(s), s))
    leading, aux = set(cfg["function_leading"]), set(cfg["auxiliary_chars"])
    records, definition_edges = {}, []

    def left_reason(start):
        if start == 0 or text[start - 1] in BOUNDARIES or not is_han(text[start - 1]):
            return "分隔边界"
        for cue in left_cues:
            if text.endswith(cue, 0, start):
                return f"左上下文：{cue}"
        return None

    def surname_of(name):
        for surname in compounds:
            if name.startswith(surname) and 1 <= len(name) - len(surname) <= 2:
                return surname
        return name[0] if name and name[0] in surnames and 1 <= len(name) - 1 <= 2 else None

    def possible(name, explicit=False):
        if not 2 <= len(name) <= 4 or not all(is_han(c) for c in name) or name in generic:
            return False
        if not explicit and name[0] in leading:
            return False
        if not explicit and (name.endswith(("的", "地", "了", "着", "著", "们", "們")) or "的" in name):
            return False
        if name[-1] in "他她你我您吗嘛呢么吧啊呀哩啦嗎" and not explicit:
            return False
        if not explicit and any(c in name for c in "他她你我您"):
            return False
        if not explicit and not surname_of(name) and any(c in name for c in "的地得着著了"):
            return False
        surname = surname_of(name)
        if surname and name[len(surname):] in generic and name[len(surname):] not in cfg["titles"]:
            return False
        return True

    def record(name):
        return records.setdefault(name, {"contexts": {}, "introductions": [], "recovered": [], "title": False, "constructed": False})

    def example(start, end, rule, trigger):
        a, b = mapping.span(start, end)
        ca, cb = mapping.span(max(0, start - 12), min(len(text), end + 12))
        return {"source_start": a, "source_end": b, "surface": raw[a:b],
                "context_source_start": ca, "context_source_end": cb, "context": raw[ca:cb],
                "rule": rule, "trigger": trigger}

    if cfg["enabled"]:
        cues = sorted(set(cfg["speech_cues"] + cfg["action_cues"]), key=lambda s: (-len(s), s))
        cue_pattern = "|".join(map(re.escape, cues)) + r"|[说說问問答道](?=[：:，,。！？!?“”‘’\"'了着著过過])"
        # A cue inside an ordinary word (e.g. the final character of a verb)
        # does not supply evidence for a new speaker immediately before it.
        blocked_cues = {pos for word, start, end in _matches(text, _trie(cfg["generic_words"]))
                        for pos in range(start + 1, end)}
        for match in re.finditer(cue_pattern, text):
            if match.start() in blocked_cues:
                continue
            endpoints = [match.start()]
            for _ in range(3):
                end = endpoints[-1]
                if end and text[end - 1] in aux:
                    endpoints.append(end - 1)
                else:
                    break
            for end in list(endpoints):
                for modifier in cfg["predicate_modifiers"]:
                    if text.endswith(modifier, 0, end):
                        endpoints.append(end - len(modifier))
            for end in endpoints:
                for length in range(2, 5):
                    start = end - length
                    if start < 0:
                        continue
                    name = text[start:end]
                    boundary = left_reason(start)
                    if all(is_han(c) for c in name) and boundary:
                        record(name)["contexts"].setdefault(start, (start, end, match.group(), boundary))

        # Introductions are explicit textual evidence, never a known-person list.
        delimiter = r"(?=[\s，,。！？?!：:；;“”‘’\"'、]|$|的|今年|正在|[是在也便又正][\u4e00-\u9fff])"
        intro = re.compile(r"(?:名叫|名为|名為|叫作|唤作|喚作|姓名是|姓名为|姓名為|大名叫|本名是|本名为|本名為)(?P<name>" + HAN + r"{2,4}?)" + delimiter)
        for match in intro.finditer(text):
            name = match["name"]
            lead = text[max(0, match.start() - 12):match.start()]
            person_intro = bool(re.search(r"(?:男子|女子|少年|少女|男孩|女孩|朋友|同学|同事|先生|女士|姑娘|丫鬟|学生|他|她)(?:的(?:名字|本名|姓名))?[，,\s]*$", lead))
            if possible(name, True) and (surname_of(name) or person_intro or match.group().startswith(("姓名", "本名", "大名"))):
                record(name)["introductions"].append((match.start("name"), match.end("name"), "名称介绍"))

        kinship = re.compile(r"之(?:父|母|子|女|兄|弟|姐|妹|妻|夫)|的(?:父亲|母亲|儿子|女儿|哥哥|弟弟|姐姐|妹妹|妻子|丈夫)")
        for marker in kinship.finditer(text):
            for length in range(2, 5):
                start = marker.start() - length
                name = text[start:marker.start()] if start >= 0 else ""
                if start >= 0 and left_reason(start) and possible(name) and surname_of(name):
                    record(name)["introductions"].append((start, marker.start(), "人物亲属关系上下文"))

        # Explicit identity statements link observed full spans on the two sides.
        marker_pattern = re.compile(r"又名|也叫|亦称|亦稱|人称|人稱|小名(?:叫|唤|喚)?|本名(?:是|叫)?|别名(?:是|叫)?|別名(?:是|叫)?|字|号|號")
        for marker in marker_pattern.finditer(text):
            sentence_tail = re.split(r"[。！？?!\n]", text[marker.end():marker.end() + 100], maxsplit=1)
            sentence_end = marker.end() + len(sentence_tail[0])
            if sentence_end < len(text) and text[sentence_end] in "？?":
                continue
            name_end = marker.start()
            while name_end and text[name_end - 1] in " ，,\t":
                name_end -= 1
            before = []
            for length in range(2, 5):
                start = name_end - length
                name = text[start:name_end] if start >= 0 else ""
                if (start >= 0 and left_reason(start) and possible(name)
                        and surname_of(name) not in ambiguous_prefixes
                        and (surname_of(name) or len(records.get(name, {}).get("contexts", {})) >= cfg["min_contexts"]
                             or records.get(name, {}).get("introductions"))):
                    before.append((name, start, name_end))
            after = re.match(r"(?P<name>" + HAN + r"{2,4}?)" + delimiter, text[marker.end():])
            if before and after:
                before.sort(key=lambda item: (-len(item[0]), item[0]))
                a, start, stop = before[0]
                b = after["name"]
                if possible(b) and a != b:
                    bstart, bend = marker.end(), marker.end() + len(b)
                    record(a)["introductions"].append((start, stop, "明确同一人物介绍"))
                    record(b)["introductions"].append((bstart, bend, "明确同一人物介绍"))
                    definition_edges.append({"a": a, "b": b, "rule": "N203",
                                             "evidence": example(start, bend, "N203", marker.group())})

        # Surname + person title is a mention, never an automatic bare-title alias.
        surname_pattern = "|".join(map(re.escape, compounds)) + "|[" + re.escape(cfg["surnames"]) + "]"
        title_pattern = "|".join(map(re.escape, sorted((t for t in cfg["titles"] if len(t) <= 3), key=lambda s: (-len(s), s))))
        for match in re.finditer("(?:" + surname_pattern + ")(?:" + title_pattern + ")", text):
            name = match.group()
            if (len(name) <= 4 and left_reason(match.start()) and possible(name, True)
                    and surname_of(name) not in ambiguous_prefixes):
                record(name)["title"] = True

        # Structured fields can spell a full name non-contiguously. These create
        # a normalized query variant, never a fabricated document occurrence.
        surname_field = re.compile(r"姓(?P<surname>" + surname_pattern + r")[，,\s]*名(?:叫|曰|为|為)?(?P<given>" + HAN + r"{1,2}?)" + delimiter)
        for match in surname_field.finditer(text):
            full, given = match["surname"] + match["given"], match["given"]
            if possible(full, True):
                rec = record(full)
                rec["constructed"] = True
                rec["introductions"].append((match.start(), match.end(), "姓与名字段规范化"))
                if len(given) >= 2 and given not in generic:
                    record(given)["introductions"].append((match.start("given"), match.end("given"), "名字段"))
                    definition_edges.append({"a": full, "b": given, "rule": "N203",
                                             "evidence": example(match.start(), match.end(), "N203", "姓与名字段")})

    # Census all proposed strings in one bounded trie scan, not N full-book scans.
    counts, first_positions = Counter(), {}
    for name, start, end in _matches(text, _trie(records)):
        counts[name] += 1
        first_positions.setdefault(name, (start, end))

    accepted, reasons = set(), {}
    for name, rec in sorted(records.items()):
        contexts = len(rec["contexts"])
        surname = surname_of(name)
        strong_intros = [i for i in rec["introductions"] if i[2] != "人物亲属关系上下文"]
        if strong_intros and possible(name, True):
            accepted.add(name); reasons[name] = "N103"
        elif rec["introductions"] and possible(name) and surname not in ambiguous_prefixes and counts[name] >= cfg["min_count"]:
            accepted.add(name); reasons[name] = "N106"
        elif possible(name) and rec["title"] and counts[name] >= cfg["min_count"]:
            accepted.add(name); reasons[name] = "N105"
        elif possible(name) and surname and contexts >= 1 and counts[name] >= 1:
            if surname not in ambiguous_prefixes:
                accepted.add(name); reasons[name] = "N102"
        elif possible(name) and contexts >= cfg["min_contexts"] and counts[name] >= cfg["min_count"]:
            accepted.add(name); reasons[name] = "N101"

    # Remove a speech particle accidentally attached to an otherwise accepted name.
    for name in sorted(accepted):
        if name[-1] in aux and name[:-1] in accepted and not records[name]["introductions"]:
            accepted.remove(name); reasons[name] = "N109"
    continuations = set(cfg["action_cues"] + cfg["speech_cues"] + cfg["predicate_modifiers"] +
                        '听 见 想 知 冷 因 方 等 皆 都 回 急 陪 点 念 喝 啐 来 一 一面 走来 进来 听见 冷笑 点头 摇头 喜得 赶 慌 立 劝 商议 诧异 等都 忙劝 忙陪 忙拦 忙联 进来 陪笑'.split()) | aux
    tail_chars = set("的地得了着便又也就却只都不再一等皆听见想知冷因方回急陪点念喝啐来赶慌立劝自思推他笑问答走忙联拦咬牙忽然兀是说连旁乱泣嚷著卻聽見問趕勸說連")
    function_prefixes = set(cfg["left_cues"]) | set(cfg["ambiguous_surname_prefixes"]) | set("回听见看拉递指依据自怎笑忽叹啐堆勉里说连")
    grammar_chars = leading | set("可又是里边还而不着便忽一二三两几个么呢嘛吧啊哩与和已其之者所则那这")
    for name in sorted(accepted, key=lambda n: (-len(n), n)):
        if (len(name) >= 3 and not surname_of(name) and sum(c in grammar_chars for c in name) >= 2
                and not records[name]["introductions"]):
            accepted.discard(name); reasons[name] = "N109"
            continue
        for cut in range(2, len(name)):
            if ((name[:cut] in accepted or name[:cut] in generic)
                    and (name[cut:] in continuations or all(c in tail_chars for c in name[cut:]))
                    and not records[name]["introductions"]):
                accepted.discard(name); reasons[name] = "N109"
                break
        if not records[name]["introductions"]:
            for cut in range(1, len(name) - 1):
                if name[:cut] in function_prefixes and (name[cut:] in accepted or name[cut:] in generic):
                    accepted.discard(name); reasons[name] = "N109"
                    break

    shorts = {name for name in accepted if len(name) == 2 and name not in generic}
    recovered = defaultdict(dict)
    for short, start, end in _matches(text, _trie(shorts)):
        for width in (1, 2):
            begin = start - width
            if begin < 0:
                continue
            surname = text[begin:start]
            if ((width == 1 and surname in surnames) or surname in compounds) and left_reason(begin):
                full = surname + short
                if possible(full) and surname not in ambiguous_prefixes:
                    recovered[full].setdefault(begin, (begin, end, short))
    for full, occurrences in sorted(recovered.items()):
        rec = record(full)
        rec["recovered"] = list(occurrences.values())
        # Full names here are literal, boundary-supported occurrences in the book.
        accepted.add(full); reasons.setdefault(full, "N104")
    if recovered:
        for name, start, end in _matches(text, _trie(recovered)):
            if name not in counts:
                first_positions.setdefault(name, (start, end))
        rec_counts = Counter(name for name, _, _ in _matches(text, _trie(recovered)))
        counts.update({name: count - counts[name] for name, count in rec_counts.items()})

    # An already discovered two-character call name plus an explicit surname
    # clause supports a normalized surname+call-name form, even if that full
    # spelling does not occur verbatim. The bridge grammar is deliberately tight.
    if cfg["enabled"]:
        prefix_trie = _trie(name for name in accepted if len(name) == 2 and not surname_of(name) and name not in generic)
        surname_clause = re.compile(r"(?:[，,:：\s]|亦|原|本|其|有些|来历|原来|本来){0,12}姓(?P<surname>" + surname_pattern + ")")
        for short, start, end in _matches(text, prefix_trie):
            clause = surname_clause.match(text, end)
            if clause:
                full = clause["surname"] + short
                if 2 <= len(full) <= 4:
                    rec = record(full)
                    rec["constructed"] = True
                    rec["introductions"].append((start, clause.end(), "称呼与姓氏字段规范化"))
                    accepted.add(full); reasons[full] = "N103"
                    definition_edges.append({"a": full, "b": short, "rule": "N203",
                                             "evidence": example(start, clause.end(), "N203", "称呼与姓氏字段")})
        introduced = {name for name in records if name not in counts}
        for name, start, end in _matches(text, _trie(introduced)):
            counts[name] += 1
            first_positions.setdefault(name, (start, end))

    # Name classification is not enough to prove identity. Only accepted, unique
    # surname/given-name relations get edges. Competing full forms stay distinct.
    potential = defaultdict(set)
    for full in sorted(accepted):
        surname = surname_of(full)
        if surname and len(full) - len(surname) == 2:
            short = full[len(surname):]
            if short not in generic:
                potential[short].add(full)
    missing = set(potential) - set(records)
    for name in missing:
        record(name)
    for name, start, end in _matches(text, _trie(missing)):
        counts[name] += 1
        first_positions.setdefault(name, (start, end))
    claims = defaultdict(set)
    for edge in definition_edges:
        claims[edge["b"]].add(edge["a"])
    definition_conflicts = {short: fulls for short, fulls in claims.items() if len(fulls) > 1}
    edges = [edge for edge in definition_edges if edge["b"] not in definition_conflicts]
    ambiguities, decisions = [], []
    for short, owners in sorted(definition_conflicts.items()):
        ambiguities.append({"short": short, "full_candidates": sorted(owners), "raw_count": counts[short],
                            "independent_contexts": len(records[short]["contexts"]), "rule": "N202",
                            "status": "ambiguous", "reason": "原文把相同称呼用于多个不同姓名，未全局合并"})
    for short, fulls in sorted(potential.items()):
        owners = sorted(fulls)
        context_hits = len(records[short]["contexts"])
        decision = {"short": short, "full_candidates": owners, "raw_count": counts[short],
                    "independent_contexts": context_hits, "rule": "N201"}
        if context_hits and counts[short] > sum(counts[full] for full in owners):
            accepted.add(short); reasons.setdefault(short, "N107")
        if len(owners) > 1 or short in definition_conflicts:
            decision.update(status="ambiguous", rule="N202", reason="相同简称对应多个全名，未自动合并")
            ambiguities.append(decision.copy())
        elif context_hits and counts[short] > counts[owners[0]]:
            full = owners[0]
            accepted.add(short); reasons.setdefault(short, "N201")
            start, end, cue, _ = next(iter(records[short]["contexts"].values()))
            edges.append({"a": full, "b": short, "rule": "N201",
                          "evidence": example(start, end, "N201", cue)})
            decision.update(status="accepted", reason="简称有独立人物上下文，且本书只有一个候选全名")
        else:
            decision.update(status="insufficient", reason="缺少简称独立出现的人物上下文")
        decisions.append(decision)

    # Explicit text edges may connect more than two spellings of one person.
    parent = {name: name for name in accepted}
    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name
    for edge in edges:
        if edge["a"] in accepted and edge["b"] in accepted:
            a, b = find(edge["a"]), find(edge["b"])
            parent[max(a, b)] = min(a, b)
    components = defaultdict(list)
    for name in sorted(accepted):
        components[find(name)].append(name)
    groups = sorted(group for group in components.values() if len(group) > 1)

    output = []
    for number, (name, rec) in enumerate(sorted(records.items()), 1):
        samples = []
        for start, end, cue, _ in list(rec["contexts"].values())[:cfg["evidence_limit"]]:
            samples.append(example(start, end, "N101", cue))
        for start, end, trigger in rec["introductions"][:cfg["evidence_limit"]]:
            samples.append(example(start, end, "N103", trigger))
        if not samples and rec["recovered"]:
            start, end, short = rec["recovered"][0]
            samples.append(example(start, end, "N104", f"姓氏与已发现名称：{short}"))
        if not samples and name in first_positions:
            start, end = first_positions[name]
            samples.append(example(start, end, "N105" if rec["title"] else "N109", "原文出现"))
        output.append({"id": f"PERS{number:06d}", "name": name, "accepted": name in accepted,
                       "kind": "surname_title" if rec["title"] else "person_name",
                       "constructed": rec["constructed"],
                       "rule": reasons.get(name, "N109"), "raw_count": counts[name],
                       "independent_contexts": len(rec["contexts"]), "surname": surname_of(name),
                       "context_kinds": sorted({c[2] for c in rec["contexts"].values()}),
                       "examples": samples[:cfg["evidence_limit"]]})
    result = {"policy": PERSON_POLICY, "config": cfg, "rules": PERSON_RULES,
              "names": sorted(accepted), "candidates": output, "groups": groups,
              "relations": edges, "relation_decisions": decisions, "ambiguities": ambiguities,
              "counts": {"candidates": len(output), "accepted_names": len(accepted),
                         "automatic_groups": len(groups), "ambiguous_shorts": len(ambiguities)}}
    result["sha256"] = _hash(result)
    return result
