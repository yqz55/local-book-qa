import base64
import html
import math
from pathlib import Path

import streamlit as st

from evidence.retrieval import canonical
from evidence.dag_tokenization import DAG_POLICY
from evidence.tracing import deserialize, read_record, replay


def highlighted(paragraph, terms, expression_spans=()):
    ranges = []
    for token in paragraph["tokens"]:
        if token["term"] in terms:
            ranges.append((token["clean_start"] - paragraph["clean_start"], token["clean_end"] - paragraph["clean_start"]))
    ranges.extend((span["clean_start"] - paragraph["clean_start"], span["clean_end"] - paragraph["clean_start"])
                  for span in expression_spans)
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    pieces, cursor = [], 0
    for start, end in merged:
        pieces.extend([html.escape(paragraph["text"][cursor:start]), f"<mark>{html.escape(paragraph['text'][start:end])}</mark>"])
        cursor = end
    pieces.append(html.escape(paragraph["text"][cursor:]))
    return "".join(pieces)


def token_rows(tokens):
    rows = [{"词项": t["term"], "实际文字": t["surface"], "规则": t["rule"],
             "清洗后起点(0基)": t["clean_start"], "清洗后终点(排他)": t["clean_end"],
             "原文起点(0基)": t["source_start"], "原文终点(排他)": t["source_end"]} for t in tokens]
    sources = {"general_dictionary": "通用词典与整句路径", "book_dictionary": "本书专用词条",
               "person_discovery": "本书姓名规则", "alias_dictionary": "本书别名保护", "book_vocabulary": "本书词表", "saved_dictionary": "已保存词典",
               "han_character_fallback": "汉字单字回退", "ascii_run": "字母数字规则", "unicode_fallback": "Unicode 回退"}
    for row, token in zip(rows, tokens):
        if "book_frequency" in token:
            row["本书词频"] = token["book_frequency"]
        if "match_source" in token:
            row["匹配来源"] = sources.get(token["match_source"], token["match_source"])
        if "selection_rule" in token:
            row["选择规则"] = token["selection_rule"]
        if "adaptation_frequency" in token:
            row["初分词频（适配依据）"] = token["adaptation_frequency"]
        if "dictionary_frequency" in token:
            row["通用词典频次"] = token["dictionary_frequency"]
    priority = ["词项", "实际文字", "本书词频", "匹配来源", "规则", "选择规则"]
    return [{**{key: row[key] for key in priority if key in row},
             **{key: value for key, value in row.items() if key not in priority}} for row in rows]


def query_source_label(term):
    sources = term.get("sources", [])
    labels = ["问题原词" if source.get("kind") == "literal"
              else f"别名扩展：{source.get('from', '')}" if source.get("kind") == "alias"
              else f"词义扩展：{source.get('from', '')}" if source.get("kind") == "lexical"
              else "问题完整短语" if source.get("kind") == "literal_expression"
              else source.get("kind", "") for source in sources]
    return "；".join(dict.fromkeys(labels)) if labels else "问题原词"


def show_source(index, paragraph, key):
    p = paragraph
    st.caption(f"章节：{p['chapter']} · 切分：{p['split_rule']} / {p['split_reason']}")
    st.caption(f"原文区间 [{p['source_start']}, {p['source_end']})；清洗后区间 "
               f"[{p['clean_start']}, {p['clean_end']})。单位为 Unicode 字符，0 起点、终点不含。")
    st.code(index["source"]["text"][p["source_start"]:p["source_end"]], language=None, wrap_lines=True)
    st.caption("实际分词与原文映射；表中规则编号可在“文本与索引”查看。")
    st.dataframe(token_rows(p["tokens"]), hide_index=True, width="stretch", key=f"tokens_{key}")
    if p["skipped"]:
        st.caption("跳过的空白、标点或格式字符（不计入段长）")
        st.json(p["skipped"], expanded=False)


def show_evidence(index, result, trace_bytes, key_prefix=""):
    heading, export = st.columns([2, 1.4])
    heading.subheader("检索证据")
    if result is None:
        st.info("输入问题或关键词，查看最多 5 个相关段落。")
        return
    with export:
        st.download_button("导出复算记录", trace_bytes, file_name=f"trace-{index['index_id'][:12]}.json.gz",
                           mime="application/gzip", width="stretch", key=f"{key_prefix}export_trace")
    st.caption("以下为段落检索依据，供你核对句子答案的上下文。分数表示词项匹配程度。")
    st.caption(f"本次问题：{result['query']} · 候选 {result['candidate_count']} 段 · "
               f"k1 = {result['config']['k1']}，b = {result['config']['b']} · N = {result['corpus']['N']}，"
               f"avgdl = {result['corpus']['avgdl']:.6f} 词项")
    if "query_expansion_policy" in result:
        enabled = result["query_expansion_policy"]["config"]["enabled"]
        lexical = result.get("lexical_expansions", [])
        st.caption(f"通用词义扩展：{'已启用' if enabled else '已关闭'} · 本次采用 {len(lexical)} 条扩展；来源与筛选依据见下方查询明细。")
    for ambiguity in result.get("query_ambiguities", []):
        st.warning(f"称呼“{ambiguity['short']}”存在歧义：{'、'.join(ambiguity['full_candidates'])}。"
                   "未自动合并这些人物；可用完整姓名检索。")
    query_sources = {term["term"]: query_source_label(term) for term in result["query_terms"]}
    with st.expander("查看查询分词与检索规则"):
        policy = result.get("query_tokenization")
        if policy:
            if policy["policy"] == DAG_POLICY:
                st.write("使用完整通用词典构建候选词图，按整句最优路径分词，HMM 关闭；姓名与专用词优先完整匹配。正文与查询使用同一份冻结配置。")
                st.caption("本书词频为正式索引 TF；初分词频来自导入时第一遍分词，仅用于有限的路径奖励。提问不会改变这些统计。")
            elif "query_name_decisions" in result:
                st.write("先完整匹配本书规则发现的姓名，再按已接受的同一人物关系扩展；其他多字词仍优先本书词频。")
            elif "query_expansions" in result:
                st.write("先保护本书别名词条的完整切分，再按本书词频选择其他多字词；同频选长词，单字只作回退。")
            else:
                st.write("当前位置优先匹配本书高频多字词；同频选长词，单字只作回退。")
            st.caption(f"查询规则：{policy['policy']} · 本书词表版本：{policy['book_vocabulary_sha256'][:16]}")
        st.caption("下表只列问题中实际出现的文字及位置；人物和词义扩展单独列出。")
        st.dataframe(token_rows(result["query_tokens"]), hide_index=True, width="stretch")
        if policy and policy["policy"] == DAG_POLICY:
            choices = [{"问题起点(0基)": t["source_start"], "候选词": c["term"],
                        "词典频次": c["dictionary_frequency"], "初分词频": c["seed_tf"],
                        "基础对数分": c["base_score"], "本书奖励": c["book_bonus"],
                        "后续最优分": c["suffix_score"], "总路径分": c["path_score"],
                        "被选中": c["term"] == t["term"]}
                       for t in result["query_tokens"] for c in t.get("match_candidates", [])]
        else:
            choices = [{"问题起点(0基)": t["source_start"], "候选词": c["term"],
                    "本书词频": c["book_tf"], "词长": c["length"], "被选中": c["term"] == t["term"]}
                   for t in result["query_tokens"] for c in t.get("match_candidates", [])]
        if choices:
            st.caption("实际比较的同起点候选（按选择顺序排列）")
            st.dataframe(choices, hide_index=True, width="stretch")
        if "query_segmentation_trace" in result:
            with st.expander("完整 DAG 候选与路径计算"):
                st.code("边分 = ln(max(词典频次, 1) / 词典总频次) + 本书奖励\n"
                        "本书奖励 = 上限 × ln(1 + 初分词频) / ln(1 + 全书最高初分词频)\n"
                        "当前位置最优分 = max(边分 + 词尾之后的最优分)\n"
                        "先比较完整路径，路径同分时当前位置长词优先。", language=None, wrap_lines=True)
                st.caption("未登录单字的原词典频次记为 0，计算路径时按频次 1 回退；这与词典中正常存在的单字词分开记录。")
                st.json(result["query_segmentation_trace"], expanded=False)
        if result.get("query_name_decisions"):
            st.caption("本次姓名整词保护的发现依据；候选与完整规则可在“文本与索引”查看。")
            st.json(result["query_name_decisions"], expanded=False)
        expansions = result.get("query_expansions", [])
        if expansions:
            st.caption("已接受的同一人物关系双向匹配。原词与扩展词取并集，每个不同词项只计分一次。")
            st.dataframe([{"问题原词": e["source_term"], "扩展词": e["term"],
                           "同组名称": " = ".join(e["group"]), "组编号": e["group_id"],
                           "规则": e["rule"], "计分权重": e["weight"],
                           "关系来源": {"automatic": "本书规则推断", "manual": "人工校正"}.get(e.get("origin"), "旧版显式组"),
                           "处理": "加入检索" if e["added_to_query"] else "已存在，合并计分"}
                          for e in expansions], hide_index=True, width="stretch")
            if any(e.get("relations") for e in expansions):
                with st.expander("查看关系判断与原文证据"):
                    st.json([{"原词": e["source_term"], "扩展词": e["term"], "判断": e.get("relations", [])}
                             for e in expansions], expanded=False)
        if "query_expansion_policy" in result:
            st.markdown("**通用词义与短语扩展**")
            lexical = result.get("lexical_expansions", [])
            st.write("按通用词义组查询候选，再过滤本书未出现的表达；扩展一层。同组只取最大计分贡献。多义候选尚未完成上下文消歧，命中也不表示赠送等事实已经成立。")
            if lexical:
                st.dataframe([{"问题表达": e["source_term"], "扩展表达": e["term"],
                    "来源": {"wordnet": "中文 WordNet", "reviewed": "项目通用校正", "manual": "本书人工校正"}[e["origin"]],
                    "词义编号": "、".join(e["sense_ids"]), "本书 DF": e["book_df"], "权重": e["weight"],
                    "问题原文起点(0基)": e["query_source_start"], "问题原文终点(排他)": e["query_source_end"]}
                    for e in lexical], hide_index=True, width="stretch")
            else:
                st.caption("本次没有采用词义扩展；可能未启用、未覆盖、被停用或候选未在本书完整出现。")
            with st.expander("查看全部词义候选与筛选结果"):
                st.json({"配置与版本": result["query_expansion_policy"], "实际判断": result.get("lexical_decisions", []),
                         "规则": index.get("query_expansion", {}).get("rules", {})}, expanded=False)
        st.caption("实际参与检索的去重特征；问题出现次数为 0 的表达来自扩展。短语 TF 来自独立连续区间，不改变正文分词与段长。")
        st.dataframe([{"词项": t["term"], "来源": query_sources[t["term"]],
                       "问题出现次数": t["query_frequency"], "计分权重": t["weight"],
                       "匹配类型": "连续表达" if t.get("kind") == "expression" else "正文词项",
                       "计分组": t.get("score_group", "独立"),
                       "DF": t["df"], "IDF": t["idf"], "词库中存在": t["in_vocabulary"]}
                      for t in result["query_terms"]], hide_index=True, width="stretch")
        if policy:
            for explanation in policy["rules"].values():
                st.write(explanation)
        else:
            st.write(index["rules"]["Q001"])
        st.write(result["candidate_policy"])
        st.code("idf = ln(1 + (N - df + 0.5) / (df + 0.5))\n"
                "L = 1 - b + b × dl / avgdl\n"
                "贡献 = weight × idf × tf × (k1 + 1) / (tf + k1 × L)\n"
                "词义组贡献 = 组内最大加权贡献；完整短语覆盖的查询碎片不另计分\n"
                "总分 = 实际计分贡献之和", language=None)
        st.caption("同分时按原文起点、段落编号升序。无命中词的段落零分且不返回。显示值经四舍五入，导出保留完整精度。")
        if result["query_skipped"]:
            st.json({"query_cleaning": result["query_cleaning"], "skipped": result["query_skipped"]}, expanded=False)
    if result["status"] == "empty_query":
        st.info("问题没有可检索词项。请输入文字、数字或有意义的符号。")
        return
    if result["status"] == "no_matches":
        st.info("没有匹配到词项。可尝试原文中的关键词；这不代表书中一定没有相关内容。")
        return
    if result["status"] == "no_content_query":
        st.info("剔除提问表达后缺少可定位的内容，请补充对象或事件。")
        return
    paragraphs = {p["id"]: p for p in index["paragraphs"]}
    terms = {t["term"] for t in result["query_terms"] if not t.get("covered_by_expression") and t.get("kind", "token") == "token"}
    for candidate in result["candidates"][:5]:
        p = paragraphs[candidate["paragraph_id"]]
        st.divider()
        st.markdown(f'<div class="evidence-heading"><strong>{candidate["rank"]:02d} · {p["id"]}</strong>'
                    f'<span class="score">BM25 <b>{candidate["score"]:.6f}</b></span></div>', unsafe_allow_html=True)
        loc = p["location"]
        st.markdown(f'<div class="source-label">第 {loc["start_line"]} 行，第 {loc["start_column"]} 列起 · '
                    f'原文字符 {p["source_start"] + 1}–{p["source_end"]} · 段长 {p["length"]} 词项</div>', unsafe_allow_html=True)
        expression_spans = [span for detail in candidate["term_details"] for span in detail.get("occurrences", [])]
        st.markdown(f'<div class="evidence-text">{highlighted(p, terms, expression_spans)}</div>', unsafe_allow_html=True)
        with st.expander(f"查看原文与位置 · {p['id']}"):
            show_source(index, p, f"{key_prefix}result_{p['id']}")
        with st.expander(f"展开计算明细 · {p['id']}"):
            st.dataframe([{"检索词项": t["term"], "来源": query_sources[t["term"]],
                           "匹配": "是" if t["matched"] else "否",
                           "TF": t["tf"], "DF": t["df"], "权重": t["weight"],
                           "IDF": t["idf"], "组内比较前贡献": t.get("raw_contribution", t["contribution"]),
                           "计分依据": {"group_maximum": "组内最大", "same_group_not_selected": "同组其他表达胜出", "covered_by_expression": "由完整短语计分"}.get(t.get("score_reason"), "独立词项"),
                           "分数贡献": t["contribution"]}
                          for t in candidate["term_details"]], hide_index=True, width="stretch",
                         column_config={"IDF": st.column_config.NumberColumn(format="%.10f"),
                                        "分数贡献": st.column_config.NumberColumn(format="%.10f")})
            st.code(f"dl = {candidate['dl']}\navgdl = {result['corpus']['avgdl']!r}\n"
                    f"L = 1 − {result['config']['b']} + {result['config']['b']} × dl / avgdl = {candidate['length_normalization']!r}\n"
                    f"词项贡献之和 = {candidate['score']!r}\n最终排名 = {candidate['rank']}", language=None, wrap_lines=True)
            with st.expander("完整精度与实际分子、分母"):
                st.json(candidate)
    with st.expander(f"全部候选排名（{result['candidate_count']} 段）"):
        page = st.number_input("候选页码（每页 100 段）", 1, max(1, math.ceil(result["candidate_count"] / 100)), key=f"{key_prefix}candidate_page")
        st.dataframe([{"排名": c["rank"], "编号": c["paragraph_id"], "总分": c["score"], "段长": c["dl"],
                       "原文起点(0基)": c["source_start"]} for c in result["candidates"][(page - 1) * 100:page * 100]],
                     hide_index=True, width="stretch")


def show_name_discovery(index):
    discovery = index.get("name_discovery")
    if not discovery:
        return
    counts = discovery["counts"]
    with st.expander("本书姓名与称呼（自动发现）"):
        st.caption(f"从全文发现 {counts['accepted_names']} 个姓名或人物称呼；提出 {counts['automatic_groups']} 组同一人物关系，"
                   f"保留 {counts['ambiguous_shorts']} 个歧义称呼。默认没有预填人名或别名组。")
        st.write("先用姓氏结构、说话动作、人物介绍和称谓上下文发现候选，再检查边界、普通词和独立出现证据。"
                 "只有简称对应唯一全名或原文明确介绍同一人物时才关联。规则推断可能漏收或误收，所有依据均保存在下方。")
        needle = st.text_input("筛选姓名或称呼", key="person_filter")
        state = st.selectbox("候选状态", ["已接受", "未接受", "全部"], key="person_status")
        matches = [r for r in discovery["candidates"] if needle in r["name"]
                   and (state == "全部" or r["accepted"] == (state == "已接受"))]
        st.caption(f"符合筛选 {len(matches)} 个；“原文出现数”含子串，仅用于发现规则，不等于 BM25 分词词频。")
        last_page = max(1, math.ceil(len(matches) / 50))
        if st.session_state.get("person_page", 1) > last_page:
            st.session_state.person_page = 1
        page = st.number_input("姓名候选页码（每页 50 个）", 1, last_page, key="person_page")
        visible = matches[(page - 1) * 50:page * 50]
        st.dataframe([{"名称": r["name"], "接受": r["accepted"], "规则": r["rule"],
                       "原文出现数": r["raw_count"], "独立人物上下文": r["independent_contexts"],
                       "类型": "姓氏称谓" if r["kind"] == "surname_title" else "姓名候选",
                       "由介绍字段构造": r["constructed"]} for r in visible], hide_index=True, width="stretch")
        if visible:
            selected = st.selectbox("查看候选判断与原文位置", [r["name"] for r in visible], key="person_detail")
            rec = next(r for r in visible if r["name"] == selected)
            st.write(f"{rec['rule']}：{discovery['rules'][rec['rule']]}")
            for example in rec["examples"]:
                st.caption(f"原文区间 [{example['source_start']}, {example['source_end']}) · 触发：{example['trigger']}")
                st.code(example["context"], language=None, wrap_lines=True)
        st.markdown("**自动提出的同一人物关系**")
        st.dataframe([{"名称组": " = ".join(group)} for group in discovery["groups"]], hide_index=True, width="stretch")
        if discovery["ambiguities"]:
            st.warning("以下简称对应多个全名，未自动合并。")
            st.dataframe([{"简称": a["short"], "候选全名": "、".join(a["full_candidates"]),
                           "规则": a["rule"], "原因": a["reason"]} for a in discovery["ambiguities"]],
                         hide_index=True, width="stretch")
        with st.expander("完整关系判定、规则与参数"):
            st.json({key: discovery[key] for key in ("policy", "config", "rules", "relations", "relation_decisions", "sha256")}, expanded=False)
        st.download_button("导出姓名发现与关系证据", canonical(discovery), file_name="book-person-discovery.json",
                           mime="application/json", key="download_person_discovery")


def show_index(index, on_book_words_submit, on_aliases_submit, on_expansion_submit=None):
    st.subheader("文本与索引")
    stats, source = index["statistics"], index["source"]
    st.caption(f"{source['filename']} · {source['encoding']} · {source['byte_length']:,} 字节 · "
               f"{stats['paragraph_count']:,} 个保留段落，其中 {stats['N']:,} 段参与检索，"
               f"{stats['excluded_empty_token_paragraphs']} 段无词项。")
    st.code(f"文本 SHA-256: {source['text_sha256']}\n字节 SHA-256: {index['identity']['bytes_sha256']}\n"
            f"索引版本: {index['index_id']}\n词典版本: {index['dictionary']['sha256']}", language=None, wrap_lines=True)
    st.download_button("下载原始 TXT", base64.b64decode(source["bytes_base64"]), file_name=source["filename"],
                       mime="application/octet-stream", key="download_original")
    show_name_discovery(index)
    if "segmentation" in index:
        segmentation = index["segmentation"]
        with st.expander("通用词典与整句分词配置"):
            st.caption(f"通用词典 {len(segmentation['lexicon']['frequencies']):,} 个汉字词条（包含正常单字词）。"
                       "一遍初分统计书内词频，再按固定奖励分词并建立正式索引；查询复用相同配置，HMM 关闭。")
            st.json({"词典来源": segmentation["lexicon"]["source"], "词典版本": segmentation["lexicon"]["sha256"],
                     "配置": segmentation["config"], "规则": segmentation["rules"],
                     "初分词表版本": segmentation["seed_vocabulary_sha256"]}, expanded=False)
    error = st.session_state.get("book_words_error")
    error = error if error and error["index_id"] == index["index_id"] else None
    with st.expander("本书专用词条（补充后重建）", expanded=error is not None):
        dictionary = index["dictionary"]
        existing = dictionary.get("book_words", [])
        st.caption("每行一个 2–32 字汉字词，例如人名或专业术语。词条与基础词典合并后，重新切分全文并统计本书词频。")
        st.caption("删除一行即可移除该专用词条；基础词典保留。新索引单独保存，旧索引与旧记录仍可使用。")
        editor_key = f"book_words_{index['index_id']}"
        with st.form(f"book_words_form_{index['index_id']}"):
            st.text_area("本书专用词条（每行一个）", value="\n".join(existing), key=editor_key, height=140)
            st.form_submit_button("应用词条并重建索引", type="primary", on_click=on_book_words_submit,
                                  args=(index["index_id"], editor_key))
        if error:
            st.error(error["message"])
        st.download_button("导出本书专用词条", canonical(existing), file_name="book-dictionary.json",
                           mime="application/json", key="download_book_words")
    aliases = index.get("aliases", {})
    alias_groups = aliases.get("manual_groups", [])
    alias_error = st.session_state.get("book_aliases_error")
    alias_error = alias_error if alias_error and alias_error["index_id"] == index["index_id"] else None
    with st.expander("人工关系校正（可选）", expanded=alias_error is not None):
        st.caption("姓名与称呼优先由全文规则发现。此处默认空白；只在需要人工修正时，每行一组，用 = 分隔同一对象的名称。")
        st.caption("同组名称双向扩展一层。每个名称都会加入正文词典，并在问题中优先完整匹配；不同检索词项各计分一次，重复扩展不重复计分。")
        st.caption("人工组覆盖与其名称相交的自动关联，避免意外传递合并。删除人工组后恢复规则推断；旧索引与旧记录保留。")
        alias_editor_key = f"book_aliases_{index['index_id']}"
        with st.form(f"book_aliases_form_{index['index_id']}"):
            st.text_area("本书别名（每行一组，用 = 分隔）",
                         value="\n".join(" = ".join(group) for group in alias_groups),
                         key=alias_editor_key, height=140)
            st.form_submit_button("应用别名并重建索引", type="primary", on_click=on_aliases_submit,
                                  args=(index["index_id"], alias_editor_key))
        if alias_error:
            st.error(alias_error["message"])
        st.caption(f"本书别名版本：{aliases.get('sha256', '旧索引未保存别名')}")
        st.json({key: aliases.get(key) for key in ("manual_groups", "automatic_groups", "suppressed_automatic_groups", "rules")}, expanded=False)
        st.download_button("导出本书别名", canonical(alias_groups), file_name="book-aliases.json",
                           mime="application/json", key="download_book_aliases")
    expansion = index.get("query_expansion")
    if expansion and on_expansion_submit:
        expansion_error = st.session_state.get("book_expansion_error")
        expansion_error = expansion_error if expansion_error and expansion_error["index_id"] == index["index_id"] else None
        with st.expander("通用词义扩展与本书校正", expanded=expansion_error is not None):
            resource, cfg = expansion["resource"], expansion["config"]
            st.caption(f"随附中文 WordNet 共 {len(resource['groups']):,} 个词义组；本书匹配到 {len(expansion['book_matches']):,} 个表达。运行时不联网。")
            st.write("通用资源自动提供候选，无需为每本书逐词建立同义表。校正只影响当前书的查询扩展，正文分词、TF 和段长保持原样。")
            prefix = f"expansion_{index['index_id']}"
            with st.form(f"{prefix}_form"):
                st.checkbox("启用通用词义扩展", value=cfg["enabled"], key=f"{prefix}_enabled")
                st.number_input("扩展表达权重", 0.05, 1.0, float(cfg["weight"]), step=0.05, key=f"{prefix}_weight")
                st.number_input("每个问题表达最多扩展数", 1, 10, cfg["max_alternatives"], key=f"{prefix}_limit")
                st.text_area("本书表达校正（每行一组，用 = 分隔）", value="\n".join(" = ".join(g) for g in expansion["manual_groups"]), key=f"{prefix}_groups")
                st.text_area("本书停用扩展表达（每行一个）", value="\n".join(expansion["blocked"]), key=f"{prefix}_blocked")
                st.form_submit_button("应用表达扩展设置", on_click=on_expansion_submit, args=(index["index_id"], prefix))
            if expansion_error:
                st.error(expansion_error["message"])
            st.caption("同组表达应在所需语境中可替换。人工组暂停相交的通用组；停用表达既不触发展开，也不作为新增候选。清空校正可恢复通用资源。")
            with st.expander("词义资源来源、版本与规则"):
                st.json({"来源": resource["source"], "资源版本": resource["sha256"], "本书版本": expansion["sha256"], "规则": expansion["rules"]}, expanded=False)
            st.download_button("导出本书表达校正", canonical({"config": cfg, "groups": expansion["manual_groups"], "blocked": expansion["blocked"]}),
                               file_name="book-query-expansion.json", mime="application/json", key="download_query_expansion")
    with st.expander("清洗、切分与分词规则"):
        st.json({"config": index["identity"]["ingest_config"], "rules": index["rules"]})
        st.caption("以上为索引生成时的规则快照；当前问题的实际选择规则见“检索证据”中的查询分词明细。")
        st.caption("清洗变更按原文位置记录；全部变更与压缩区间映射包含在导出记录。")
        page = st.number_input("清洗记录页码（每页 100 条）", 1, max(1, math.ceil(len(index["cleaning"]["edits"]) / 100)), key="clean_page")
        st.json(index["cleaning"]["edits"][(page - 1) * 100:page * 100], expanded=False)
    st.markdown("### 段落与原文")
    number = st.number_input("段落序号", 1, len(index["paragraphs"]), key="paragraph_number")
    p = index["paragraphs"][number - 1]
    st.caption(f"{p['id']} · {'参与检索' if p['indexed'] else '无词项，保留但不参与检索'} · 段长 {p['length']} 词项")
    show_source(index, p, "index")
    st.markdown("### 词频与文档频率")
    term_filter = st.text_input("筛选词项", key="term_filter")
    terms = sorted((t for t in stats["collection_tf"] if term_filter in t),
                   key=lambda t: (-stats["collection_tf"][t], -len(t), t))
    st.caption("按正式索引词频降序展示；正文与查询共用通用词典、姓名与专用词保护及冻结的整句路径评分。")
    page = st.number_input("词项页码（每页 100 个）", 1, max(1, math.ceil(len(terms) / 100)), key="term_page")
    st.dataframe([{"词项": t, "全书词频": stats["collection_tf"][t], "文档频率 DF": stats["document_frequency"][t]}
                  for t in terms[(page - 1) * 100:page * 100]], hide_index=True, width="stretch")
    with st.expander(f"查看实际词典（{len(index['dictionary']['words'])} 个词条）"):
        st.caption("这里分页展示多字词条。通用词典还包含正常单字词，其频次可在查询明细及导出记录中查看。")
        dictionary_filter = st.text_input("筛选词典词条", key="dictionary_filter")
        dictionary_words = [word for word in index["dictionary"]["words"] if dictionary_filter in word]
        last_page = max(1, math.ceil(len(dictionary_words) / 100))
        if st.session_state.get("dictionary_page", 1) > last_page:
            st.session_state.dictionary_page = 1
        dictionary_page = st.number_input("词典页码（每页 100 个）", 1, last_page, key="dictionary_page")
        st.caption(f"符合筛选 {len(dictionary_words):,} 个；完整词典与频次随复算记录导出。")
        st.json(dictionary_words[(dictionary_page - 1) * 100:dictionary_page * 100])


def show_replay(store):
    st.subheader("记录复算")
    st.caption("导出的记录自带原始字节、词典、索引与全部候选计算。复算会从头重建索引，再比较分数和排名。")
    uploaded = st.file_uploader("导入复算记录（JSON / JSON.GZ）", type=["json", "gz"], key="replay_upload")
    history = store.list_traces()
    labels = {h["path"]: f"{h['saved_at'][:19]} UTC · {h.get('engine_version', '历史版本')} · {h['query'][:40]} · {h['filename']}" for h in history[:50]}
    selected = st.selectbox("本地历史记录（最近 50 次）", [None, *labels],
                            format_func=lambda v: "请选择记录" if v is None else labels[v], key="history_select")
    if selected:
        st.download_button("下载所选历史记录", Path(selected).read_bytes(), file_name="history-trace.json.gz",
                           mime="application/gzip", key="download_history")
    if st.button("开始独立复算", disabled=not (uploaded or selected), key="replay_button"):
        try:
            with st.spinner("重新清洗、分词、建立索引并计算全部候选…"):
                trace = deserialize(uploaded.getvalue()) if uploaded else read_record(selected)
                report = replay(trace)
            st.success("复算通过：全部候选、计算贡献及最终排名一致。")
            st.write(f"问题：{trace['retrieval']['query']}")
            st.json(report)
        except (ValueError, OSError, KeyError, TypeError, OverflowError, EOFError) as exc:
            st.error(f"复算失败：{exc}")
