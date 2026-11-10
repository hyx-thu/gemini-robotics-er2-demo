"""Scenario Persona & Hybrid RAG Knowledge Base for Embodied Service Robots.

Provides selectable industry scenarios (e.g. Cymbal Hospital outpatient triage
concierge, Cymbal Bank flagship branch lobby manager), scenario-specific
navigation waypoints, concise persona prompt blocks, and a fast hybrid RAG
retriever (`search_knowledge_base`) combining:
  1. Field-weighted BM25 over English words, CJK unigrams/bigrams, and
     bilingual colloquial query-expansion synonyms (<1 ms, zero network deps).
  2. Optional semantic vector similarity (`text-embedding-005` on Vertex AI
     `us-central1` with in-memory chunk caching and 1.0s timeout guard), fused
     with BM25 via Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
import time
from typing import Any
import urllib.request

from model import vertex_rest

logger = logging.getLogger(__name__)

SCENARIO_RAG_TOOL_NAME = "search_knowledge_base"
DEFAULT_SCENARIO_ID = "default"

_KNOWLEDGE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "knowledge",
    "scenarios.json",
)

# Colloquial bilingual synonym clusters so natural visitor phrasing ("交钱",
# "验血", "存钱利息", "买房流水", "打钱出国") hits exact policy chunks even
# before vector reranking.
_SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("缴费", "交费", "交钱", "付钱", "付款", "买单", "收费", "结算", "刷医保", "医保", "报销", "发票", "收据", "水费", "电费", "燃气费", "煤气", "学费", "pay", "payment", "billing", "cashier", "invoice", "utilities"),
    ("抽血", "采血", "验血", "查血", "化验", "血常规", "空腹", "肝功", "肾功", "血糖", "血脂", "blood", "phlebotomy", "fasting", "lab"),
    ("尿检", "验尿", "尿常规", "大便", "便常规", "留样", "标本", "urine", "stool", "specimen"),
    ("理财", "买理财", "收益", "收益率", "年化", "活钱", "稳健", "固收", "基金", "风险测评", "风险评估", "天天盈", "wealth", "investment", "yield", "fund", "r1", "r2", "r3"),
    ("存款", "存钱", "定期", "定存", "利息", "利率", "大额存单", "国债", "保本", "养老金", "黄金", "积存金", "deposit", "interest", "bond", "gold", "cd"),
    ("取钱", "取现", "取款", "拿现金", "大额取现", "预约取款", "零钱", "零钱包", "新钞", "残币", "atm", "cash", "withdrawal", "change"),
    ("办卡", "开卡", "开户", "储蓄卡", "借记卡", "一类卡", "二类卡", "工资卡", "社保卡", "debit", "account"),
    ("密码", "忘密码", "锁住", "挂失", "补卡", "换卡", "手机号", "限额", "提额", "password", "lost", "limit"),
    ("外汇", "换汇", "购汇", "结汇", "美元", "日元", "欧元", "港币", "跨境", "电汇", "汇款", "打钱", "留学", "swift", "forex", "wire", "currency", "usd"),
    ("贷款", "借钱", "房贷", "按揭", "买房", "提前还款", "提前还贷", "消费贷", "经营贷", "抵押", "mortgage", "loan", "prepayment"),
    ("流水", "打流水", "存款证明", "资信证明", "资产证明", "征信", "信用报告", "签证", "statement", "credit", "certificate", "visa"),
    ("对公", "公司", "企业", "基本户", "营业执照", "公章", "回单", "u盾", "corporate", "business"),
    ("贵宾", "金卡", "金葵馨", "私人银行", "优先叫号", "vip", "private"),
    ("冻结", "止付", "非柜面", "解冻", "解控", "诈骗", "反诈", "frozen", "scam", "fraud"),
    ("老人", "长者", "卧床", "上门", "已故", "去世", "遗产", "轮椅", "无障碍", "elderly", "senior", "wheelchair"),
    ("挂号", "预约", "加号", "放号", "退号", "签到", "报到", "過號", "过号", "register", "appointment"),
    ("几楼", "在哪", "楼层", "怎么走", "科室", "导航", "floor", "where", "directory", "department"),
    ("挂什么科", "看什么科", "分诊", "头晕", "头痛", "胸闷", "胃痛", "咳嗽", "发烧", "发热", "腰痛", "扭伤", "triage", "symptom", "fever"),
    ("取药", "拿药", "药房", "中药", "代煎", "快递", "处方", "pharmacy", "medicine", "prescription"),
    ("核磁", "核磁共振", "拍片", "胸片", "彩超", "b超", "超声", "心电图", "憋尿", "金属", "mri", "ct", "xray", "ultrasound"),
    ("胃镜", "肠镜", "胃肠镜", "无痛", "内镜", "清肠", "endoscopy", "colonoscopy"),
    ("专家", "主任", "排班", "出诊", "坐诊", "几点开门", "营业时间", "周末", "挂号费", "特需", "schedule", "specialist", "hours", "weekend"),
    ("住院", "入院", "出院", "押金", "病房", "探视", "陪护", "inpatient", "ward", "admission"),
    ("体检", "入职体检", "驾照体检", "健康证", "早癌", "checkup", "physical"),
    ("病历", "病假条", "请假条", "诊断证明", "盖章", "理赔", "record", "stamp", "certificate"),
    ("急诊", "急救", "狗咬", "猫抓", "狂犬", "破伤风", "24小时", "emergency", "rabies", "tetanus"),
    ("停车", "停车费", "车库", "母婴室", "餐厅", "食堂", "咖啡", "充电宝", "parking", "cafe"),
)

_TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]+", re.IGNORECASE)
_CJK_RUN_RE = re.compile(r"^[\u4e00-\u9fff]+$")

_scenarios_cache: dict[str, dict[str, Any]] | None = None
_embed_lock = threading.Lock()
_embed_cache: dict[str, list[float]] = {}
_embed_disabled: bool = False


def _load_scenarios_map() -> dict[str, dict[str, Any]]:
  global _scenarios_cache
  if _scenarios_cache is not None:
    return _scenarios_cache
  with open(_KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
    raw = json.load(f)
  out: dict[str, dict[str, Any]] = {}
  for item in raw.get("scenarios", []):
    sid = str(item.get("id") or "").strip()
    if sid:
      out[sid] = item
  _scenarios_cache = out
  return out


def is_valid_scenario(scenario_id: str | None) -> bool:
  if not scenario_id:
    return True
  return scenario_id.strip() in _load_scenarios_map()


def normalize_scenario_id(scenario_id: str | None) -> str:
  sid = (scenario_id or "").strip()
  if not sid or sid not in _load_scenarios_map():
    return DEFAULT_SCENARIO_ID
  return sid


def is_rag_active(scenario_id: str | None, custom_kb: str = "") -> bool:
  sid = normalize_scenario_id(scenario_id)
  if sid != DEFAULT_SCENARIO_ID:
    return True
  return bool((custom_kb or "").strip())


def parse_custom_kb(custom_kb: str, scenario_id: str = "") -> list[dict[str, Any]]:
  """Parse user-entered Markdown / text snippets into searchable RAG chunks."""
  raw = (custom_kb or "").strip()
  if not raw:
    return []
  # Split by ### headings if present, otherwise by blank-line blocks.
  if re.search(r"(?m)^#{1,3}\s+\S+", raw):
    blocks = re.split(r"(?m)(?=^#{1,3}\s+\S+)", raw)
  else:
    blocks = re.split(r"\n\s*\n+", raw)

  chunks: list[dict[str, Any]] = []
  for idx, blk in enumerate(blocks):
    blk = blk.strip()
    if not blk:
      continue
    lines = [ln.strip() for ln in blk.splitlines() if ln.strip()]
    if not lines:
      continue
    first = re.sub(r"^#{1,3}\s*", "", lines[0]).strip()
    if len(lines) > 1:
      title = first[:72]
      body = "\n".join(lines[1:]).strip()
    else:
      # Single-line "Title: details" or plain note
      sep = re.split(r"[:：]\s*", first, maxsplit=1)
      if len(sep) == 2 and len(sep[0]) <= 36 and sep[1]:
        title, body = sep[0].strip(), sep[1].strip()
      else:
        title = first[:42] + ("…" if len(first) > 42 else "")
        body = first
    summary = body.replace("\n", " ")
    if len(summary) > 180:
      summary = summary[:179].rstrip() + "…"
    chunks.append({
        "id": f"custom_{scenario_id or 'kb'}_{idx + 1}",
        "category": "自定义补充知识",
        "title": title,
        "location": "",
        "waypoint": "",
        "keywords": [title],
        "summary": summary,
        "content": body,
        "is_custom": True,
    })
  return chunks


def get_scenario(
    scenario_id: str | None,
    custom_kb: str = "",
    custom_persona: str = "",
) -> dict[str, Any]:
  """Return resolved scenario metadata + merged built-in & custom documents."""
  smap = _load_scenarios_map()
  sid = normalize_scenario_id(scenario_id)
  base = dict(smap[sid])
  docs = list(base.get("documents") or [])
  extra = parse_custom_kb(custom_kb, sid)
  if extra:
    docs = extra + docs
  base["documents"] = docs
  if (custom_persona or "").strip():
    base["custom_persona"] = custom_persona.strip()
  return base


def get_scenario_waypoints(scenario_id: str | None) -> list[str] | None:
  sid = normalize_scenario_id(scenario_id)
  if sid == DEFAULT_SCENARIO_ID:
    return None
  smap = _load_scenarios_map()
  wps = smap.get(sid, {}).get("waypoints")
  return list(wps) if wps else None


def list_scenarios() -> list[dict[str, Any]]:
  """Return all scenarios with lightweight document catalogs for the UI."""
  smap = _load_scenarios_map()
  out = []
  for sid, sc in smap.items():
    docs = [
        {
            "id": d.get("id", ""),
            "category": d.get("category", ""),
            "title": d.get("title", ""),
            "location": d.get("location", ""),
            "waypoint": d.get("waypoint", ""),
            "summary": d.get("summary", ""),
        }
        for d in (sc.get("documents") or [])
    ]
    out.append({
        "id": sid,
        "name": sc.get("name", sid),
        "label_zh": sc.get("label_zh", sc.get("name", sid)),
        "badge": sc.get("badge", ""),
        "organization": sc.get("organization", ""),
        "role": sc.get("role", ""),
        "persona_prompt": sc.get("persona_prompt", ""),
        "waypoints": list(sc.get("waypoints") or []),
        "doc_count": len(docs),
        "documents": docs,
    })
  return out


def build_scenario_persona_block(
    scenario_id: str | None,
    custom_persona: str = "",
    custom_kb: str = "",
) -> str:
  """Build the concise developer-instruction persona block when active."""
  sid = normalize_scenario_id(scenario_id)
  cp = (custom_persona or "").strip()
  has_custom_kb = bool((custom_kb or "").strip())
  if sid == DEFAULT_SCENARIO_ID and not cp and not has_custom_kb:
    return ""

  sc = _load_scenarios_map().get(sid, {})
  org = sc.get("organization") or "自定义业务场景"
  role = sc.get("role") or "智能场景服务机器人"
  persona_text = cp if cp else (sc.get("persona_prompt") or "").strip()
  wps = sc.get("waypoints") or []
  wp_line = (
      f"- 当前场景专属导航路点（`get_waypoints` / `navigate`）：{', '.join(f'`{w}`' for w in wps)}。"
      if sid != DEFAULT_SCENARIO_ID and wps
      else ""
  )
  parts = [
      f"## 当前激活业务场景与角色人设：{org} — {role}",
  ]
  if persona_text:
    parts.append(persona_text)
  elif has_custom_kb:
    parts.append(
        "当访客询问任何场馆规定、办理流程或业务信息时，必须先调用"
        " `search_knowledge_base` 检索知识库，并严格基于检索结果作答。"
    )
  if wp_line:
    parts.append(wp_line)
  parts.append(
      "- 回答 `search_knowledge_base` 知识问答时无需调用 `update_plan`"
      "（`update_plan` 仅用于机械臂物理抓取/放置等多步操作任务）；"
      "收到 `[HEARTBEAT]` 心跳时严禁调用 `search_knowledge_base`。"
  )
  return "\n".join(parts)


def search_knowledge_base_tool(scenario_id: str | None = None) -> dict[str, Any]:
  """Gemini FunctionDeclaration for `search_knowledge_base` (strictly isolated per scenario)."""
  sid = normalize_scenario_id(scenario_id)
  sc = _load_scenarios_map().get(sid, {})
  org = sc.get("organization") or "当前激活业务场馆"

  if sid == "cymbal_hospital":
    desc = (
        f"检索 {org} 官方门诊导诊 RAG 知识库，获取准确的门诊缴费与医保结算窗口、"
        "抽血采血地点与空腹要求、全院楼层科室分布、常见症状挂号分诊、门诊取药与中药代煎、"
        "CT/核磁共振（MRI）/胃肠镜检查预约与禁忌、专家排班、出入院办理、健康体检、"
        "24小时急诊及院内导航路点。回答任何医院流程、科室位置或就诊规定问题前必须先调用此工具。"
    )
    query_desc = (
        "患者或家属的咨询问题或关键词，例如：'去哪里缴费支持刷医保吗'、"
        "'去哪里抽血需要空腹几个小时'、'头晕胸闷应该挂什么科'、'核磁共振检查在哪里有什么禁忌'。"
    )
    cat_desc = (
        "可选医院导诊分类提示（如 '挂号与缴费'、'检验与抽血'、'楼层与科室导航'、"
        "'智能分诊'、'取药与药事服务'、'检查预约与准备'、'健康体检与入职体检'、'急诊与犬伤门诊'）。"
    )
  elif sid == "cymbal_bank":
    desc = (
        f"检索 {org} 旗舰支行官方业务 RAG 知识库，获取准确的稳健理财产品与业绩比较基准、"
        "定期存款与大额存单利率、水电燃气与社保生活缴费、个人储蓄卡开户与改密提额、"
        "大额取现预约规定、跨境电汇（SWIFT）与留学购汇、房贷按揭与提前还款、"
        "银行流水/存款证明/征信自助打印、企业对公金融、VIP贵宾权益及网点导航路点。"
        "回答任何银行业务、利率门槛或窗口分流问题前必须先调用此工具。"
    )
    query_desc = (
        "客户的咨询问题或业务关键词，例如：'有哪些稳健型理财产品收益率多少'、"
        "'大额存单和定期存款利率'、'水电燃气费去哪里缴'、'取十万现金需要预约吗'、'跨境汇款SWIFT代码'。"
    )
    cat_desc = (
        "可选银行业务分类提示（如 '理财与资产配置'、'便民生活与政务缴费'、'现金与外汇服务'、"
        "'房贷与个人信贷'、'账户与银行卡业务'、'证明打印与征信查询'、'企业对公金融'）。"
    )
  else:
    desc = (
        f"检索 {org} 的专属 RAG 知识库，查询场馆服务指南、办理流程、窗口楼层位置、"
        "收费与政策标准以及关联导航路点。回答场馆服务或业务流程问题前请先调用此工具，确保回答真实准确。"
    )
    query_desc = "访客咨询的完整问题或核心检索关键词。"
    cat_desc = "可选的知识分类提示。"

  return {
      "name": SCENARIO_RAG_TOOL_NAME,
      "description": desc,
      "parameters": {
          "type": "OBJECT",
          "properties": {
              "query": {
                  "type": "STRING",
                  "description": query_desc,
              },
              "category": {
                  "type": "STRING",
                  "description": cat_desc,
              },
          },
          "required": ["query"],
      },
      "behavior": "BLOCKING",
  }


# ---------------------------------------------------------------------------
# Hybrid Tokenizer + Synonym Expansion + BM25 + Optional Vector RRF
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
  """Extract lowercase ASCII words plus CJK unigrams & bigrams."""
  tokens: list[str] = []
  for raw in _TOKEN_RE.findall((text or "").lower()):
    if _CJK_RUN_RE.match(raw):
      for ch in raw:
        tokens.append(ch)
      for i in range(len(raw) - 1):
        tokens.append(raw[i : i + 2])
      for i in range(len(raw) - 2):
        tokens.append(raw[i : i + 3])
    else:
      tokens.append(raw)
  return tokens


def _expand_query_tokens(query: str, base_tokens: list[str]) -> dict[str, float]:
  """Return weighted query terms (original terms = 1.0, synonym expansions = 0.48)."""
  weights: dict[str, float] = {}
  for tok in base_tokens:
    weights[tok] = weights.get(tok, 0.0) + 1.0

  q_low = (query or "").lower()
  for group in _SYNONYM_GROUPS:
    if any(term in q_low for term in group):
      for syn in group:
        for st in _tokenize(syn):
          if st not in weights:
            weights[st] = 0.48
  return weights


def _bm25_scores(
    query: str,
    category: str,
    docs: list[dict[str, Any]],
) -> list[float]:
  """Compute field-weighted BM25 + exact phrase bonus across scenario chunks."""
  if not docs:
    return []
  combined_q = f"{query} {category}".strip()
  base_q_tokens = _tokenize(combined_q)
  q_weights = _expand_query_tokens(combined_q, base_q_tokens)
  if not q_weights:
    return [0.0 for _ in docs]

  doc_term_freqs: list[dict[str, float]] = []
  doc_lengths: list[float] = []
  df: dict[str, int] = {}

  for d in docs:
    tf: dict[str, float] = {}
    fields = (
        (d.get("title", ""), 3.4),
        (" ".join(d.get("keywords") or []), 3.0),
        (d.get("category", ""), 2.0),
        (d.get("location", ""), 1.6),
        (d.get("summary", ""), 1.5),
        (d.get("content", ""), 1.0),
    )
    total_len = 0.0
    for text_val, weight in fields:
      toks = _tokenize(str(text_val))
      total_len += len(toks) * weight
      for t in toks:
        tf[t] = tf.get(t, 0.0) + weight
    for t in tf:
      df[t] = df.get(t, 0) + 1
    doc_term_freqs.append(tf)
    doc_lengths.append(max(total_len, 1.0))

  n_docs = len(docs)
  avg_dl = sum(doc_lengths) / max(n_docs, 1)
  k1 = 1.5
  b = 0.72
  q_low = (query or "").lower().strip()

  scores: list[float] = []
  for idx, d in enumerate(docs):
    tf = doc_term_freqs[idx]
    dl = doc_lengths[idx]
    score = 0.0
    for term, qw in q_weights.items():
      f = tf.get(term, 0.0)
      if f <= 0.0:
        continue
      n_q = df.get(term, 0)
      idf = math.log(1.0 + (n_docs - n_q + 0.5) / (n_q + 0.5))
      denom = f + k1 * (1.0 - b + b * (dl / avg_dl))
      score += qw * idf * ((f * (k1 + 1.0)) / denom)

    # Exact keyword / substring hit bonus for crisp precision on key terms
    for kw in d.get("keywords") or []:
      kw_l = str(kw).lower()
      if len(kw_l) >= 2 and kw_l in q_low:
        score += 3.5
    if q_low and len(q_low) >= 2 and q_low in str(d.get("title", "")).lower():
      score += 4.0
    if d.get("is_custom") and score > 0:
      score *= 1.18
    scores.append(score)
  return scores


def _vertex_embed_batch(texts: list[str], timeout_s: float = 1.1) -> list[list[float]] | None:
  """Embed texts via Vertex AI `text-embedding-005` (`us-central1`) with cache."""
  global _embed_disabled
  if _embed_disabled or os.environ.get("SCENARIO_RAG_EMBEDDINGS", "1") == "0":
    return None
  if os.environ.get("LIVE_BACKEND", "").lower() == "aistudio":
    return None
  project = vertex_rest._vertex_project()  # pylint: disable=protected-access
  if not project:
    return None

  keys = [hashlib.sha256(t.encode("utf-8")).hexdigest()[:24] for t in texts]
  with _embed_lock:
    missing_indices = [i for i, k in enumerate(keys) if k not in _embed_cache]
  if not missing_indices:
    with _embed_lock:
      return [_embed_cache[k] for k in keys]

  try:
    token = vertex_rest._token()  # pylint: disable=protected-access
    loc = os.environ.get("VERTEX_LOCATION", "us-central1")
    url = (
        f"https://{loc}-aiplatform.googleapis.com/v1/projects/{project}"
        f"/locations/{loc}/publishers/google/models/text-embedding-005:predict"
    )
    payload = {
        "instances": [{"content": texts[i][:1800]} for i in missing_indices]
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
      body = json.loads(resp.read().decode("utf-8"))
    preds = body.get("predictions") or []
    if len(preds) != len(missing_indices):
      return None
    with _embed_lock:
      for idx_pos, pred in zip(missing_indices, preds):
        vals = (pred.get("embeddings") or {}).get("values")
        if not vals:
          return None
        _embed_cache[keys[idx_pos]] = [float(x) for x in vals]
      return [_embed_cache[k] for k in keys]
  except Exception as exc:  # pylint: disable=broad-except
    logger.debug("Scenario RAG vector embedding skipped (using BM25): %s", exc)
    if "401" in str(exc) or "403" in str(exc) or "DefaultCredentialsError" in type(exc).__name__:
      _embed_disabled = True
    return None


def _cosine(a: list[float], b: list[float]) -> float:
  dot = sum(x * y for x, y in zip(a, b))
  na = math.sqrt(sum(x * x for x in a))
  nb = math.sqrt(sum(y * y for y in b))
  if na <= 1e-9 or nb <= 1e-9:
    return 0.0
  return dot / (na * nb)


def run_rag_tool(
    args: dict[str, Any],
    scenario_id: str | None,
    custom_kb: str = "",
    top_k: int = 3,
) -> dict[str, Any]:
  """Execute `search_knowledge_base` and return top-K grounded chunks."""
  t0 = time.monotonic()
  query = str((args or {}).get("query") or "").strip()
  category = str((args or {}).get("category") or "").strip()
  if not query:
    return {"error": "query must be a non-empty string"}

  sc = get_scenario(scenario_id, custom_kb=custom_kb)
  docs = sc.get("documents") or []
  if not docs:
    return {
        "scenario": sc.get("id", DEFAULT_SCENARIO_ID),
        "organization": sc.get("organization", ""),
        "query": query,
        "matches": [],
        "note": "No scenario knowledge base entries are configured in this session.",
    }

  bm25 = _bm25_scores(query, category, docs)
  bm25_order = sorted(range(len(docs)), key=lambda i: bm25[i], reverse=True)
  bm25_rank = {doc_idx: rank for rank, doc_idx in enumerate(bm25_order)}

  retrieval_mode = "bm25_ngram"
  vec_texts = [
      f"{d.get('title', '')} | {d.get('category', '')} | {d.get('summary', '')} | {' '.join(d.get('keywords') or [])}"
      for d in docs
  ]
  all_vecs = _vertex_embed_batch([f"{query} {category}".strip()] + vec_texts)
  if all_vecs and len(all_vecs) == len(docs) + 1:
    q_vec = all_vecs[0]
    cos_scores = [_cosine(q_vec, dv) for dv in all_vecs[1:]]
    cos_order = sorted(range(len(docs)), key=lambda i: cos_scores[i], reverse=True)
    cos_rank = {doc_idx: rank for rank, doc_idx in enumerate(cos_order)}
    # Reciprocal Rank Fusion (BM25 weight 1.15 + Semantic Vector weight 1.0)
    fused = [
        1.15 / (12.0 + bm25_rank[i])
        + (0.015 if bm25[i] > 0 else 0.0)
        + 1.0 / (12.0 + cos_rank[i])
        + 0.08 * max(cos_scores[i], 0.0)
        for i in range(len(docs))
    ]
    ranked = sorted(range(len(docs)), key=lambda i: fused[i], reverse=True)
    retrieval_mode = "hybrid_vector_bm25"
  else:
    ranked = bm25_order

  # Select top_k relevant chunks (keep at least top 1 even if phrasing was broad)
  selected: list[dict[str, Any]] = []
  max_bm25 = max(bm25) if bm25 else 0.0
  for rank_pos, idx in enumerate(ranked[: max(1, top_k)]):
    if rank_pos > 0 and max_bm25 > 0 and bm25[idx] <= 0.02 * max_bm25 and retrieval_mode == "bm25_ngram":
      continue
    d = docs[idx]
    norm_score = (
        round(min(0.99, 0.55 + 0.44 * (bm25[idx] / max(max_bm25, 1.0))), 2)
        if max_bm25 > 0
        else 0.60
    )
    selected.append({
        "id": d.get("id", ""),
        "category": d.get("category", ""),
        "title": d.get("title", ""),
        "location": d.get("location", ""),
        "waypoint": d.get("waypoint", ""),
        "summary": d.get("summary", ""),
        "content": d.get("content", ""),
        "score": norm_score,
    })

  latency_ms = int(round((time.monotonic() - t0) * 1000))
  return {
      "scenario": sc.get("id", DEFAULT_SCENARIO_ID),
      "badge": sc.get("badge", ""),
      "organization": sc.get("organization", ""),
      "role": sc.get("role", ""),
      "query": query,
      "retrieval_mode": retrieval_mode,
      "latency_ms": latency_ms,
      "matches": selected,
      "available_waypoints": list(sc.get("waypoints") or []),
      "instruction": (
          "Answer the visitor concisely in their own language using ONLY the"
          " facts in `matches` (state exact floor/window/hours/thresholds/prep"
          " rules clearly in 1-3 short spoken sentences). If the visitor wants"
          " to go there, offer to guide them or call navigate(waypoint=...)."
      ),
  }
