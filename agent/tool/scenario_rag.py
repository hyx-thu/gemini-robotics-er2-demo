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

# Colloquial bilingual synonym clusters so natural visitor phrasing (in either
# English or Chinese) hits exact English policy chunks even before vector RRF.
_SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("缴费", "交费", "交钱", "付钱", "付款", "买单", "收费", "结算", "发票", "收据", "pay", "payment", "billing", "cashier", "invoice", "receipt"),
    ("刷医保", "医保", "报销", "异地医保", "商保", "insurance", "medicare", "reimbursement", "copay"),
    ("水费", "电费", "燃气", "煤气", "暖气", "宽带", "话费", "社保缴费", "交社保", "罚款", "非税", "utility", "utilities", "electricity", "water", "gas", "heating", "social_security", "municipal"),
    ("抽血", "采血", "验血", "查血", "化验", "血常规", "空腹", "肝功", "肾功", "血糖", "血脂", "blood", "phlebotomy", "venipuncture", "fasting", "cbc", "lab"),
    ("尿检", "验尿", "尿常规", "大便", "便常规", "留样", "标本", "痰", "urine", "urinalysis", "stool", "sputum", "specimen", "sample"),
    ("理财", "买理财", "收益", "收益率", "年化", "活钱", "稳健", "固收", "基金", "风险测评", "风险评估", "天天盈", "闲钱", "打理", "wealth", "investment", "yield", "return", "annualized", "fund", "savings", "idle"),
    ("存款", "存钱", "定期", "定存", "利息", "利率", "大额存单", "国债", "保本", "养老金", "黄金", "积存金", "deposit", "deposits", "interest", "cd", "cds", "treasury", "bond", "bonds", "pension", "gold", "bullion"),
    ("取钱", "取现", "取款", "拿现金", "大额取现", "预约取款", "十万", "五万", "零钱", "零钱包", "新钞", "残币", "atm", "cash", "withdrawal", "withdraw", "change", "banknotes"),
    ("办卡", "开卡", "开户", "储蓄卡", "借记卡", "一类卡", "二类卡", "工资卡", "社保卡激活", "debit", "account", "payroll"),
    ("密码", "忘密码", "忘记密码", "锁住", "挂失", "补卡", "换卡", "手机号", "限额", "提额", "password", "pin", "locked", "lost", "limit"),
    ("外汇", "换汇", "购汇", "结汇", "美元", "日元", "欧元", "港币", "跨境", "电汇", "汇款", "打钱", "留学", "学费", "swift", "forex", "fx", "wire", "remittance", "currency", "usd", "overseas", "tuition"),
    ("贷款", "借钱", "房贷", "按揭", "提前还款", "提前还贷", "消费贷", "经营贷", "抵押", "mortgage", "mortgages", "loan", "loans", "prepayment", "repayment", "refinance", "lpr", "sme", "apr"),
    ("流水", "打流水", "交易明细", "存款证明", "资信证明", "资产证明", "征信", "信用报告", "签证", "statement", "statements", "solvency", "certificate", "certificates", "credit_report", "visa", "embassy"),
    ("对公", "公司开户", "企业开户", "基本户", "营业执照", "公章", "回单", "u盾", "corporate", "ukey", "voucher", "vouchers"),
    ("贵宾", "金卡", "金葵馨", "私人银行", "优先叫号", "vip", "private_banking", "aum"),
    ("冻结", "止付", "非柜面", "解冻", "解控", "不能转账", "诈骗", "反诈", "安全账户", "frozen", "blocked", "suspended", "scam", "fraud", "unfreeze"),
    ("老人", "长者", "卧床", "不能本人来", "上门", "已故", "去世", "遗产", "elderly", "senior", "bedridden", "homebound", "doorstep", "mobility", "bereavement", "estate", "deceased"),
    ("挂号", "预约挂号", "加号", "放号", "退号", "签到", "报到", "过号", "初诊", "register", "registration", "appointment", "booking", "missed"),
    ("几楼", "在哪", "楼层", "怎么走", "科室分布", "导航", "floor", "directory", "department", "departments"),
    ("挂什么科", "看什么科", "分诊", "头晕", "头痛", "胸闷", "心慌", "肚子疼", "胃痛", "咳嗽", "发烧", "发热", "腰痛", "扭伤", "皮疹", "triage", "symptom", "dizziness", "dizzy", "headache", "chest", "stomachache", "cough", "fever", "sprain", "rash"),
    ("取药", "拿药", "药房", "中药", "代煎", "煎药", "寄回家", "快递", "处方", "用药", "pharmacy", "dispensary", "medication", "medicine", "prescription", "herbal", "decoction", "delivery"),
    ("核磁", "核磁共振", "拍片", "胸片", "彩超", "b超", "超声", "心电图", "憋尿", "金属", "起搏器", "mri", "ct", "xray", "radiology", "imaging", "ultrasound", "ecg", "scan", "metal", "pacemaker"),
    ("胃镜", "肠镜", "胃肠镜", "无痛", "内镜", "清肠", "麻醉", "endoscopy", "gastroscopy", "colonoscopy", "sedation", "sedated"),
    ("专家", "主任", "排班", "出诊", "坐诊", "几点开门", "营业时间", "周末", "挂号费", "特需", "schedule", "specialist", "weekend"),
    ("住院", "入院", "出院", "押金", "病房", "探视", "陪护", "inpatient", "admission", "discharge", "ward"),
    ("体检", "入职体检", "驾照体检", "健康证", "早癌", "checkup", "physical", "onboarding", "employment", "screening"),
    ("病历", "病假条", "请假条", "诊断证明", "盖章", "复印病历", "record", "records", "stamp", "seal", "chart"),
    ("急诊", "急救", "狗咬", "猫抓", "狂犬", "破伤风", "24小时", "emergency", "urgent", "rabies", "tetanus", "scratch", "bite"),
    ("轮椅", "平车", "停车", "停车费", "车库", "母婴室", "餐厅", "食堂", "咖啡", "充电宝", "wheelchair", "parking", "garage", "cafe", "cafeteria"),
)

_STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do",
    "does", "for", "from", "get", "go", "have", "how", "i", "if", "in", "into",
    "is", "it", "me", "my", "need", "of", "on", "or", "our", "please", "should",
    "so", "some", "take", "tell", "that", "the", "their", "there", "these",
    "this", "to", "up", "us", "want", "we", "what", "when", "where", "which",
    "who", "why", "will", "with", "work", "would", "you", "your",
})

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
        "category": "Custom Knowledge",
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
    subtitle = sc.get("subtitle", sc.get("label_zh", sc.get("name", sid)))
    out.append({
        "id": sid,
        "name": sc.get("name", sid),
        "subtitle": subtitle,
        "label_zh": subtitle,
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
  org = sc.get("organization") or "Custom Service Venue"
  role = sc.get("role") or "Embodied Service Concierge Robot"
  persona_text = cp if cp else (sc.get("persona_prompt") or "").strip()
  wps = sc.get("waypoints") or []
  wp_line = (
      f"- Scenario navigation waypoints (`get_waypoints` / `navigate`): {', '.join(f'`{w}`' for w in wps)}."
      if sid != DEFAULT_SCENARIO_ID and wps
      else ""
  )
  parts = [
      f"## Active Industry Scenario & Role Persona: {org} — {role}",
  ]
  if persona_text:
    parts.append(persona_text)
  elif has_custom_kb:
    parts.append(
        "Whenever a visitor asks about venue policies, service workflows, floor"
        " locations, or operating hours, call `search_knowledge_base` first and"
        " answer strictly based on the retrieved knowledge entries in the"
        " visitor's own language."
    )
  if wp_line:
    parts.append(wp_line)
  parts.append(
      "- Do NOT call `update_plan` for `search_knowledge_base` Q&A"
      " (`update_plan` is strictly reserved for physical multi-step arm"
      " pick/place manipulation); NEVER call `search_knowledge_base` on"
      " `[HEARTBEAT]` prompts."
  )
  return "\n".join(parts)


def search_knowledge_base_tool(scenario_id: str | None = None) -> dict[str, Any]:
  """Gemini FunctionDeclaration for `search_knowledge_base` (strictly isolated per scenario)."""
  sid = normalize_scenario_id(scenario_id)
  sc = _load_scenarios_map().get(sid, {})
  org = sc.get("organization") or "the active venue"

  if sid == "cymbal_hospital":
    desc = (
        f"Search the official {org} outpatient triage & concierge RAG knowledge"
        " base for grounded hospital workflows: outpatient billing & health"
        " insurance settlement windows, blood draw (phlebotomy) locations &"
        " fasting rules, floor-by-floor department directory, symptom-based"
        " clinic triage, pharmacy prescription pickup & herbal delivery,"
        " CT/MRI/endoscopy preparation & safety rules, specialist clinic"
        " schedules, inpatient admission/discharge, health checkup packages,"
        " 24/7 emergency & animal-bite care, and hospital navigation waypoints."
        " Always call this tool before answering any hospital policy, location,"
        " or preparation question."
    )
    query_desc = (
        "The patient or visitor's question or keywords, e.g. 'where to pay my"
        " outpatient bill and settle insurance', 'where to get a blood draw and"
        " how long to fast', 'which clinic to register for dizziness', or 'MRI"
        " scan location and metal safety rules'."
    )
    cat_desc = (
        "Optional hospital category filter hint (e.g. 'Registration & Billing',"
        " 'Lab & Phlebotomy', 'Floor & Department Directory', 'Symptom Triage',"
        " 'Pharmacy & Prescriptions', 'Imaging & Endoscopy Prep', 'Health"
        " Checkup Center', '24/7 Emergency Care')."
    )
  elif sid == "cymbal_bank":
    desc = (
        f"Search the official {org} retail & wealth lobby RAG knowledge base"
        " for grounded banking policies: low-volatility wealth products &"
        " annualized benchmarks, term deposit & CD interest rates, municipal"
        " utility & social-security bill payment, personal debit card opening &"
        " PIN/limit services, large cash withdrawal booking thresholds,"
        " cross-border SWIFT wire transfers & currency exchange, residential"
        " mortgages & SME loans, bank statements & solvency certificates,"
        " corporate banking, VIP wealth lounge tiers, and branch navigation"
        " waypoints. Always call this tool before answering any banking"
        " product, rate, fee, or counter routing question."
    )
    query_desc = (
        "The client's banking question or keywords, e.g. 'low-risk wealth"
        " management products and yield', '3-year CD and term deposit rates',"
        " 'where to pay utility bills', 'do I need an appointment to withdraw"
        " 100k cash', or 'overseas tuition SWIFT wire code'."
    )
    cat_desc = (
        "Optional banking category filter hint (e.g. 'Wealth Management',"
        " 'Deposits, CDs & Bonds', 'Utility & Municipal Bill Pay', 'Cash &"
        " Foreign Exchange', 'Mortgages & Loans', 'Cards & Account Services',"
        " 'Statements & Certificates', 'Corporate Banking')."
    )
  else:
    desc = (
        f"Search the dedicated RAG knowledge base for {org} to retrieve"
        " grounded venue service guides, workflows, floor/counter locations,"
        " fee schedules, policy rules, and associated navigation waypoints."
        " Always call this tool before answering venue-specific policy or"
        " workflow questions."
    )
    query_desc = "The visitor's full question or key search terms."
    cat_desc = "Optional knowledge category filter hint."

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
  """Extract lowercase ASCII words (excluding stop words) plus CJK unigrams & bigrams."""
  tokens: list[str] = []
  for raw in _TOKEN_RE.findall((text or "").lower()):
    if _CJK_RUN_RE.match(raw):
      for ch in raw:
        tokens.append(ch)
      for i in range(len(raw) - 1):
        tokens.append(raw[i : i + 2])
      for i in range(len(raw) - 2):
        tokens.append(raw[i : i + 3])
    elif raw not in _STOP_WORDS:
      tokens.append(raw)
  return tokens


def _term_matches_query(term: str, q_low: str, token_set: set[str]) -> bool:
  """Check if a synonym term appears in query (whole-token for ASCII, substring for CJK)."""
  if _CJK_RUN_RE.search(term):
    return term in q_low
  if "_" in term:
    phrase = term.replace("_", " ")
    return phrase in q_low
  return term in token_set


def _expand_query_tokens(query: str, base_tokens: list[str]) -> dict[str, float]:
  """Return weighted query terms (original terms = 1.0, synonym expansions = 0.45)."""
  weights: dict[str, float] = {}
  for tok in base_tokens:
    weights[tok] = weights.get(tok, 0.0) + 1.0

  q_low = (query or "").lower()
  token_set = set(base_tokens)
  for group in _SYNONYM_GROUPS:
    if any(_term_matches_query(term, q_low, token_set) for term in group):
      for syn in group:
        for st in _tokenize(syn.replace("_", " ")):
          if st not in weights:
            weights[st] = 0.45
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

    # Exact keyword / phrase hit bonus (whole-word for ASCII, substring for CJK)
    base_tok_set = set(base_q_tokens)
    for kw in d.get("keywords") or []:
      kw_l = str(kw).lower().strip()
      if len(kw_l) < 2:
        continue
      if _CJK_RUN_RE.search(kw_l):
        hit = kw_l in q_low
      elif " " in kw_l or "-" in kw_l:
        hit = bool(re.search(rf"\b{re.escape(kw_l)}\b", q_low))
      else:
        hit = kw_l in base_tok_set
      if hit:
        score += 3.5
    if q_low and len(q_low) >= 3 and q_low in str(d.get("title", "")).lower():
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
