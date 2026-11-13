"""Scenario Persona & Hybrid RAG Knowledge Base for Embodied Service Robots.

Provides selectable industry scenarios (e.g. Cymbal Hospital outpatient triage
concierge, Cymbal Bank flagship branch lobby manager), scenario-specific
navigation waypoints, concise persona prompt blocks, programmatic RAG-grounded
topic synthesis (`build_topic_guide_from_documents`), and a fast hybrid RAG
retriever (`search_knowledge_base`) combining:
  1. Field-weighted BM25 over English words, CJK unigrams/bigrams, and
     bilingual colloquial query-expansion synonyms (<2 ms, zero network deps),
     with salient-token out-of-KB rejection and unmatched-term detection so the
     model never fabricates ungrounded policies.
  2. Optional semantic vector similarity (`text-embedding-005` on Vertex AI
     `us-central1` with in-memory chunk caching and timeout guard), fused
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

# Visual icons per category (topic metadata is 100% derived from `documents`).
_CATEGORY_ICONS: dict[str, str] = {
    "Custom Knowledge": "📝",
    # Cymbal Hospital (10 Medical Domains)
    "Registration & Billing": "🪪",
    "Lab & Phlebotomy": "🩸",
    "Symptom Triage & Navigation": "🧭",
    "Radiology, Ultrasound & Endoscopy": "☢️",
    "Pharmacy & Prescriptions": "💊",
    "Maternity, Pediatrics & Vaccines": "👶",
    "Dental, Eye, ENT & Rehab": "🦷",
    "Inpatient, Surgery & Day Care": "🛏️",
    "Health Checkups & Medical Records": "📋",
    "24/7 Emergency & Campus Amenities": "🚑",
    # Cymbal Bank (10 Banking Domains)
    "Lobby Routing & Smart Kiosks": "🏛️",
    "Cards & Personal Accounts": "💳",
    "Savings, CDs & Sovereign Bonds": "🏦",
    "Wealth Management & Bullion": "📈",
    "Cash, Coins & Foreign Exchange": "💱",
    "Loans, Mortgages & Credit Cards": "🏠",
    "Statements, Certificates & Credit": "📄",
    "Utility, Tax & Municipal Pay": "🧾",
    "Corporate & SME Banking": "🏢",
    "Security, Accessibility & Special Care": "🛡️",
}

# Colloquial bilingual synonym clusters so natural visitor phrasing (in either
# English or Chinese) hits exact English policy chunks even before vector RRF.
# Kept fine-grained per sub-topic so 120-doc per-scenario catalogs distinguish
# sibling policies cleanly.
_SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    # Hospital — Registration, Billing & International Insurance
    ("缴费", "交费", "交钱", "付钱", "付款", "买单", "门诊缴费", "收费窗口", "pay", "payment", "billing", "cashier", "outpatient_bill", "kiosk", "apple_pay", "hsa"),
    ("发票", "电子发票", "收据", "费用清单", "报销清单", "报销明细", "打发票", "superbill", "icd_10", "cpt", "invoice", "receipt", "itemized", "reimbursement"),
    ("刷医保", "医保", "医保卡", "政府医保", "公费医疗", "院外报销", "自付额", "medicare", "public_health_plan", "out_of_network", "deductible", "eligibility"),
    ("商保", "商业保险", "国际保险", "直付", "免现金直付", "高端医疗", "担保函", "commercial_insurance", "direct_billing", "gop", "bupa", "cigna", "allianz", "aetna", "unitedhealthcare", "axa"),
    ("挂号", "预约", "预约挂号", "专家号", "加号", "放号", "当日号", "初诊", "建卡", "病历号", "护照挂号", "appointment", "booking", "scheduling", "specialist", "mrn", "new_patient", "passport"),
    ("签到", "报到", "过号", "迟到", "回诊", "看结果", "复诊", "missed_turn", "check_in", "late_arrival", "return_visit"),
    ("退号", "取消预约", "改约", "退费", "退款", "refund", "cancel", "reschedule"),
    ("慢病管理", "慢性病报销", "年度封顶", "工伤", "车祸", "交通事故", "chronic_care_plan", "out_of_pocket_maximum", "workers_comp", "motor_vehicle_accident"),
    # Hospital — Lab & Phlebotomy
    ("抽血", "采血", "验血", "查血", "化验", "抽血窗口", "blood_draw", "phlebotomy", "venipuncture", "adult"),
    ("空腹", "禁食", "喝水", "几个小时", "肝功", "肾功", "血脂", "fasting", "hours", "water", "lipid", "liver"),
    ("儿童抽血", "小孩抽血", "指尖血", "末梢血", "血常规", "fingerstick", "pediatric_blood_draw", "cbc"),
    ("糖耐", "糖耐量", "喝糖水", "妊娠糖尿病", "胰岛素释放", "ogtt", "glucose_tolerance"),
    ("尿检", "验尿", "尿常规", "中段尿", "24小时尿", "urine", "urinalysis", "midstream"),
    ("大便", "便常规", "潜血", "便潜血", "留样", "标本", "stool", "fecal", "occult", "fit_test", "specimen"),
    ("痰培养", "咽拭子", "流感拭子", "支原体", "链球菌", "sputum", "throat_swab", "influenza_swab", "strep_swab"),
    # Hospital — Triage, Imaging & Endoscopy
    ("几楼", "在哪", "楼层", "怎么走", "科室分布", "导航", "带路", "floor", "directory", "department", "escort"),
    ("头晕", "眩晕", "天旋地转", "耳石症", "偏头痛", "dizziness", "dizzy", "vertigo", "bppv", "neurology"),
    ("胸闷", "胸痛", "心慌", "肚子疼", "胃痛", "反酸", "咳嗽", "发烧", "高烧", "发热", "皮疹", "扭伤", "triage", "symptom", "chest", "stomachache", "cough", "fever", "pulmonology", "rash", "sprain"),
    ("核磁", "核磁共振", "金属", "种植牙核磁", "起搏器", "幽闭", "mri", "titanium", "pacemaker", "metal"),
    ("增强ct", "造影剂", "二甲双胍", "拍片", "胸片", "辐射", "心电图", "动态心电", "ct", "contrast", "metformin", "xray", "holter", "ecg"),
    ("彩超", "b超", "超声", "憋尿", "空腹超声", "骨密度", "钼靶", "ultrasound", "bladder", "dexa", "mammogram"),
    ("胃镜", "肠镜", "胃肠镜", "无痛胃镜", "无痛肠镜", "静脉镇静", "内镜", "清肠", "呼气试验", "幽门螺杆菌", "endoscopy", "egd", "gastroscopy", "colonoscopy", "iv_sedation", "urea_breath"),
    # Hospital — Pharmacy, Maternity/Peds, Specialties, Inpatient, Travel/Checkup & Emergency
    ("取药", "拿药", "药房", "自动药房", "用药咨询", "药物相互作用", "pharmacy", "dispensary", "medication", "prescription", "polypharmacy"),
    ("特配药", "院内制剂", "儿童调味药", "送药", "送药上门", "寄回家", "快递到家", "送到酒店", "compounding", "flavored_suspension", "home_delivery", "hotel_delivery", "medication_courier", "courier"),
    ("胰岛素", "冷链", "冰袋", "冷藏药", "慢病长处方", "长处方", "开三个月", "输液", "生物制剂", "静脉补铁", "打点滴", "靶向药", "预授权", "insulin", "ice_pack", "90_day_supply", "infusion_center", "biologics_infusion", "iv_iron", "prior_authorization"),
    ("产检", "建档", "孕早期", "助产士", "参观产房", "四维", "大排畸", "无创", "胎心监护", "分娩", "宫缩", "破水", "prenatal_intake", "midwife", "maternity_tour", "nt_scan", "20_week_anatomy_scan", "nipt", "nst", "labor"),
    ("儿科", "小孩看病", "雾化", "儿保", "生长发育", "骨龄", "哺乳室", "母婴室", "产后复查", "盆底肌", "通乳", "pediatrics", "nebulizer", "well_child", "bone_age", "lactation", "postpartum", "pelvic_floor"),
    ("疫苗", "打疫苗", "儿童疫苗", "预防接种", "黄卡", "九价", "宫颈癌疫苗", "带状疱疹", "流感疫苗", "宫颈筛查", "vaccine", "immunization", "yellow_card", "hpv", "shingles", "flu", "pap_test"),
    ("牙科", "口腔", "拔牙", "智齿", "阻生齿", "洗牙", "补牙", "根管", "种牙", "dental", "wisdom", "tooth", "extraction", "scaling"),
    ("眼科", "散瞳", "眼底", "近视", "配镜", "角膜塑形镜", "白内障", "青光眼", "干眼", "ophthalmology", "dilated", "retinal", "myopia", "ortho_k", "cataract"),
    ("耳鼻喉", "耳鸣", "突聋", "听力", "鼻窦炎", "鼻炎", "流鼻血", "鱼刺", "卡鱼刺", "咽喉异物", "打呼噜", "睡眠监测", "ent", "tinnitus", "audiometry", "sinusitis", "nosebleed", "fishbone", "foreign_body", "snoring", "sleep_apnea"),
    ("康复", "理疗", "运动康复", "冲击波", "针灸", "干针", "整合医学", "肌筋膜", "失眠", "心理门诊", "冷冻祛疣", "激光祛疤", "换药", "拆线", "picc", "physical_therapy", "physiotherapy", "acupuncture", "dry_needling", "integrative_medicine", "insomnia", "wart_cryotherapy", "suture_removal", "wound_care"),
    ("住院", "入院", "出院", "床旁结算", "送药到床", "探视", "陪护", "私人陪护", "注册护士", "日间手术", "inpatient", "admission", "discharge", "meds_to_beds", "visiting_hours", "cna", "private_duty_sitter", "day_surgery"),
    ("住院餐", "病房点餐", "营养餐", "糖尿病餐", "清真", "清真餐", "犹太洁食", "素食", "无麸质", "halal", "kosher", "vegan", "gluten_free", "diabetic_diet", "low_sodium", "inpatient_meals"),
    ("国际医疗", "国际部", "医学翻译", "医疗口译", "口译", "翻译", "多语种", "英语陪诊", "西语翻译", "阿语翻译", "日语翻译", "医疗签证", "特需套房", "international_patient", "medical_interpreter", "spanish_interpreter", "arabic_interpreter", "medical_visa", "deluxe_suite"),
    ("体检", "年度体检", "高管体检", "入职体检", "留学体检", "签证体检", "驾照体检", "视力体检", "运动体检", "早癌筛查", "executive_physical", "pre_employment_physical", "visa_medical", "driver_medical", "sports_clearance"),
    ("旅行门诊", "国际旅行", "旅行疫苗", "出国疫苗", "黄热病", "黄皮书", "伤寒", "疟疾", "高原反应", "travel_clinic", "travel_vaccine", "yellow_fever", "icvp", "typhoid", "malaria_pills", "altitude_sickness"),
    ("病历", "病假条", "请假条", "医疗证明", "适航证明", "乘机证明", "复印病历", "出生证明", "新生儿护照", "病理切片", "工作能力评估", "sick_note", "work_excuse", "fmla", "fit_to_fly", "medical_records", "hipaa_release", "birth_certificate", "pathology_slide", "functional_capacity"),
    ("急诊", "急救", "分级分诊", "胸痛急救", "卒中", "创伤", "狗咬", "猫抓", "蝙蝠", "狂犬", "狂犬疫苗", "免疫球蛋白", "破伤风", "emergency", "esi_triage", "stemi", "stroke", "rabies_vaccine", "hrig", "tetanus_shot", "bite", "scratch"),
    ("轮椅", "助行器", "停车", "停车费", "充电桩", "食堂", "餐厅", "轻食", "咖啡", "班车", "地铁", "祈祷室", "礼拜室", "冥想室", "静思室", "行李寄存", "手语", "盲文", "无障碍", "社工", "慈善救助", "失物招领", "丢护照", "wheelchair", "parking_validation", "ev_charging", "cafeteria", "healthy_deli", "free_shuttle", "metro", "prayer_room", "meditation_room", "luggage_locker", "sign_language", "asl_interpreter", "braille", "social_worker", "lost_and_found"),
    # Bank — Lobby, Cards, Deposits & Wealth
    ("智能柜员机", "超柜", "取号", "排号", "贵宾室", "私人银行", "stm", "kiosk", "vip", "private_banking", "aum"),
    ("办卡", "开卡", "储蓄卡", "借记卡", "一类卡", "二类卡", "工资卡", "社保卡", "外籍开户", "debit", "tier_i", "payroll", "expat"),
    ("忘密码", "密码锁了", "重置密码", "挂失", "补卡", "同号换卡", "转账限额", "提额", "休眠户", "pin", "locked", "reset", "lost_card", "transfer_limit", "dormant"),
    ("存款", "存钱", "定期", "定存", "大额存单", "存单转让", "国债", "个人养老金", "结构性存款", "deposit", "cd", "large_cd", "treasury", "sovereign_bond", "pension", "structured_deposit"),
    ("理财", "买理财", "理财产品", "稳健", "稳健型", "固收", "收益率", "年化", "收益", "wealth", "steady", "fixed_income", "r2", "yield", "annualized", "benchmark"),
    ("风险测评", "风险评估", "首次理财", "首次风险", "risk_assessment", "first_time_wealth", "questionnaire", "c1"),
    ("活钱", "天天盈", "闲钱", "灵活赎回", "cash_management", "daily_plus", "liquid_wealth"),
    ("基金", "定投", "指数基金", "黄金", "积存金", "金条", "回购", "mutual_fund", "sip", "gold", "bullion", "buyback"),
    # Bank — Cash, FX, Loans, Statements, Utilities, Corporate & Security
    ("大额取现", "预约取款", "提前预约", "十万", "五万", "二十万", "large_cash_withdrawal", "withdraw_100000", "withdraw_50000", "cash_appointment", "50000", "100000"),
    ("取钱", "取现", "取款", "存现金", "扫码取款", "atm", "cash_deposit", "cardless"),
    ("零钱", "零钱包", "新钞", "换零钱", "硬币", "存硬币", "残币", "破损币", "change_pack", "crisp", "coins", "coin_machine", "damaged_banknote"),
    ("外汇", "换汇", "购汇", "结汇", "外币现钞", "美元", "日元", "欧元", "跨境汇款", "电汇", "留学学费", "swift", "forex", "fx_quota", "foreign_cash", "wire", "tuition"),
    ("房贷", "按揭", "提前还款", "提前还贷", "重定价", "解押", "消费贷", "经营贷", "车贷", "信用卡", "白金卡", "盗刷", "mortgage", "prepayment", "lpr", "lien", "consumer_loan", "sme_loan", "credit_card", "chargeback"),
    ("流水", "打流水", "交易明细", "存款证明", "资信证明", "征信", "个人征信", "签证流水", "statement", "solvency", "certificate", "credit_report", "visa"),
    ("水费", "电费", "燃气", "煤气", "暖气", "取暖费", "罚款", "非税", "物业费", "高速etc", "维修资金", "utility", "electricity", "water", "heating", "traffic_fine", "etc", "maintenance_fund"),
    ("对公", "公司开户", "企业开户", "基本户", "营业执照", "公章", "回单", "对账", "u盾", "代发工资", "承兑汇票", "信用证", "收款码", "open_corporate_account", "basic_account", "ukey", "hui_dan", "payroll_service", "lc", "pos"),
    ("冻结", "止付", "非柜面", "暂停非柜面", "解冻", "解控", "不能转账", "诈骗", "反诈", "96110", "frozen", "suspended", "non_counter", "unfreeze", "scam", "fraud"),
    ("保管箱", "保险箱", "金库", "上门服务", "卧床老人", "不能本人来", "已故", "去世", "遗产提取", "手语", "无障碍", "safe_deposit_box", "vault", "doorstep", "bedridden", "bereavement", "deceased", "sign_language"),
)

_STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do",
    "does", "for", "from", "get", "go", "have", "how", "i", "if", "in", "into",
    "is", "it", "me", "my", "need", "of", "on", "or", "our", "please", "should",
    "so", "some", "take", "tell", "that", "the", "their", "there", "these",
    "this", "to", "up", "us", "want", "we", "what", "when", "where", "which",
    "who", "why", "will", "with", "work", "would", "you", "your",
})

# Generic venue / conversational framing tokens that appear in almost any query
# ("Can I buy X at the counter?", "Do you offer Y at the hospital?") and MUST NOT
# by themselves trigger a false-positive RAG match when no domain-specific token
# matches any document.
_GENERIC_FRAMING_TOKENS = frozenset({
    "cymbal", "hospital", "medical", "center", "clinic", "clinics",
    "patient", "patients", "doctor", "doctors", "nurse", "nurses",
    "bank", "banking", "branch", "flagship", "client", "clients",
    "customer", "customers", "visitor", "visitors", "person", "people",
    "lobby", "hall", "counter", "counters", "window", "windows",
    "desk", "room", "rooms", "zone", "area", "wing", "east", "west",
    "north", "south", "central", "level", "building", "tower",
    "buy", "sell", "open", "apply", "book", "make", "check", "find",
    "use", "offer", "provide", "available", "support", "handle",
    "service", "services", "rule", "rules", "policy", "policies",
    "process", "procedure", "guide", "info", "information", "detail",
    "details", "question", "help", "ask", "know", "explain", "about",
    "here", "today", "now", "state", "regarding", "knowledge", "base",
    "package", "packages", "perform", "plan", "plans",
})

# Common Chinese conversational/venue framing phrases stripped when identifying
# unmatched out-of-scope CJK terms in partial-overlap visitor questions.
_CJK_FRAMING_PHRASES: tuple[str, ...] = (
    "请问一下", "想问一下", "帮我查下", "帮我看下", "能不能", "可以吗", "有没有",
    "卖不卖", "办不办", "支持吗", "需要注意什么", "需要注意", "怎么办理",
    "请问", "你们", "我们", "咱们", "这里", "那边", "医院", "银行", "网点",
    "支行", "柜台", "窗口", "大厅", "门诊", "服务", "业务", "可以", "是否",
    "怎么", "哪里", "在哪", "去哪", "什么", "多少", "需要", "支持", "提供",
    "办理", "一下", "现在", "今天", "请问",
)
_CJK_FRAMING_CHARS = frozenset("吗呢啊吧呀的了和跟与或在有是能会要想去到给把被做办打看买卖不没多几哪")

_TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]+", re.IGNORECASE)
_CJK_RUN_RE = re.compile(r"^[\u4e00-\u9fff]+$")

_scenarios_cache: dict[str, dict[str, Any]] | None = None
_embed_lock = threading.Lock()
_embed_cache: dict[str, list[float]] = {}
_embed_disabled: bool = False


def derive_question_from_doc(doc: dict[str, Any]) -> str:
  """Return a strictly document-grounded sample question for `doc`.

  Uses the verified `sample_question` stored on built-in RAG entries, or builds
  a pure title-scoped prompt for user-entered custom RAG notes without adding
  any unbacked assumptions.
  """
  sq = str(doc.get("sample_question") or "").strip()
  if sq:
    return sq
  title = re.sub(r"…+$", "", str(doc.get("title") or "this entry")).strip()
  if not title:
    title = "this entry"
  if title.endswith("?"):
    return title
  return f"What does the knowledge base state regarding {title}?"


def build_topic_guide_from_documents(
    docs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
  """Programmatically derive Topic Scopes & Sample Prompts strictly from `docs`.

  Every topic category, summary, and sample question is aggregated directly
  from the actual RAG `documents` so UI topic scopes can never drift or invent
  topics absent from the RAG knowledge base.
  """
  if not docs:
    return []

  grouped: dict[str, list[dict[str, Any]]] = {}
  for d in docs:
    cat = str(d.get("category") or "General").strip() or "General"
    grouped.setdefault(cat, []).append(d)

  topics: list[dict[str, Any]] = []
  for cat, cat_docs in grouped.items():
    short_titles: list[str] = []
    for d in cat_docs[:4]:
      raw_t = str(d.get("title") or "").strip()
      short_t = re.split(r"\s*(?:[:(—]|–)\s*", raw_t)[0].strip()
      if short_t:
        short_titles.append(short_t)
    more_cnt = max(0, len(cat_docs) - len(short_titles))
    more_suffix = f" (+{more_cnt} more)" if more_cnt > 0 else ""
    entry_word = "entry" if len(cat_docs) == 1 else "entries"
    summary = (
        f"{len(cat_docs)} RAG {entry_word}: "
        + " · ".join(short_titles)
        + more_suffix
    )

    grounded_prompts = [
        {
            "doc_id": str(d.get("id") or ""),
            "doc_title": str(d.get("title") or ""),
            "location": str(d.get("location") or ""),
            "prompt": derive_question_from_doc(d),
        }
        for d in cat_docs
    ]
    topics.append({
        "category": cat,
        "icon": _CATEGORY_ICONS.get(cat, "📌"),
        "doc_count": len(cat_docs),
        "doc_ids": [str(d.get("id") or "") for d in cat_docs],
        "summary": summary,
        "sample_prompts": [gp["prompt"] for gp in grounded_prompts],
        "grounded_prompts": grounded_prompts,
    })
  return topics


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
    kw_tokens = list(dict.fromkeys(_tokenize(f"{title} {body}")))[:15]
    doc_entry: dict[str, Any] = {
        "id": f"custom_{scenario_id or 'kb'}_{idx + 1}",
        "category": "Custom Knowledge",
        "title": title,
        "location": "",
        "waypoint": "",
        "keywords": [title] + kw_tokens,
        "summary": summary,
        "content": body,
        "is_custom": True,
    }
    doc_entry["sample_question"] = derive_question_from_doc(doc_entry)
    chunks.append(doc_entry)
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
  base["topic_guide"] = build_topic_guide_from_documents(docs)
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
  """Return all scenarios with lightweight document catalogs & derived topics."""
  smap = _load_scenarios_map()
  out = []
  for sid, sc in smap.items():
    raw_docs = list(sc.get("documents") or [])
    docs = [
        {
            "id": d.get("id", ""),
            "category": d.get("category", ""),
            "title": d.get("title", ""),
            "location": d.get("location", ""),
            "waypoint": d.get("waypoint", ""),
            "summary": d.get("summary", ""),
            "sample_question": derive_question_from_doc(d),
        }
        for d in raw_docs
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
        "topic_guide": build_topic_guide_from_documents(docs),
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
  if cp or (sid == DEFAULT_SCENARIO_ID and has_custom_kb):
    parts.append(
        "- **Mandatory RAG Grounding & Zero-Hallucination Rule**: Whenever a"
        " visitor asks about venue policies, service workflows, floor"
        " locations, or operating hours, call `search_knowledge_base` first and"
        " answer strictly from the retrieved `matches` in the visitor's own"
        " language. If `grounded` is `false` or a detail is absent from"
        " `matches`, state clearly that it is not in the active knowledge base"
        " rather than guessing or inventing details."
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
        f"Search the official {org} 120-entry outpatient triage & concierge RAG"
        " knowledge base across 10 hospital domains: Registration & Billing,"
        " Lab & Phlebotomy, Symptom Triage & Navigation, Radiology/Ultrasound &"
        " Endoscopy, Pharmacy & Prescriptions, Maternity/Pediatrics & Vaccines,"
        " Dental/Eye/ENT & Rehab, Inpatient/Surgery & Day Care, Health Checkups"
        " & Medical Records, and 24/7 Emergency & Campus Amenities. Always call"
        " this tool before answering any hospital policy, floor/window,"
        " preparation, or fee question, and never invent facts outside returned"
        " matches."
    )
    query_desc = (
        "The patient or visitor's question or keywords, e.g. 'international"
        " insurance cashless direct billing GOP', 'blood draw fasting water"
        " rule', 'MRI titanium dental implant safety', 'WHO Yellow Fever travel"
        " vaccine', 'Fit-to-Fly medical certificate', or '15-language medical"
        " interpreter'."
    )
    cat_desc = (
        "Optional hospital category filter hint (e.g. 'Registration & Billing',"
        " 'Lab & Phlebotomy', 'Symptom Triage & Navigation', 'Radiology,"
        " Ultrasound & Endoscopy', 'Pharmacy & Prescriptions', 'Maternity,"
        " Pediatrics & Vaccines', 'Dental, Eye, ENT & Rehab', 'Inpatient,"
        " Surgery & Day Care', 'Health Checkups & Medical Records', '24/7"
        " Emergency & Campus Amenities')."
    )
  elif sid == "cymbal_bank":
    desc = (
        f"Search the official {org} 120-entry retail, wealth & corporate lobby"
        " RAG knowledge base across 10 banking domains: Lobby Routing & Smart"
        " Kiosks, Cards & Personal Accounts, Savings/CDs & Sovereign Bonds,"
        " Wealth Management & Bullion, Cash/Coins & Foreign Exchange,"
        " Loans/Mortgages & Credit Cards, Statements/Certificates & Credit,"
        " Utility/Tax & Municipal Pay, Corporate & SME Banking, and"
        " Security/Accessibility & Special Care. Always call this tool before"
        " answering any banking product, rate, threshold, document, or counter"
        " routing question, and never invent facts outside returned matches."
    )
    query_desc = (
        "The client's banking question or keywords, e.g. 'R2 fixed-income"
        " wealth yield and first-time risk assessment', '3-year Large CD rate',"
        " '100k cash withdrawal booking', 'overseas tuition SWIFT wire code',"
        " 'safe deposit box rental', or 'unfreeze non-counter suspended card'."
    )
    cat_desc = (
        "Optional banking category filter hint (e.g. 'Lobby Routing & Smart"
        " Kiosks', 'Cards & Personal Accounts', 'Savings, CDs & Sovereign"
        " Bonds', 'Wealth Management & Bullion', 'Cash, Coins & Foreign"
        " Exchange', 'Loans, Mortgages & Credit Cards', 'Statements,"
        " Certificates & Credit', 'Utility, Tax & Municipal Pay', 'Corporate &"
        " SME Banking', 'Security, Accessibility & Special Care')."
    )
  else:
    desc = (
        f"Search the dedicated RAG knowledge base for {org} to retrieve"
        " grounded venue service guides, workflows, floor/counter locations,"
        " fee schedules, policy rules, and associated navigation waypoints."
        " Always call this tool before answering venue-specific policy or"
        " workflow questions, and never invent facts outside returned matches."
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
# Hybrid Tokenizer + Synonym Expansion + Salient BM25 + Optional Vector RRF
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


def _is_salient_token(tok: str) -> bool:
  """Return True when `tok` carries specific topic meaning rather than generic framing."""
  if not tok:
    return False
  if _CJK_RUN_RE.match(tok):
    if len(tok) == 1 and tok in _CJK_FRAMING_CHARS:
      return False
    if tok in _CJK_FRAMING_PHRASES:
      return False
    return True
  return tok not in _GENERIC_FRAMING_TOKENS


def _term_matches_query(term: str, q_low: str, token_set: set[str]) -> bool:
  """Check if a synonym term appears in query (whole-token for ASCII, substring for CJK)."""
  if _CJK_RUN_RE.search(term):
    return term in q_low
  if "_" in term:
    phrase = term.replace("_", " ")
    return phrase in q_low
  return term in token_set


def _expand_query_tokens(
    query: str,
    base_tokens: list[str],
    salient_only: bool = False,
) -> dict[str, float]:
  """Return weighted query terms (original = 1.0, generic = 0.2, synonyms = 0.45)."""
  weights: dict[str, float] = {}
  for tok in base_tokens:
    is_sal = _is_salient_token(tok)
    if salient_only and not is_sal:
      continue
    w = 1.0 if is_sal else 0.2
    weights[tok] = weights.get(tok, 0.0) + w

  q_low = (query or "").lower()
  token_set = (
      {t for t in base_tokens if _is_salient_token(t)}
      if salient_only
      else set(base_tokens)
  )
  for group in _SYNONYM_GROUPS:
    if any(_term_matches_query(term, q_low, token_set) for term in group):
      for syn in group:
        for st in _tokenize(syn.replace("_", " ")):
          if salient_only and not _is_salient_token(st):
            continue
          if st not in weights:
            weights[st] = 0.45
  return weights


_bm25_index_cache: dict[str, tuple[list[dict[str, float]], list[float], dict[str, int]]] = {}


def _get_or_build_bm25_index(
    docs: list[dict[str, Any]],
) -> tuple[list[dict[str, float]], list[float], dict[str, int]]:
  """Build or retrieve cached BM25 term-frequency & DF statistics for `docs`."""
  sig_raw = "|".join(f"{d.get('id', '')}:{len(str(d.get('content', '')))}" for d in docs)
  cache_key = hashlib.md5(sig_raw.encode("utf-8")).hexdigest()
  cached = _bm25_index_cache.get(cache_key)
  if cached is not None:
    return cached

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

  built = (doc_term_freqs, doc_lengths, df)
  if len(_bm25_index_cache) > 16:
    _bm25_index_cache.clear()
  _bm25_index_cache[cache_key] = built
  return built


def _bm25_scores(
    query: str,
    category: str,
    docs: list[dict[str, Any]],
    salient_only: bool = False,
) -> list[float]:
  """Compute field-weighted BM25 + exact phrase bonus across scenario chunks."""
  if not docs:
    return []
  combined_q = f"{query} {category}".strip()
  base_q_tokens = _tokenize(combined_q)
  q_weights = _expand_query_tokens(combined_q, base_q_tokens, salient_only=salient_only)
  if not q_weights:
    return [0.0 for _ in docs]

  doc_term_freqs, doc_lengths, df = _get_or_build_bm25_index(docs)

  n_docs = len(docs)
  avg_dl = sum(doc_lengths) / max(n_docs, 1)
  k1 = 1.5
  b = 0.72
  q_low = (query or "").lower().strip()
  cat_low = (category or "").lower().strip()
  base_tok_set = (
      {t for t in base_q_tokens if _is_salient_token(t)}
      if salient_only
      else set(base_q_tokens)
  )

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
    for kw in d.get("keywords") or []:
      kw_l = str(kw).lower().strip()
      if len(kw_l) < 2:
        continue
      if salient_only and kw_l in _GENERIC_FRAMING_TOKENS:
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
      if not salient_only or q_low not in _GENERIC_FRAMING_TOKENS:
        score += 4.0
    if cat_low and cat_low in str(d.get("category", "")).lower() and score > 0:
      score *= 1.15
    if d.get("is_custom") and score > 0:
      score *= 1.45
    scores.append(score)
  return scores


def _find_unmatched_query_terms(
    query: str,
    selected_indices: list[int],
    docs: list[dict[str, Any]],
) -> list[str]:
  """Identify salient visitor query terms not covered by the retrieved `matches`."""
  q_low = (query or "").lower()
  doc_term_freqs, _, _ = _get_or_build_bm25_index(docs)
  covered_doc_tokens: set[str] = set()
  for idx in selected_indices:
    covered_doc_tokens.update(doc_term_freqs[idx].keys())

  raw_ascii: list[str] = []
  raw_cjk_runs: list[str] = []
  for raw in _TOKEN_RE.findall(q_low):
    if _CJK_RUN_RE.match(raw):
      raw_cjk_runs.append(raw)
    elif raw not in _STOP_WORDS and _is_salient_token(raw):
      raw_ascii.append(raw)

  # Which synonym groups matched the query AND also hit at least one selected doc?
  ascii_set = set(raw_ascii)
  covered_groups: list[tuple[str, ...]] = []
  for grp in _SYNONYM_GROUPS:
    if any(_term_matches_query(t, q_low, ascii_set) for t in grp):
      grp_toks: set[str] = set()
      for syn in grp:
        grp_toks.update(_tokenize(syn.replace("_", " ")))
      if grp_toks & covered_doc_tokens:
        covered_groups.append(grp)

  covered_ascii_syns: set[str] = set()
  covered_cjk_phrases: list[str] = []
  for grp in covered_groups:
    for syn in grp:
      if _CJK_RUN_RE.search(syn):
        covered_cjk_phrases.append(syn)
      else:
        covered_ascii_syns.update(_tokenize(syn.replace("_", " ")))

  unmatched: list[str] = []
  for tok in raw_ascii:
    if len(tok) < 3 or tok.isdigit():
      continue
    if tok in covered_doc_tokens or tok in covered_ascii_syns:
      continue
    # Simple singular/plural stem check
    stem = tok[:-1] if tok.endswith("s") and len(tok) > 4 else tok
    if stem in covered_doc_tokens or f"{tok}s" in covered_doc_tokens:
      continue
    if tok not in unmatched:
      unmatched.append(tok)

  # CJK residual check: strip matched synonym phrases + framing phrases/chars
  covered_cjk_sorted = sorted(set(covered_cjk_phrases), key=len, reverse=True)
  for run in raw_cjk_runs:
    rem = run
    for ph in covered_cjk_sorted:
      if ph in rem:
        rem = rem.replace(ph, " ")
    for fp in _CJK_FRAMING_PHRASES:
      if fp in rem:
        rem = rem.replace(fp, " ")
    for part in rem.split():
      cleaned = "".join(ch for ch in part if ch not in _CJK_FRAMING_CHARS)
      if len(cleaned) >= 2 and cleaned not in covered_doc_tokens:
        if cleaned not in unmatched:
          unmatched.append(cleaned)
  return unmatched[:6]


def _vertex_embed_batch(texts: list[str], timeout_s: float = 1.1) -> list[list[float]] | None:
  """Embed texts via Vertex AI `text-embedding-005` (`us-central1`) with chunked cache."""
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
    batch_size = 50
    eff_timeout = 2.2 if len(missing_indices) > 5 else timeout_s
    for start in range(0, len(missing_indices), batch_size):
      sub_indices = missing_indices[start : start + batch_size]
      payload = {
          "instances": [{"content": texts[i][:1800]} for i in sub_indices]
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
      with urllib.request.urlopen(req, timeout=eff_timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
      preds = body.get("predictions") or []
      if len(preds) != len(sub_indices):
        return None
      with _embed_lock:
        for idx_pos, pred in zip(sub_indices, preds):
          vals = (pred.get("embeddings") or {}).get("values")
          if not vals:
            return None
          _embed_cache[keys[idx_pos]] = [float(x) for x in vals]
    with _embed_lock:
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
    top_k: int = 4,
) -> dict[str, Any]:
  """Execute `search_knowledge_base` and return strictly grounded chunks."""
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
        "grounded": False,
        "matches": [],
        "note": "No scenario knowledge base entries are configured in this session.",
    }

  bm25 = _bm25_scores(query, category, docs, salient_only=False)
  salient_bm25 = _bm25_scores(query, category, docs, salient_only=True)
  bm25_order = sorted(range(len(docs)), key=lambda i: (salient_bm25[i], bm25[i]), reverse=True)
  bm25_rank = {doc_idx: rank for rank, doc_idx in enumerate(bm25_order)}

  retrieval_mode = "bm25_ngram"
  cos_scores = [0.0 for _ in docs]
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
    # Reciprocal Rank Fusion (Salient BM25 weight 1.15 + Semantic Vector weight 1.0)
    fused = [
        1.15 / (12.0 + bm25_rank[i])
        + (0.015 if salient_bm25[i] > 0 else 0.0)
        + 1.0 / (12.0 + cos_rank[i])
        + 0.08 * max(cos_scores[i], 0.0)
        for i in range(len(docs))
    ]
    ranked = sorted(range(len(docs)), key=lambda i: fused[i], reverse=True)
    retrieval_mode = "hybrid_vector_bm25"
  else:
    ranked = bm25_order

  max_salient = max(salient_bm25) if salient_bm25 else 0.0
  max_cos = max(cos_scores) if cos_scores else 0.0
  top1_unmatched = _find_unmatched_query_terms(query, ranked[:1] if ranked else [], docs)
  # Reject out-of-KB queries when NO salient domain token matches any entry, OR
  # when a multi-term query only hits one weak accidental word while >=3 salient
  # terms are completely absent from the KB (unless rescued by vector >= 0.78).
  weak_accidental_overlap = len(top1_unmatched) >= 3 and max_salient < 9.5
  is_grounded = (max_salient >= 1.2 and not weak_accidental_overlap) or (
      retrieval_mode == "hybrid_vector_bm25" and max_cos >= 0.78
  )

  if not is_grounded:
    unmatched = _find_unmatched_query_terms(query, [], docs)
    latency_ms = int(round((time.monotonic() - t0) * 1000))
    return {
        "scenario": sc.get("id", DEFAULT_SCENARIO_ID),
        "badge": sc.get("badge", ""),
        "organization": sc.get("organization", ""),
        "role": sc.get("role", ""),
        "query": query,
        "grounded": False,
        "retrieval_mode": retrieval_mode,
        "latency_ms": latency_ms,
        "matches": [],
        "unmatched_query_terms": unmatched,
        "available_topics": [
            t["category"] for t in (sc.get("topic_guide") or [])
        ],
        "available_waypoints": list(sc.get("waypoints") or []),
        "instruction": (
            "NO_RAG_MATCH: The visitor's requested topic is NOT present in the"
            f" {sc.get('organization') or 'active venue'} RAG knowledge base."
            " Do NOT invent, guess, or extrapolate any policy, floor, counter,"
            " rate, or fee. Politely inform the visitor in their own language"
            " that this specific service/topic is not in the current knowledge"
            " base, mention supported service categories if helpful, or refer"
            " them to the 1F staffed reception desk / official hotline."
        ),
    }

  selected: list[dict[str, Any]] = []
  selected_indices: list[int] = []
  for rank_pos, idx in enumerate(ranked[: max(1, top_k)]):
    if rank_pos > 0:
      if (
          retrieval_mode == "bm25_ngram"
          and (salient_bm25[idx] < 1.5 or salient_bm25[idx] < 0.18 * max_salient)
      ):
        continue
      if (
          retrieval_mode == "hybrid_vector_bm25"
          and salient_bm25[idx] < 1.2
          and cos_scores[idx] < 0.72
      ):
        continue
    d = docs[idx]
    # Honest calibrated score (saturation curve rather than blind max-ratio 0.99)
    s_val = salient_bm25[idx]
    vec_bonus = 0.22 * max(cos_scores[idx] - 0.55, 0.0) if retrieval_mode == "hybrid_vector_bm25" else 0.0
    norm_score = round(min(0.99, max(0.18, (s_val / (s_val + 6.5)) + vec_bonus)), 2)
    selected_indices.append(idx)
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

  unmatched = _find_unmatched_query_terms(query, selected_indices, docs)
  latency_ms = int(round((time.monotonic() - t0) * 1000))
  unmatched_note = (
      f" Note: query terms {unmatched} do NOT appear in `matches`—do NOT assume"
      " or fabricate coverage for those unmentioned terms; clarify strictly"
      " what `matches` covers."
      if unmatched
      else ""
  )
  return {
      "scenario": sc.get("id", DEFAULT_SCENARIO_ID),
      "badge": sc.get("badge", ""),
      "organization": sc.get("organization", ""),
      "role": sc.get("role", ""),
      "query": query,
      "grounded": True,
      "retrieval_mode": retrieval_mode,
      "latency_ms": latency_ms,
      "matches": selected,
      "unmatched_query_terms": unmatched,
      "available_waypoints": list(sc.get("waypoints") or []),
      "instruction": (
          "STRICT_RAG_GROUNDING: Answer the visitor concisely in their own"
          " language using ONLY facts explicitly stated in `matches` (state"
          " exact floor/window/hours/thresholds/prep rules in 1-3 short spoken"
          " sentences; NEVER invent or extrapolate unmentioned policies)."
          + unmatched_note
          + " If the visitor wants to go there, offer to guide them or call"
          " navigate(waypoint=...)."
      ),
  }
