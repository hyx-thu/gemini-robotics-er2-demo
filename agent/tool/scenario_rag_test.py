"""Unit tests for Scenario Persona, Hybrid RAG retriever, waypoints, and API."""

from __future__ import annotations

import asyncio
import json
import os
import re
import unittest
from unittest import mock

from fastapi import testclient

import server
from core import tool_call_handler
from embodiment.spot import spot_sim_embodiment
from tool import scenario_rag

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


class ScenarioRagTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._orig_embed_env = os.environ.get("SCENARIO_RAG_EMBEDDINGS")
    os.environ["SCENARIO_RAG_EMBEDDINGS"] = "0"
    scenario_rag._scenarios_cache = None  # pylint: disable=protected-access

  def tearDown(self):
    if self._orig_embed_env is None:
      os.environ.pop("SCENARIO_RAG_EMBEDDINGS", None)
    else:
      os.environ["SCENARIO_RAG_EMBEDDINGS"] = self._orig_embed_env
    super().tearDown()

  def test_catalog_contains_hospital_and_bank_pure_english(self):
    items = {s["id"]: s for s in scenario_rag.list_scenarios()}
    self.assertIn("default", items)
    self.assertIn("cymbal_hospital", items)
    self.assertIn("cymbal_bank", items)
    self.assertEqual(items["default"]["doc_count"], 0)
    self.assertEqual(items["cymbal_hospital"]["doc_count"], 120)
    self.assertEqual(items["cymbal_bank"]["doc_count"], 120)
    self.assertEqual(len(items["cymbal_hospital"]["topic_guide"]), 10)
    self.assertEqual(len(items["cymbal_bank"]["topic_guide"]), 10)
    self.assertIn("welcome_desk_1f", items["cymbal_hospital"]["waypoints"])
    self.assertIn("quiet_comfort_lounge_1f", items["cymbal_hospital"]["waypoints"])
    self.assertIn("healing_garden_4f", items["cymbal_hospital"]["waypoints"])
    self.assertIn("vip_wealth_lounge", items["cymbal_bank"]["waypoints"])

    # Every UI-facing scenario field, persona block, and tool schema must be 100% English (zero CJK).
    catalog_json = json.dumps(items, ensure_ascii=False)
    self.assertIsNone(
        _CJK_RE.search(catalog_json),
        f"Found unexpected CJK characters in scenario catalog: {_CJK_RE.findall(catalog_json)[:10]}",
    )
    for sid in ("default", "cymbal_hospital", "cymbal_bank"):
      persona = scenario_rag.build_scenario_persona_block(
          sid,
          custom_kb="### Lobby Info\nOpen daily." if sid == "default" else "",
      )
      tool_str = json.dumps(scenario_rag.search_knowledge_base_tool(sid), ensure_ascii=False)
      self.assertIsNone(_CJK_RE.search(persona), f"CJK found in persona for {sid}")
      self.assertIsNone(_CJK_RE.search(tool_str), f"CJK found in tool schema for {sid}")

  def test_hospital_poc_strictly_non_medical_visit_and_comfort_only(self):
    """Enforce zero clinical/medical treatment/prep content in Hospital PoC."""
    hosp_sc = scenario_rag.get_scenario("cymbal_hospital")
    expected_categories = [
        "Check-In & Visit Flow",
        "Campus & Floor Wayfinding",
        "Billing, Insurance & Forms",
        "Anxiety Relief & Calm Spaces",
        "Waiting Comfort & Family Care",
        "Accessibility & Senior Support",
        "Language & Guest Concierge",
        "Dining, Lounges & Amenities",
        "Parking, Transit & Drop-Off",
        "Safety, Privacy & Help Desks",
    ]
    actual_categories = [t["category"] for t in hosp_sc["topic_guide"]]
    self.assertEqual(actual_categories, expected_categories)

    hosp_bundle = (
        json.dumps(hosp_sc, ensure_ascii=False).lower()
        + " "
        + json.dumps(scenario_rag.search_knowledge_base_tool("cymbal_hospital"), ensure_ascii=False).lower()
        + " "
        + scenario_rag.build_scenario_persona_block("cymbal_hospital").lower()
    )
    forbidden_clinical_terms = (
        "phlebotomy",
        "venipuncture",
        "blood draw",
        "fasting",
        "urinalysis",
        "urine",
        "stool",
        "sputum",
        "swab",
        "biopsy",
        "pathology",
        "mri",
        "ultrasound",
        "mammogram",
        "endoscopy",
        "colonoscopy",
        "gastroscopy",
        "sedation",
        "anesthesia",
        "contrast",
        "xray",
        "ecg",
        "holter",
        "vaccine",
        "vaccination",
        "immunization",
        "rabies",
        "tetanus",
        "yellow fever",
        "malaria",
        "typhoid",
        "hpv",
        "shingles",
        "prescription",
        "compounding",
        "infusion",
        "biologics",
        "insulin",
        "metformin",
        "antibiotic",
        "antihistamine",
        "anticoagulation",
        "chemotherapy",
        "extraction",
        "cataract",
        "dilated",
        "acupuncture",
        "cryotherapy",
        "suture",
        "picc",
        "ostomy",
        "tumor",
        "stemi",
        "stroke",
        "fracture",
        "laceration",
        "vertigo",
        "hypertension",
        "asthma",
        "arthritis",
        "eczema",
        "diabetes",
        "superbill",
        "icd-10",
        "cpt",
    )
    for term in forbidden_clinical_terms:
      self.assertIsNone(
          re.search(rf"\b{re.escape(term)}\b", hosp_bundle),
          f"Forbidden clinical/medical term {term!r} found in non-medical Hospital PoC bundle",
      )

    ui_js_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "ui",
        "agent-ui.js",
    )
    with open(ui_js_path, "r", encoding="utf-8") as f:
      ui_js = f.read().lower()
    for ui_forbidden in ("mri titanium", "yellow fever", "pre-exam preparation"):
      self.assertNotIn(ui_forbidden, ui_js)

  def test_default_mode_zero_overhead(self):
    self.assertFalse(scenario_rag.is_rag_active("default", ""))
    self.assertEqual(scenario_rag.build_scenario_persona_block("default"), "")
    self.assertIsNone(scenario_rag.get_scenario_waypoints("default"))

  def test_strict_scenario_isolation_no_cross_domain_terms(self):
    hosp_full = (
        json.dumps(scenario_rag.get_scenario("cymbal_hospital"), ensure_ascii=False).lower()
        + " "
        + json.dumps(scenario_rag.search_knowledge_base_tool("cymbal_hospital"), ensure_ascii=False).lower()
        + " "
        + scenario_rag.build_scenario_persona_block("cymbal_hospital").lower()
    )
    for bank_term in ("wealth", "mortgage", "swift", "teller", "atm", "bullion"):
      self.assertIsNone(
          re.search(rf"\b{re.escape(bank_term)}\b", hosp_full),
          f"Unexpected bank word {bank_term!r} found in Cymbal Hospital scenario",
      )
    for bank_cjk in ("理财", "大额存单"):
      self.assertNotIn(bank_cjk, hosp_full)

    bank_full = (
        json.dumps(scenario_rag.get_scenario("cymbal_bank"), ensure_ascii=False).lower()
        + " "
        + json.dumps(scenario_rag.search_knowledge_base_tool("cymbal_bank"), ensure_ascii=False).lower()
        + " "
        + scenario_rag.build_scenario_persona_block("cymbal_bank").lower()
    )
    for hosp_term in ("hospital", "patient", "phlebotomy", "triage", "pharmacy", "mri", "endoscopy", "rabies"):
      self.assertIsNone(
          re.search(rf"\b{re.escape(hosp_term)}\b", bank_full),
          f"Unexpected hospital word {hosp_term!r} found in Cymbal Bank scenario",
      )
    for hosp_cjk in ("义诊", "抽血"):
      self.assertNotIn(hosp_cjk, bank_full)

  def test_hospital_rag_queries_english_and_chinese(self):
    cases = [
        ("How do I scan my QR booking pass at the self-service arrival check-in kiosks?", {"hosp_chk_02"}),
        ("What happens if traffic makes me late or I miss my queue number?", {"hosp_chk_05"}),
        ("Where is the priority fast-track check-in counter for seniors aged 65+ and expectant mothers?", {"hosp_chk_07"}),
        ("Can you explain what facilities are on each floor from Level B1 to 6F?", {"hosp_nav_01"}),
        ("How do the Ocean Blue, Forest Green, and Warm Amber floor stripes guide me?", {"hosp_nav_02"}),
        ("Which elevator bank serves Basement Parking B1–B3 versus the 6F–18F Tower?", {"hosp_nav_03"}),
        ("Do you offer cashless direct-billing GOP paperwork for Cigna, Bupa, or Allianz insurance?", {"hosp_bil_03"}),
        ("Where can I get a stamped administrative proof-of-attendance slip showing my visit hours?", {"hosp_bil_07"}),
        ("I feel really nervous about being in a hospital today—how can I settle in calmly?", {"hosp_anx_01"}),
        ("Can you guide me through a slow 4-4-4-4 box-breathing relaxation exercise while I wait?", {"hosp_anx_02"}),
        ("Is there a silent, dimmed-lighting Quiet Comfort Lounge away from busy lobby noise?", {"hosp_anx_03"}),
        ("Where is the 4F Botanical Healing Atrium if I want natural sunlight and greenery?", {"hosp_anx_04", "hosp_nav_10"}),
        ("Can I borrow a vibrating coaster pager so I will not miss my turn while walking in the garden?", {"hosp_anx_07"}),
        ("I came alone today and feel overwhelmed—can a hospitality volunteer walk with me?", {"hosp_anx_08"}),
        ("Where is the 3F Kids Storybook and Play Corner with coloring packs?", {"hosp_fam_02"}),
        ("Where are the private lockable nursing and lactation suites with bottle warmers?", {"hosp_fam_04"}),
        ("Is there a 24/7 multi-faith prayer, meditation, and quiet reflection room?", {"hosp_fam_09"}),
        ("Where can I borrow a free UV-sanitized wheelchair or rollator with zero deposit?", {"hosp_acc_01"}),
        ("Can I borrow a live ASL sign-language video interpretation tablet at the lobby desk?", {"hosp_acc_04"}),
        ("Are certified guide dogs welcome inside public lounges, and where is the outdoor relief station?", {"hosp_acc_09"}),
        ("How do I request complimentary visit interpretation in Spanish, Arabic, Mandarin, or Japanese?", {"hosp_lng_01"}),
        ("Where can out-of-town visitors store suitcases in free day-luggage lockers?", {"hosp_lng_05"}),
        ("Which counters in the B1 Garden Dining Court serve certified Halal, Kosher-sealed, and vegan meals?", {"hosp_amn_03", "hosp_lng_09"}),
        ("How do I connect to the free Cymbal-Guest Wi-Fi without an SMS code or password?", {"hosp_amn_06"}),
        ("How do I validate my check-in QR pass for the 3-hour discounted $4 visitor parking rate?", {"hosp_trn_02"}),
        ("Where in the underground garage are the 24 EV charging bays located?", {"hosp_trn_03"}),
        ("What should I do if a family member suddenly feels unwell in the lobby right now?", {"hosp_hlp_01"}),
        ("Where do I speak with licensed staff if I have a question about health concerns, test prep, or medications?", {"hosp_hlp_02"}),
        ("请问怎么在自助签到机扫码报到？", {"hosp_chk_02"}),
        ("支持国际商业保险 Cigna 或 Bupa 免现金直付担保函吗？", {"hosp_bil_03"}),
        ("我第一次来医院心里很紧张焦虑，有安静休息区或者深呼吸放松引导吗？", {"hosp_anx_01", "hosp_anx_02", "hosp_anx_03"}),
        ("4楼植物中庭疗愈花园怎么走？怕过号可以借震动叫号器吗？", {"hosp_anx_07", "hosp_anx_04", "hosp_nav_10"}),
        ("一个人独自来院有点慌，可以申请陪行志愿者陪我走吗？", {"hosp_anx_08"}),
        ("去哪里免费借轮椅或助行器？需要交押金吗？", {"hosp_acc_01"}),
        ("怎么预约西语、阿语或中文多语种口译引导？大件行李箱可以寄存吗？", {"hosp_lng_01", "hosp_lng_05"}),
        ("B1花园餐厅有清真餐、犹太洁食和纯素餐窗口吗？", {"hosp_amn_03", "hosp_lng_09"}),
        ("地下停车场扫码减免停车费怎么弄？B2层有新能源充电桩吗？", {"hosp_trn_02", "hosp_trn_03"}),
        ("我现在突然头晕身体不舒服应该找谁求助？", {"hosp_hlp_01"}),
        ("我想问看病挂哪个科或者吃药咨询和检查准备要求，应该去哪问人工？", {"hosp_hlp_02"}),
    ]
    for query, allowed_top_ids in cases:
      with self.subTest(query=query):
        res = scenario_rag.run_rag_tool({"query": query}, "cymbal_hospital")
        self.assertNotIn("error", res)
        ids = [m["id"] for m in res["matches"]]
        self.assertTrue(ids, f"No matches returned for {query}")
        self.assertIn(
            ids[0],
            allowed_top_ids,
            f"Expected one of {allowed_top_ids} first for {query!r}, got {ids}",
        )

  def test_hospital_rag_international_standards_no_china_only_terms(self):
    hosp_full = (
        json.dumps(scenario_rag.get_scenario("cymbal_hospital"), ensure_ascii=False).lower()
        + " "
        + json.dumps(scenario_rag.search_knowledge_base_tool("cymbal_hospital"), ensure_ascii=False).lower()
        + " "
        + scenario_rag.build_scenario_persona_block("cymbal_hospital").lower()
    )
    china_only_terms = (
        "out-of-province",
        "cross-region",
        "12123",
        "hu-gong",
        "accompany-free",
        "400-820",
        "red prescription",
        "dual-channel",
        "dual channel",
        "civil-service",
        "food-handler",
        "moxibustion",
        "tuina",
        "37.3",
        "fever clinic",
        "fiscal",
    )
    for term in china_only_terms:
      self.assertNotIn(
          term,
          hosp_full,
          f"Found China-only healthcare term {term!r} in internationalized Cymbal Hospital scenario",
      )

  def test_bank_rag_queries_english_and_chinese(self):
    cases = [
        ("What tasks can I finish in 3 minutes at a Smart Teller Machine STM?", {"bank_lob_02"}),
        ("I forgot my debit card PIN and locked it—how do I reset it?", {"bank_crd_04"}),
        ("What are the 3-year Large-Denomination CD rates and can CDs be transferred?", {"bank_dep_02", "bank_dep_01"}),
        ("What R2 low-volatility fixed-income wealth products and yields do you offer?", {"bank_wlt_03", "bank_wlt_02", "bank_wlt_01"}),
        ("Do I need an advance appointment to withdraw 100,000 in cash?", {"bank_cfx_02"}),
        ("What is Cymbal Bank's SWIFT code for an overseas university tuition wire?", {"bank_cfx_08", "bank_cfx_09"}),
        ("What is the first-home mortgage rate and is there an early repayment fee?", {"bank_lon_01", "bank_lon_03"}),
        ("Where can I print stamped 5-year bank statements or a visa solvency certificate?", {"bank_doc_01", "bank_doc_02"}),
        ("Where do I pay electricity, water, and winter heating bills?", {"bank_utl_01"}),
        ("What documents are needed to open a Corporate Basic Settlement Account?", {"bank_cor_01"}),
        ("What sizes and rental fees are available for Level B1 Safe Deposit Boxes?", {"bank_sec_03"}),
        ("Can you arrange doorstep banking for a bedridden senior?", {"bank_sec_04"}),
        ("请问稳健型理财产品收益率多少？首次风险测评去哪做？", {"bank_wlt_03", "bank_wlt_01", "bank_wlt_02"}),
        ("取十万现金需要提前预约吗？", {"bank_cfx_02"}),
        ("银行卡显示非柜面交易中止怎么解冻？", {"bank_sec_01"}),
    ]
    for query, allowed_top_ids in cases:
      with self.subTest(query=query):
        res = scenario_rag.run_rag_tool({"query": query}, "cymbal_bank")
        self.assertNotIn("error", res)
        ids = [m["id"] for m in res["matches"]]
        self.assertTrue(ids, f"No matches returned for {query}")
        self.assertIn(
            ids[0],
            allowed_top_ids,
            f"Expected one of {allowed_top_ids} first for {query!r}, got {ids}",
        )

  def test_custom_kb_parsing_and_retrieval(self):
    custom_md = (
        "### Evening Acoustic Harp Hour\n"
        "Soft live acoustic harp music plays in the 4F Botanical Healing Atrium every Friday 15:00–16:00; concierge hotline +1-800-555-9999.\n\n"
        "### Free Metro Shuttle Schedule\n"
        "Complimentary wheelchair-accessible shuttle departs Metro Line 2 Exit A every 15 minutes straight to the 1F South Canopy."
    )
    chunks = scenario_rag.parse_custom_kb(custom_md, "cymbal_hospital")
    self.assertEqual(len(chunks), 2)
    self.assertEqual(chunks[0]["category"], "Custom Knowledge")
    res = scenario_rag.run_rag_tool(
        {"query": "Is there a free shuttle bus from the metro station?"},
        "cymbal_hospital",
        custom_kb=custom_md,
    )
    self.assertEqual(res["matches"][0]["title"], "Free Metro Shuttle Schedule")
    self.assertIn("15 minutes", res["matches"][0]["content"])

  def test_hybrid_vector_rrf_reranking(self):
    os.environ["SCENARIO_RAG_EMBEDDINGS"] = "1"
    sc = scenario_rag.get_scenario("cymbal_bank")
    n_docs = len(sc["documents"])
    target_idx = next(
        i for i, d in enumerate(sc["documents"]) if d["id"] == "bank_wlt_03"
    )
    fake_vecs = [[1.0, 0.0]]  # query vector
    for i in range(n_docs):
      fake_vecs.append([0.99, 0.01] if i == target_idx else [0.0, 1.0])
    with mock.patch.object(scenario_rag, "_vertex_embed_batch", return_value=fake_vecs):
      res = scenario_rag.run_rag_tool({"query": "how to grow idle savings safely"}, "cymbal_bank")
      self.assertEqual(res["retrieval_mode"], "hybrid_vector_bm25")
      self.assertEqual(res["matches"][0]["id"], "bank_wlt_03")

  def test_embodiment_scenario_waypoints_and_tool_handler(self):
    wps = scenario_rag.get_scenario_waypoints("cymbal_hospital")
    emb = spot_sim_embodiment.SimulatedSpotEmbodiment(waypoints=wps)

    async def _run():
      wp_res = await emb.execute_action("get_waypoints")
      self.assertEqual([w["name"] for w in wp_res["waypoints"]], wps)
      nav_res = await emb.execute_action(
          "navigate", waypoint="quiet_comfort_lounge_1f"
      )
      self.assertEqual(nav_res["status"], "arrived")
      self.assertEqual(nav_res["destination"], "quiet_comfort_lounge_1f")

      class _FakeBus:

        def __init__(self):
          self.events = []

        def subscribe(self, _types, _handler):
          return None

        async def publish(self, ev):
          self.events.append(ev)

      class _FakeSession:

        def __init__(self):
          self.scenario_id = "cymbal_hospital"
          self.scenario_custom_kb = ""
          self.messages = []
          self.frames_sent = 0
          self.ui_notifications = []

        async def send_message(self, msg):
          self.messages.append(msg)

        async def send_latest_video_frame(self):
          self.frames_sent += 1
          return True

        def notify_ui(self, payload):
          self.ui_notifications.append(payload)

      bus = _FakeBus()
      sm = _FakeSession()
      handler = tool_call_handler.ToolCallHandler(
          bus=bus,
          session_manager=sm,
          embodiment=emb,
          blocking_tools={"search_knowledge_base"},
      )
      await handler._execute_single_tool({
          "id": "call-rag-1",
          "name": "search_knowledge_base",
          "args": {"query": "where is the quiet comfort lounge with dimmed lights"},
      })
      return bus, sm

    bus, sm = asyncio.run(_run())
    self.assertEqual(sm.frames_sent, 0)
    self.assertEqual(len(sm.messages), 1)
    fn_resp = sm.messages[0]["toolResponse"]["functionResponses"][0]["response"]
    self.assertEqual(fn_resp["matches"][0]["id"], "hosp_anx_03")
    tool_res_events = [
        ev.data for ev in bus.events
        if ev.type == tool_call_handler.event_bus.EventType.TOOL_RESULT
    ]
    self.assertTrue(tool_res_events)
    self.assertEqual(tool_res_events[-1]["result"]["matches"][0]["id"], "hosp_anx_03")

  def test_server_scenarios_endpoints(self):
    app = server.create_app(server.ServerConfig(api_key="test-key"))
    client = testclient.TestClient(app)

    r0 = client.get("/api/scenarios")
    self.assertEqual(r0.status_code, 200)
    self.assertEqual(r0.json()["active_scenario"], "default")

    # Default mode does NOT expose search_knowledge_base
    cfg_default = client.get("/api/agent_config/robot?scenario=default").json()
    tool_names_default = [t["name"] for t in cfg_default["tools"]]
    self.assertNotIn("search_knowledge_base", tool_names_default)
    self.assertIn("talk_motion", tool_names_default)

    # Updating scenario to cymbal_bank exposes search_knowledge_base
    r1 = client.post(
        "/api/scenarios",
        json={
            "scenario_id": "cymbal_bank",
            "custom_kb": "### Saturday Wealth Seminar\nEvery Saturday at 14:00 in the 1F South VIP Lounge: gold allocation & retirement planning briefing.",
        },
    )
    self.assertEqual(r1.status_code, 200)
    self.assertEqual(r1.json()["active_scenario"], "cymbal_bank")
    self.assertEqual(r1.json()["custom_doc_count"], 1)

    cfg_bank = client.get("/api/agent_config/robot").json()
    tool_names_bank = [t["name"] for t in cfg_bank["tools"]]
    self.assertIn("search_knowledge_base", tool_names_bank)

  def test_no_standalone_topic_guide_and_all_240_questions_rank_first(self):
    with open(scenario_rag._KNOWLEDGE_FILE, "r", encoding="utf-8") as f:  # pylint: disable=protected-access
      raw_file = json.load(f)
    for sc in raw_file["scenarios"]:
      self.assertNotIn(
          "topic_guide",
          sc,
          f"Standalone topic_guide must not exist in scenarios.json for {sc['id']}",
      )

    catalog = {s["id"]: s for s in scenario_rag.list_scenarios()}
    for sid in ("cymbal_hospital", "cymbal_bank"):
      sc = catalog[sid]
      self.assertEqual(len(sc["documents"]), 120)
      self.assertEqual(len(sc["topic_guide"]), 10)
      all_doc_ids = {d["id"] for d in sc["documents"]}
      derived_prompt_ids = set()
      for t in sc["topic_guide"]:
        self.assertEqual(t["doc_count"], 12)
        self.assertEqual(len(t["grounded_prompts"]), 12)
        for gp in t["grounded_prompts"]:
          self.assertIn(gp["doc_id"], all_doc_ids)
          derived_prompt_ids.add(gp["doc_id"])
      self.assertEqual(derived_prompt_ids, all_doc_ids)

      # Verify 100% 1-to-1 Rank-#1 retrieval across all 120 grounded questions
      for doc in sc["documents"]:
        q = doc["sample_question"]
        self.assertTrue(q, f"Missing sample_question on {doc['id']}")
        res = scenario_rag.run_rag_tool({"query": q}, sid)
        self.assertTrue(res["grounded"], f"Expected grounded=True for {doc['id']}: {q}")
        self.assertEqual(
            res["matches"][0]["id"],
            doc["id"],
            f"Expected {doc['id']} at Rank #1 for {q!r}, got {[m['id'] for m in res['matches']]}",
        )

  def test_out_of_kb_rejection_and_partial_overlap_guardrails(self):
    out_of_kb_cases = [
        ("cymbal_hospital", "你们医院一楼卖高铁票和电影票吗？"),
        ("cymbal_hospital", "Do you offer pet grooming and veterinary kennel boarding?"),
        ("cymbal_hospital", "Can I book Broadway concert tickets and cruise ship cabins at the desk?"),
        ("cymbal_bank", "Can I buy Bitcoin and Ethereum NFTs at the counter?"),
        ("cymbal_bank", "请问你们银行网点可以办理结婚证和护照签证吗？"),
        ("cymbal_bank", "Do you provide dry cleaning and car wash services in the lobby?"),
    ]
    for sid, query in out_of_kb_cases:
      with self.subTest(sid=sid, query=query):
        res = scenario_rag.run_rag_tool({"query": query}, sid)
        self.assertFalse(res["grounded"], f"Expected grounded=False for {query!r}, got {res}")
        self.assertEqual(res["matches"], [])
        self.assertIn("NO_RAG_MATCH", res["instruction"])

    partial_cases = [
        ("cymbal_hospital", "请问你们医院能给宠物狗办访客证吗？", "宠物狗"),
        ("cymbal_hospital", "Do you offer luxury yacht charter at the underground parking garage?", "yacht"),
        ("cymbal_hospital", "Can I book helicopter sightseeing tours at the partner hotel desk?", "helicopter"),
        ("cymbal_bank", "Can I open a cryptocurrency debit card at a Smart Teller Machine?", "cryptocurrency"),
    ]
    for sid, query, expected_unmatched in partial_cases:
      with self.subTest(sid=sid, query=query):
        res = scenario_rag.run_rag_tool({"query": query}, sid)
        self.assertTrue(res["grounded"])
        self.assertIn(expected_unmatched, res["unmatched_query_terms"])
        self.assertIn("do NOT appear in `matches`", res["instruction"])

  def test_hospital_persona_playful_and_quiet_on_medical_unless_asked(self):
    hosp = scenario_rag.get_scenario("cymbal_hospital")
    persona = scenario_rag.build_scenario_persona_block("cymbal_hospital")
    tool_decl = json.dumps(
        scenario_rag.search_knowledge_base_tool("cymbal_hospital"),
        ensure_ascii=False,
    ).lower()

    # Role, subtitle, and tool declaration must not prime 'non-medical PoC' disclaimers.
    for meta_val in (hosp["subtitle"], hosp["role"], tool_decl):
      low = meta_val.lower()
      self.assertNotIn("non-medical", low)
      self.assertNotIn("non-clinical", low)
      self.assertNotIn("poc", low.split())

    # Persona must specify playful/witty demeanor, quiet-by-default on medical disclaimers,
    # and one-time light offer to book/connect with staff & guide to the place ONLY when asked.
    p_low = persona.lower()
    self.assertIn("playful", p_low)
    self.assertIn("witty", p_low)
    self.assertIn("never bring up medical disclaimers", p_low)
    self.assertIn("only when a visitor specifically asks", p_low)
    self.assertIn("welcome_desk_1f", p_low)

    # All 119 ordinary Hospital RAG entries (everything except hosp_hlp_02) must have ZERO
    # 'non-medical' / 'non-clinical' / 'medical advice' / 'medical consultation' disclaimers.
    for doc in hosp["documents"]:
      if doc["id"] == "hosp_hlp_02":
        continue
      blob = json.dumps(doc, ensure_ascii=False).lower()
      for disclaimer_term in ("non-medical", "non-clinical", "medical advice", "medical consultation"):
        self.assertNotIn(
            disclaimer_term,
            blob,
            f"Unexpected disclaimer {disclaimer_term!r} in ordinary entry {doc['id']}",
        )

    # Specific medical-service questions in Chinese & English hit hosp_hlp_02 at Rank #1
    # and return practical booking + wayfinding handoff content.
    for med_q in (
        "我嗓子疼发烧应该吃什么药？",
        "请问感冒挂哪个科？能帮我做医疗咨询吗？",
        "Do you provide medical consultations or medication advice?",
    ):
      res = scenario_rag.run_rag_tool({"query": med_q}, "cymbal_hospital")
      self.assertTrue(res["grounded"], f"Expected hosp_hlp_02 match for {med_q!r}")
      self.assertEqual(res["matches"][0]["id"], "hosp_hlp_02")
      self.assertIn("welcome_desk_1f", res["matches"][0]["content"])


if __name__ == "__main__":
  unittest.main()
