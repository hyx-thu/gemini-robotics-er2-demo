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
    self.assertIn("blood_draw_center_1f", items["cymbal_hospital"]["waypoints"])
    self.assertIn("vip_wealth_lounge", items["cymbal_bank"]["waypoints"])

    # Every UI-facing scenario field, persona block, and tool schema must be 100% English (zero CJK).
    catalog_json = json.dumps(items, ensure_ascii=False)
    self.assertIsNone(
        _CJK_RE.search(catalog_json),
        f"Found unexpected CJK characters in scenario catalog: {_CJK_RE.findall(catalog_json)[:10]}",
    )
    for sid in ("default", "cymbal_hospital", "cymbal_bank"):
      persona = scenario_rag.build_scenario_persona_block(sid, custom_kb="### Lobby Info\nOpen daily." if sid == "default" else "")
      tool_str = json.dumps(scenario_rag.search_knowledge_base_tool(sid), ensure_ascii=False)
      self.assertIsNone(_CJK_RE.search(persona), f"CJK found in persona for {sid}")
      self.assertIsNone(_CJK_RE.search(tool_str), f"CJK found in tool schema for {sid}")

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
        ("Where do I pay my outpatient bill and use a self-service kiosk?", {"hosp_reg_01", "hosp_reg_03", "hosp_reg_06"}),
        ("Where do I go for a blood draw and can I drink plain water during an 8-hour fast?", {"hosp_lab_02", "hosp_lab_01"}),
        ("Where do I drop off a midstream urine specimen cup?", {"hosp_lab_05"}),
        ("Can I undergo an MRI with a titanium dental implant or pacemaker?", {"hosp_img_01"}),
        ("I feel dizzy when rolling over in bed—should I see Neurology or ENT?", {"hosp_tri_02"}),
        ("Can herbal decoctions be delivered home by courier?", {"hosp_phm_02"}),
        ("Where can adults get the 9-valent HPV or shingles vaccine?", {"hosp_mpv_08"}),
        ("Can I drive home myself after a dilated retinal eye exam?", {"hosp_der_03"}),
        ("What are the daily inpatient visiting hours and caregiver pass rules?", {"hosp_inp_04"}),
        ("How fast can I get my pre-employment onboarding physical report?", {"hosp_chk_02"}),
        ("Where can I get a rabies vaccine after a cat scratch at midnight?", {"hosp_emg_03"}),
        ("请问门诊去哪里缴费？支持刷异地医保吗？", {"hosp_reg_01", "hosp_reg_03"}),
        ("去哪里抽血化验？需要空腹几个小时？", {"hosp_lab_01", "hosp_lab_02"}),
        ("中药可以代煎寄回家吗？", {"hosp_phm_02"}),
        ("拔智齿周末可以约吗？需要注意什么？", {"hosp_der_01"}),
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
        "### International VIP Clinic\n"
        "Located on 6F of the Outpatient Podium with dedicated English and Japanese medical interpreters; booking hotline 400-800-9999.\n\n"
        "### Free Metro Shuttle Schedule\n"
        "Complimentary wheelchair-accessible shuttle departs Metro Line 2 Exit A every 15 minutes straight to the 1F Main Atrium."
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
          "navigate", waypoint="blood_draw_center_1f"
      )
      self.assertEqual(nav_res["status"], "arrived")
      self.assertEqual(nav_res["destination"], "blood_draw_center_1f")

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
          "args": {"query": "where do I get a blood draw"},
      })
      return bus, sm

    bus, sm = asyncio.run(_run())
    self.assertEqual(sm.frames_sent, 0)
    self.assertEqual(len(sm.messages), 1)
    fn_resp = sm.messages[0]["toolResponse"]["functionResponses"][0]["response"]
    self.assertEqual(fn_resp["matches"][0]["id"], "hosp_lab_01")
    tool_res_events = [
        ev.data for ev in bus.events
        if ev.type == tool_call_handler.event_bus.EventType.TOOL_RESULT
    ]
    self.assertTrue(tool_res_events)
    self.assertEqual(tool_res_events[-1]["result"]["matches"][0]["id"], "hosp_lab_01")

  def test_server_scenarios_endpoints(self):
    app = server.create_app(server.ServerConfig(api_key="test-key"))
    client = testclient.TestClient(app)

    r0 = client.get("/api/scenarios")
    self.assertEqual(r0.status_code, 200)
    self.assertEqual(r0.json()["active_scenario"], "default")

    # Default mode does NOT expose search_knowledge_base
    cfg_default = client.get("/api/agent_config/spot?scenario=default").json()
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

    cfg_bank = client.get("/api/agent_config/spot").json()
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
        ("cymbal_hospital", "你们医院一楼卖高铁票和飞机票吗？"),
        ("cymbal_hospital", "Do you offer pet grooming and veterinary kennel boarding?"),
        ("cymbal_hospital", "Can I book airline hotel vacation packages at the desk?"),
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
        ("cymbal_hospital", "请问你们医院能给宠物狗打疫苗吗？", "宠物狗"),
        ("cymbal_hospital", "Do you perform cosmetic facelift surgery at the clinic?", "facelift"),
        ("cymbal_bank", "Can I open a cryptocurrency debit card at a Smart Teller Machine?", "cryptocurrency"),
    ]
    for sid, query, expected_unmatched in partial_cases:
      with self.subTest(sid=sid, query=query):
        res = scenario_rag.run_rag_tool({"query": query}, sid)
        self.assertTrue(res["grounded"])
        self.assertIn(expected_unmatched, res["unmatched_query_terms"])
        self.assertIn("do NOT appear in `matches`", res["instruction"])


if __name__ == "__main__":
  unittest.main()
