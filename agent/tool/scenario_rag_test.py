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
    self.assertGreaterEqual(items["cymbal_hospital"]["doc_count"], 15)
    self.assertGreaterEqual(items["cymbal_bank"]["doc_count"], 15)
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
    for bank_term in ("wealth", "mortgage", "swift", "teller", "atm", "bullion", "理财", "大额存单"):
      self.assertNotIn(bank_term.lower(), hosp_full)

    bank_full = (
        json.dumps(scenario_rag.get_scenario("cymbal_bank"), ensure_ascii=False).lower()
        + " "
        + json.dumps(scenario_rag.search_knowledge_base_tool("cymbal_bank"), ensure_ascii=False).lower()
        + " "
        + scenario_rag.build_scenario_persona_block("cymbal_bank").lower()
    )
    for hosp_term in ("hospital", "patient", "phlebotomy", "triage", "pharmacy", "mri", "endoscopy", "rabies", "义诊", "抽血"):
      self.assertNotIn(hosp_term.lower(), bank_full)

  def test_hospital_rag_queries_english_and_chinese(self):
    cases = [
        ("Where do I pay my outpatient bill and print an invoice?", "hosp_billing"),
        ("Where do I go for a blood draw and how many hours should I fast?", "hosp_blood_draw"),
        ("Where do I drop off a urine sample?", "hosp_specimen_drop"),
        ("Can I wear a metal watch or pacemaker into the MRI room?", "hosp_imaging_mri"),
        ("Which clinic should I register for dizziness and chest tightness?", "hosp_symptom_triage"),
        ("Where is the pharmacy and can herbal decoctions be delivered home?", "hosp_pharmacy"),
        ("What floor is the pre-employment health checkup center on?", "hosp_checkup"),
        ("Where can I get a rabies vaccine after a cat scratch?", "hosp_emergency"),
        ("请问去哪里缴费？支持刷医保码吗？", "hosp_billing"),
        ("去哪里抽血化验？需要空腹几个小时？", "hosp_blood_draw"),
        ("交完钱去哪里取药？中药可以代煎寄回家吗？", "hosp_pharmacy"),
    ]
    for query, expected_id in cases:
      with self.subTest(query=query):
        res = scenario_rag.run_rag_tool({"query": query}, "cymbal_hospital")
        self.assertNotIn("error", res)
        ids = [m["id"] for m in res["matches"]]
        self.assertTrue(ids, f"No matches returned for {query}")
        self.assertEqual(
            ids[0],
            expected_id,
            f"Expected {expected_id} first for {query!r}, got {ids}",
        )

  def test_bank_rag_queries_english_and_chinese(self):
    cases = [
        ("What low-volatility wealth management products do you offer and what is the yield?", "bank_wealth_products"),
        ("What are your 3-year term deposit and CD interest rates?", "bank_deposits_bonds"),
        ("Where can I pay my electricity, water, and gas utility bills?", "bank_bill_payment"),
        ("Do I need an advance appointment to withdraw 100,000 in cash?", "bank_cash_withdrawal"),
        ("What is your SWIFT code for an overseas university tuition wire transfer?", "bank_forex_wire"),
        ("Where can I print bank statements and a visa solvency certificate?", "bank_certificates"),
        ("What is the first-home mortgage rate and how does early repayment work?", "bank_loans_mortgage"),
        ("Can you arrange a home visit for a bedridden senior who forgot their PIN?", "bank_accessibility"),
        ("请问有什么稳健型的理财产品？收益率多少？", "bank_wealth_products"),
        ("我要取十万现金需要提前预约吗？", "bank_cash_withdrawal"),
    ]
    for query, expected_id in cases:
      with self.subTest(query=query):
        res = scenario_rag.run_rag_tool({"query": query}, "cymbal_bank")
        self.assertNotIn("error", res)
        ids = [m["id"] for m in res["matches"]]
        self.assertTrue(ids, f"No matches returned for {query}")
        self.assertEqual(
            ids[0],
            expected_id,
            f"Expected {expected_id} first for {query!r}, got {ids}",
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
        i for i, d in enumerate(sc["documents"]) if d["id"] == "bank_wealth_products"
    )
    fake_vecs = [[1.0, 0.0]]  # query vector
    for i in range(n_docs):
      fake_vecs.append([0.99, 0.01] if i == target_idx else [0.0, 1.0])
    with mock.patch.object(scenario_rag, "_vertex_embed_batch", return_value=fake_vecs):
      res = scenario_rag.run_rag_tool({"query": "how to grow idle savings safely"}, "cymbal_bank")
      self.assertEqual(res["retrieval_mode"], "hybrid_vector_bm25")
      self.assertEqual(res["matches"][0]["id"], "bank_wealth_products")

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
    self.assertEqual(fn_resp["matches"][0]["id"], "hosp_blood_draw")
    tool_res_events = [
        ev.data for ev in bus.events
        if ev.type == tool_call_handler.event_bus.EventType.TOOL_RESULT
    ]
    self.assertTrue(tool_res_events)
    self.assertEqual(tool_res_events[-1]["result"]["matches"][0]["id"], "hosp_blood_draw")

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


if __name__ == "__main__":
  unittest.main()
