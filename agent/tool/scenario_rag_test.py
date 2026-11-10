"""Unit tests for Scenario Persona, Hybrid RAG retriever, waypoints, and API."""

from __future__ import annotations

import asyncio
import os
import unittest
from unittest import mock

from fastapi import testclient

import server
from core import tool_call_handler
from embodiment.spot import spot_sim_embodiment
from tool import scenario_rag


class ScenarioRagTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._orig_embed_env = os.environ.get("SCENARIO_RAG_EMBEDDINGS")
    os.environ["SCENARIO_RAG_EMBEDDINGS"] = "0"

  def tearDown(self):
    if self._orig_embed_env is None:
      os.environ.pop("SCENARIO_RAG_EMBEDDINGS", None)
    else:
      os.environ["SCENARIO_RAG_EMBEDDINGS"] = self._orig_embed_env
    super().tearDown()

  def test_catalog_contains_hospital_and_bank(self):
    items = {s["id"]: s for s in scenario_rag.list_scenarios()}
    self.assertIn("default", items)
    self.assertIn("cymbal_hospital", items)
    self.assertIn("cymbal_bank", items)
    self.assertEqual(items["default"]["doc_count"], 0)
    self.assertGreaterEqual(items["cymbal_hospital"]["doc_count"], 15)
    self.assertGreaterEqual(items["cymbal_bank"]["doc_count"], 15)
    self.assertIn("blood_draw_center_1f", items["cymbal_hospital"]["waypoints"])
    self.assertIn("vip_wealth_lounge", items["cymbal_bank"]["waypoints"])

  def test_default_mode_zero_overhead(self):
    self.assertFalse(scenario_rag.is_rag_active("default", ""))
    self.assertEqual(scenario_rag.build_scenario_persona_block("default"), "")
    self.assertIsNone(scenario_rag.get_scenario_waypoints("default"))

  def test_hospital_rag_queries(self):
    cases = [
        ("请问去哪里缴费？支持刷医保码吗？", "hosp_billing"),
        ("去哪里抽血化验？需要空腹几个小时？", "hosp_blood_draw"),
        ("尿常规和大便样本去哪里留样？", "hosp_specimen_drop"),
        ("做核磁共振能带手机和手表吗？", "hosp_imaging_mri"),
        ("我最近头晕胸闷应该挂什么科？", "hosp_symptom_triage"),
        ("交完钱去哪里取药？中药可以代煎寄回家吗？", "hosp_pharmacy"),
        ("入职体检在几楼？几点截止采血？", "hosp_checkup"),
        ("被猫抓伤了去哪里打狂犬疫苗？", "hosp_emergency"),
        ("Where do I pay my bill and print an invoice?", "hosp_billing"),
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

  def test_bank_rag_queries(self):
    cases = [
        ("请问有什么稳健型的理财产品？收益率多少？", "bank_wealth_products"),
        ("现在定期存款和大额存单利率是多少？", "bank_deposits_bonds"),
        ("水电燃气费和社保去哪里缴费？", "bank_bill_payment"),
        ("我要取十万现金需要提前预约吗？", "bank_cash_withdrawal"),
        ("给国外大学汇学费需要什么材料？SWIFT代码是多少？", "bank_forex_wire"),
        ("买房打银行流水和签证存款证明去哪里办？", "bank_certificates"),
        ("首套房贷利率和提前还款怎么办理？", "bank_loans_mortgage"),
        ("家里老人卧床忘记银行卡密码可以上门办理吗？", "bank_accessibility"),
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
        "### 国际医疗部特需门诊\n"
        "国际医疗部位于门诊裙楼 6F，提供英语与日语全流程陪诊，预约电话 400-800-9999。\n\n"
        "### 免费摆渡车时刻表\n"
        "地铁2号线馨博站 A 口每 15 分钟一班免费院区接驳巴士。"
    )
    chunks = scenario_rag.parse_custom_kb(custom_md, "cymbal_hospital")
    self.assertEqual(len(chunks), 2)
    res = scenario_rag.run_rag_tool(
        {"query": "地铁站有免费接驳摆渡车吗"},
        "cymbal_hospital",
        custom_kb=custom_md,
    )
    self.assertEqual(res["matches"][0]["title"], "免费摆渡车时刻表")
    self.assertIn("15 分钟", res["matches"][0]["content"])

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
      res = scenario_rag.run_rag_tool({"query": "手头闲钱怎么打理"}, "cymbal_bank")
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
          "args": {"query": "去哪里抽血"},
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
            "custom_kb": "### 周末理财沙龙\n每周六下午14:00在1F贵宾厅举办黄金投资讲座。",
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
