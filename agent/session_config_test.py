"""Tests for thinking level resolution."""

import unittest

import server
import session_config


class ThinkingLevelResolutionTest(unittest.TestCase):

  def test_model_default_level_is_used(self):
    for model_name, spec in session_config.KNOWN_MODELS.items():
      with self.subTest(model=model_name):
        self.assertEqual(
            spec.thinking_level,
            server.resolve_thinking_level(model_name),
        )

  def test_session_override_takes_precedence(self):
    # Use a model that supports thinking; unsupported ones always pin "none".
    model = "gemini-robotics-er-2-streaming-preview"
    override = "high"
    self.assertEqual(
        override,
        server.resolve_thinking_level(model, session_override=override),
    )

  def test_unknown_model_gets_global_default(self):
    self.assertEqual(
        session_config.DEFAULT_THINKING_LEVEL,
        server.resolve_thinking_level("some-unknown-model"),
    )

  def test_robotics_er_defaults_to_low(self):
    """The Robotics-ER streaming model (our default) thinks at 'low'."""
    for model in (
        "gemini-robotics-er-2-streaming-preview",
        "models/gemini-robotics-er-2-streaming-preview",
    ):
      with self.subTest(model=model):
        self.assertEqual("low", server.resolve_thinking_level(model))

  def test_gemini_live_defaults_to_no_thinking(self):
    """gemini-3.8-live is latency-optimised: thinking disabled by default."""
    self.assertEqual("none", server.resolve_thinking_level("gemini-3.8-live"))

  def test_every_ui_level_is_accepted_as_override(self):
    """Every option in the UI dropdown must round-trip through resolution."""
    for level in session_config.THINKING_LEVELS:
      with self.subTest(level=level):
        self.assertEqual(
            level,
            server.resolve_thinking_level(
                "gemini-robotics-er-2-streaming-preview", session_override=level
            ),
        )

  def test_models_prefix_stripped(self):
    """Ensure models/ prefix is stripped for lookup."""
    model = next(iter(session_config.KNOWN_MODELS))
    self.assertEqual(
        session_config.KNOWN_MODELS[model].thinking_level,
        server.resolve_thinking_level(f"models/{model}"),
    )

  def test_all_levels_are_valid(self):
    """All THINKING_LEVELS should be accepted."""
    for level in session_config.THINKING_LEVELS:
      with self.subTest(level=level):
        result = server.resolve_thinking_level("any-model", session_override=level)
        self.assertEqual(level, result)


if __name__ == "__main__":
  unittest.main()


class ThinkingSupportGuardTest(unittest.TestCase):
  """Models that reject thinkingConfig must always resolve to 'none'."""

  def test_unsupported_model_ignores_override(self):
    for override in (None, "none", "low", "high"):
      with self.subTest(override=override):
        self.assertEqual(
            "none",
            server.resolve_thinking_level("gemini-3.8-live", override),
        )
        self.assertEqual(
            "none",
            server.resolve_thinking_level("models/gemini-3.8-live", override),
        )

  def test_supported_model_honours_override(self):
    self.assertEqual(
        "high",
        server.resolve_thinking_level(
            "gemini-robotics-er-2-streaming-preview", "high"
        ),
    )


class ModelCapabilityTest(unittest.TestCase):
  """Setup payload must never contain fields a model rejects with 1007."""

  def test_response_modality_is_clamped_for_text_only_models(self):
    self.assertEqual(
        "TEXT",
        server.resolve_response_modality(
            "models/gemini-robotics-er-2-streaming-preview", "AUDIO"
        ),
    )
    self.assertEqual(
        "AUDIO", server.resolve_response_modality("gemini-3.8-live", "AUDIO")
    )

  def test_blocking_behavior_stripped_for_3_8_live(self):
    import asyncio
    import session_manager

    class FakeStream:
      def __init__(self):
        self.sent = []

      def Send(self, d):
        self.sent.append(d)

    tool = {"functionDeclarations": [
        {"name": "drive", "behavior": "BLOCKING", "parameters": {}},
        {"name": "ack", "parameters": {}},
    ]}

    async def setup_for(model):
      spec = server.model_spec_for(model)
      sm = session_manager.SessionManager.__new__(session_manager.SessionManager)
      sm.model = model
      sm.tools = [tool]
      sm.supports_blocking_tools = spec.supports_blocking_tools
      sm.thinking_level = "none"
      sm.system_instruction = "si"
      sm.developer_instruction = ""
      sm.response_modality = server.resolve_response_modality(model, "AUDIO")
      sm.stream = FakeStream()
      sm._setup_complete = asyncio.Event()  # pylint: disable=protected-access
      await sm._send_setup()  # pylint: disable=protected-access
      return sm.stream.sent[0]["setup"]

    for model in ("gemini-3.8-live", "gemini-3.8-live-extended-thinking"):
      with self.subTest(model=model):
        setup = asyncio.run(setup_for(model))
        decls = setup["tools"][0]["functionDeclarations"]
        self.assertTrue(all("behavior" not in d for d in decls))
        self.assertEqual(["drive", "ack"], [d["name"] for d in decls])
        self.assertEqual({}, setup["inputAudioTranscription"])
        self.assertEqual({}, setup["outputAudioTranscription"])

    er_setup = asyncio.run(setup_for("gemini-robotics-er-2-streaming-preview"))
    decls = er_setup["tools"][0]["functionDeclarations"]
    self.assertEqual("BLOCKING", decls[0]["behavior"])
    self.assertEqual({}, er_setup["inputAudioTranscription"])
    self.assertNotIn("outputAudioTranscription", er_setup)
    # Original declarations are untouched (handler keeps blocking semantics).
    self.assertEqual("BLOCKING", tool["functionDeclarations"][0]["behavior"])

  def test_strips_text_only_comms_rules_for_native_audio(self):
    from agent import agent as agent_lib
    for name in ("spot", "tinybot"):
      ag = agent_lib.Agent.from_name(name)
      stripped = server._strip_text_only_comms_rules(ag.developer_instruction)  # pylint: disable=protected-access
      self.assertNotIn("You MUST NOT reply directly with text or audio", stripped)
      self.assertNotIn("talk to the user via the `send_message` tool", stripped)
      self.assertNotIn("send_message(target='user'", stripped)
      self.assertNotIn('send_message(target="user"', stripped)
      if name == "spot":
        self.assertIn("LANGUAGE RULE", stripped)
        self.assertIn("Always speak and reply in the SAME language", stripped)


