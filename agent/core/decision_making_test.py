"""Tests for model response routing."""

import unittest

from core import decision_making


class FakeBus:

  def __init__(self):
    self.events = []

  def subscribe(self, _event_types, _handler):
    return None

  async def publish(self, event):
    self.events.append(event)


class DecisionMakingTest(unittest.IsolatedAsyncioTestCase):

  async def test_hides_heartbeat_output_transcription(self):
    bus = FakeBus()
    router = decision_making.DecisionMaking(bus)

    await router._handle_server_content({
        "outputTranscription": {
            "text": "[HEARTBEAT] inspect the scene and call ack"
        }
    })

    self.assertEqual([], bus.events)

  async def test_output_transcription_is_routed_to_ui(self):
    """Valid spoken outputTranscription chunks publish GEMINI_TEXT for the UI."""
    bus = FakeBus()
    router = decision_making.DecisionMaking(bus)

    await router._handle_server_content({
        "outputTranscription": {"text": "I can see the desk."}
    })

    self.assertEqual(1, len(bus.events))
    self.assertEqual(
        decision_making.event_bus.EventType.GEMINI_TEXT, bus.events[0].type
    )
    self.assertEqual({"text": "I can see the desk."}, bus.events[0].data)

  async def test_hides_placeholder_and_error_output_transcription(self):
    bus = FakeBus()
    router = decision_making.DecisionMaking(bus)

    for ignored in ("<no speech>", " <noise> ", "An error occurred.", "   "):
      await router._handle_server_content({
          "outputTranscription": {"text": ignored}
      })

    self.assertEqual([], bus.events)

  async def test_input_transcription_does_not_interrupt_when_tts_idle(self):
    bus = FakeBus()
    router = decision_making.DecisionMaking(bus)

    await router._handle_server_content({
        "inputTranscription": {"text": "pick up the phone"}
    })

    self.assertEqual(
        [decision_making.event_bus.EventType.USER_TRANSCRIPT],
        [e.type for e in bus.events],
    )

  async def test_routes_reasoning_leak_in_output_transcription_to_thought(self):
    bus = FakeBus()
    router = decision_making.DecisionMaking(bus)

    await router._handle_server_content({
        "outputTranscription": {
            "text": (
                "    *   Current step: 0 (detect).\n"
                "    *   Goal: Pick up red mug.\n\n"
                "I already called `update_plan` and `detect` in the first turn."
            )
        }
    })

    self.assertEqual(
        [decision_making.event_bus.EventType.GEMINI_THOUGHT],
        [e.type for e in bus.events],
    )

  def test_normalize_input_transcription_strips_cjk_spaces_and_vad_noise(self):
    self.assertEqual(
        "你能帮我把笔夹在中间吗?",
        decision_making.normalize_input_transcription(
            " 你 能 帮 我 把 笔 夹 在 中 间 吗 ? "
        ),
    )
    self.assertEqual(
        "怎么把笔夹到本子中间?",
        decision_making.normalize_input_transcription(
            "pauca怎么把笔夹到本子中间?"
        ),
    )
    self.assertEqual(
        "你可以听到吗?",
        decision_making.normalize_input_transcription(
            "Hallo. 你可以听到吗?"
        ),
    )

  def test_is_garbled_input_transcription_detects_foreign_and_pinyin_gibberish(self):
    self.assertTrue(
        decision_making.is_garbled_input_transcription("Sí. Hoy es ¿Qué?")
    )
    self.assertTrue(
        decision_making.is_garbled_input_transcription("pão. Vamos ver...")
    )
    self.assertTrue(
        decision_making.is_garbled_input_transcription(
            "ni bao cha yi xia zhe ge bi 。"
        )
    )
    self.assertFalse(
        decision_making.is_garbled_input_transcription("Pick up the blue pen.")
    )
    self.assertFalse(
        decision_making.is_garbled_input_transcription("帮我把笔放到本子上")
    )

  async def test_sequential_tts_queue_plays_all_items_without_self_preempt(self):
    bus = FakeBus()
    spoken = []

    async def fake_tts(text: str):
      spoken.append(text)

    router = decision_making.DecisionMaking(bus, text_output_callback=fake_tts)
    await router._start_tts_task("First clause of plan.")
    await router._start_tts_task("Second clause after detect.")
    if router._active_tts_task is not None:
      await router._active_tts_task

    self.assertEqual(
        ["First clause of plan.", "Second clause after detect."], spoken
    )
    self.assertFalse(router.speaking.is_set())


class LanguageLockTest(unittest.TestCase):

  def test_detect_language_scripts_and_latin_markers(self):
    import session_manager  # pylint: disable=g-import-not-at-top

    self.assertEqual("zh", session_manager.detect_language("帮我把笔放到本子上"))
    self.assertEqual("ja", session_manager.detect_language("ペンを取ってください"))
    self.assertEqual("ko", session_manager.detect_language("펜을 집어 주세요"))
    self.assertEqual(
        "en", session_manager.detect_language("Please pick up the pen and put it on the notebook")
    )
    self.assertEqual(
        "de",
        session_manager.detect_language(
            "Ich habe den Stift gefunden und werde ihn jetzt für Sie greifen."
        ),
    )
    self.assertEqual(
        "es",
        session_manager.detect_language(
            "He localizado el bolígrafo y ahora voy a recogerlo para usted."
        ),
    )
    self.assertIsNone(session_manager.detect_language("237, 614"))

  def test_reply_language_mismatch(self):
    import session_manager  # pylint: disable=g-import-not-at-top

    # Chinese user, German reply -> mismatch.
    self.assertEqual(
        "de",
        session_manager.reply_language_mismatch(
            "zh", "Ich habe den Stift gefunden und greife ihn jetzt für Sie."
        ),
    )
    # Chinese user, Chinese reply with English tokens -> fine.
    self.assertIsNone(
        session_manager.reply_language_mismatch(
            "zh", "好的，我已经定位到 blue pen（坐标 237, 614），正在抓取。"
        )
    )
    # Short acks never trigger.
    self.assertIsNone(session_manager.reply_language_mismatch("zh", "OK."))
    # English user, English reply -> fine.
    self.assertIsNone(
        session_manager.reply_language_mismatch(
            "en", "I found the pen and I am picking it up now."
        )
    )

  def test_every_tool_set_includes_information_tools(self):
    from tool import tools as tools_lib  # pylint: disable=g-import-not-at-top

    for build in (
        tools_lib.human_tools,
        tools_lib.spot_tools,
        tools_lib.spot_sim_tools,
        tools_lib.tinybot_tools,
    ):
      names = {
          d["name"]
          for group in build()
          for d in group["functionDeclarations"]
      }
      self.assertTrue(
          {"get_current_location", "google_search", "google_maps_search"}
          <= names,
          build.__name__,
      )
      # No duplicates.
      all_names = [
          d["name"] for group in build() for d in group["functionDeclarations"]
      ]
      self.assertEqual(len(all_names), len(set(all_names)), build.__name__)


if __name__ == "__main__":
  unittest.main()


