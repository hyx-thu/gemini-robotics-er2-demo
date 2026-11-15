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
    # Normal English sentences with words that overlap Pinyin syllables ("me",
    # "ben", "ma", "bang", "na") must NEVER be marked garbled.
    for en_text in (
        "I get a cold. Where should I go?",
        "Give me a pen please",
        "Can Ben tell me where to pay the bill?",
        "I got a dental issue. Which room should I go?",
        "I need to do a CT. Which room should I go?",
    ):
      self.assertFalse(
          decision_making.is_garbled_input_transcription(en_text), en_text
      )

  async def test_streaming_input_transcription_accumulates_full_text_per_utterance(self):
    bus = FakeBus()
    router = decision_making.DecisionMaking(bus)

    for chunk in ("Where ", "should ", "I ", "go?"):
      await router._handle_server_content(
          {"inputTranscription": {"text": chunk}}
      )

    self.assertEqual(4, len(bus.events))
    self.assertEqual("go?", bus.events[-1].data["text"])
    self.assertEqual(
        "Where should I go?", bus.events[-1].data["full_text"]
    )
    first_uid = bus.events[0].data["utterance_id"]
    self.assertTrue(all(e.data["utterance_id"] == first_uid for e in bus.events))

    # Starting a model response closes the utterance so the next user turn gets
    # a fresh utterance_id and fresh full_text buffer.
    router._mark_first_response()
    for chunk in ("Hello", "good", "morning"):
      await router._handle_server_content(
          {"inputTranscription": {"text": chunk}}
      )
    self.assertGreater(bus.events[-1].data["utterance_id"], first_uid)
    self.assertEqual("Hello good morning", bus.events[-1].data["full_text"])

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

  async def test_tool_call_does_not_split_concurrent_input_transcription_chunks(self):
    """Ensure toolCall arriving mid-utterance keeps full_text intact."""
    from core import event_bus  # pylint: disable=g-import-not-at-top

    bus = FakeBus()
    dm = decision_making.DecisionMaking(bus)
    for payload in (
        {"voiceActivity": {"type": "ACTIVITY_START"}},
        {"serverContent": {"inputTranscription": {"text": "Where can"}}},
        {
            "toolCall": {
                "functionCalls": [
                    {"name": "get_waypoints", "args": {}, "id": "c1"}
                ]
            }
        },
        {
            "serverContent": {
                "inputTranscription": {"text": "I do the payment?"}
            }
        },
    ):
      await dm._handle(
          event_bus.Event(
              type=event_bus.EventType.MODEL_RESPONSE,
              source=event_bus.EventSource.ASSISTANT,
              data=payload,
          )
      )

    transcripts = [
        e.data
        for e in bus.events
        if e.type == event_bus.EventType.USER_TRANSCRIPT
    ]
    self.assertEqual(2, len(transcripts))
    self.assertEqual(
        transcripts[0]["utterance_id"], transcripts[1]["utterance_id"]
    )
    self.assertEqual(
        "Where can I do the payment?", transcripts[1]["full_text"]
    )
    self.assertGreater(dm.last_model_output_time, 0.0)


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
    # English user, Chinese or Portuguese reply -> mismatch caught!
    self.assertEqual(
        "zh",
        session_manager.reply_language_mismatch(
            "en", "感冒发烧请前往一楼中央分诊台挂号就诊。"
        ),
    )
    self.assertEqual(
        "pt",
        session_manager.reply_language_mismatch(
            "en",
            "Para sintomas de resfriado, por favor dirija-se ao Balcão de Triagem.",
        ),
    )

  def test_live_session_english_sentences_never_misclassify_as_portuguese_or_other(self):
    """Regression test for live Cloud Run bug where 'a' matched Portuguese ('pt')."""
    import ast  # pylint: disable=g-import-not-at-top
    import inspect  # pylint: disable=g-import-not-at-top
    import session_manager  # pylint: disable=g-import-not-at-top

    english_sentences = (
        "I get a cold. Where should I go?",
        "I have a cold. Where should I do?",
        "I got a dental issue. Which room should I go?",
        "I need to do a CT. Which room should I go?",
        "Can you speak English?",
        "Hello, good morning.",
        "Where can I pay the bill?",
        "Where do I go for a blood draw?",
        "I need a wheelchair",
        "Book an appointment as soon as possible",
        "Pour a cup of water",
        "Come here and look at the desk",
        "Hi",
        "Hello",
        "Thanks",
        "Yes, please",
        "Help",
        "Stop",
    )
    for sentence in english_sentences:
      self.assertEqual(
          "en",
          session_manager.detect_language(sentence),
          f"Expected 'en' for {sentence!r}",
      )

    # Bare universal "OK" must remain neutral so it doesn't flip a non-English session.
    self.assertIsNone(session_manager.detect_language("OK."))

    # Multilingual coverage across all supported scripts & Latin languages.
    multilingual_samples = {
        "zh": "请问去哪里抽血和缴费？",
        "ja": "採血室はどこにありますか？",
        "ko": "채혈실은 어디로 가야 하나요?",
        "ru": "Здравствуйте, где можно оплатить счёт?",
        "ar": "مرحبا، أين يمكنني دفع الفاتورة؟",
        "hi": "नमस्ते, मुझे रक्त जांच के लिए कहाँ जाना चाहिए?",
        "th": "สวัสดีครับ ไปเจาะเลือดที่ชั้นไหนครับ",
        "es": "Hola, buenos días, ¿dónde puedo pagar la factura?",
        "fr": "Bonjour, où est le laboratoire pour une prise de sang ?",
        "de": "Guten Tag, ich brauche einen Termin beim Zahnarzt.",
        "pt": "Olá, preciso fazer um exame de sangue, onde fica o atendimento?",
        "it": "Buongiorno, vorrei sapere dove si trova la farmacia per favore.",
    }
    for expected_code, sample in multilingual_samples.items():
      self.assertEqual(
          expected_code,
          session_manager.detect_language(sample),
          f"Expected {expected_code!r} for {sample!r}",
      )
      display = session_manager.language_display_name(expected_code)
      self.assertIsNotNone(display)
      self.assertNotEqual(expected_code, display)

    # Static AST & anti-mocking check on session_manager.detect_language.
    src = inspect.getsource(session_manager.detect_language)
    tree = ast.parse(src)
    self.assertTrue(any(isinstance(n, ast.FunctionDef) for n in tree.body))
    for forbidden in ("mock_success", "fake_fallback", "DUMMY_"):
      self.assertNotIn(forbidden, src)

  def test_note_user_language_switches_cleanly_and_clears_stale_language_nudge(self):
    import session_manager  # pylint: disable=g-import-not-at-top

    dummy = type("DummySession", (), {})()
    dummy.user_language = "zh"
    dummy.user_language_name = "Mandarin Chinese (简体中文)"
    dummy._last_user_input_time = 0.0
    dummy._language_nudge_count = 2
    dummy._pending_action_nudge = (
        "[HEARTBEAT] LANGUAGE CORRECTION: speak ONLY Mandarin Chinese"
    )
    dummy._language_lock_line = (
        session_manager.SessionManager._language_lock_line.__get__(dummy)
    )
    note = session_manager.SessionManager.note_user_language.__get__(dummy)

    note("I get a cold. Where should I go?")
    self.assertEqual("en", dummy.user_language)
    self.assertEqual("English", dummy.user_language_name)
    self.assertEqual(0, dummy._language_nudge_count)
    self.assertEqual("", dummy._pending_action_nudge)
    self.assertIn("100% in English ONLY", dummy._language_lock_line())

    # Switching back to Chinese updates immediately as well.
    note("那我去几楼缴费？")
    self.assertEqual("zh", dummy.user_language)
    self.assertIn("Mandarin Chinese", dummy.user_language_name)
    self.assertIn("Mandarin Chinese", dummy._language_lock_line())

  def test_cjk_breath_fillers_and_mixed_asr_never_flip_english_to_chinese(self):
    """Exact regression test for 2026-11-15T02:08:38Z live Cloud Run log."""
    import session_manager  # pylint: disable=g-import-not-at-top

    # 1. Exact live Cloud Run utterance at 02:08:38.993Z:
    #    '啊。Yes, can you take me to the self-service kiosk?'
    live_utterance = "啊。Yes, can you take me to the self-service kiosk?"
    self.assertEqual(
        "Yes, can you take me to the self-service kiosk?",
        decision_making.normalize_input_transcription(live_utterance),
    )
    self.assertEqual("en", session_manager.detect_language(live_utterance))

    # 2. Other breath/hesitation fillers or 1-char stray ASR homophones before English:
    for mixed_en in (
        "嗯，where can I do the blood test?",
        "呃... I've caught a cold and keep coughing. Which floor should I go to?",
        "哦, can you take me there?",
        "好。Yes, can you take me to the self-service kiosk?",
        "喂, where can I pay the bill?",
    ):
      self.assertEqual(
          "en",
          session_manager.detect_language(mixed_en),
          f"Expected 'en' for {mixed_en!r}",
      )

    # 3. Standalone breath/hesitation fillers MUST return None (never flip en <-> zh):
    for filler_only in ("啊。", "嗯", "呃...", "哦！", "唔", "Uh...", "Um", "Hmm"):
      self.assertIsNone(
          session_manager.detect_language(filler_only),
          f"Expected None for standalone filler {filler_only!r}",
      )

    # 4. Real Chinese utterances with English loanwords or leading 'Hallo' stay 'zh':
    for real_zh in (
        "Hallo, hallo. 你可以听到吗?",
        "我想做个 MRI 检查",
        "请问 VIP 窗口在哪里？",
        "我想预约一个 English speaking doctor",
    ):
      self.assertEqual(
          "zh",
          session_manager.detect_language(real_zh),
          f"Expected 'zh' for {real_zh!r}",
      )

    # 5. Navigation phrase ("Can you take me there?") and get_waypoints must NOT
    #    trigger tabletop manipulation action_nudge:
    self.assertFalse(
        session_manager._is_action_request_text("Can you take me there?")
    )
    self.assertFalse(
        session_manager._is_action_request_text(
            "Yes, can you take me to the self-service kiosk?"
        )
    )
    self.assertIn("get_waypoints", session_manager._REAL_EMBODIMENT_ACTION_TOOLS)
    self.assertTrue(
        session_manager._is_action_request_text("Pick up the blue pen")
    )

  def test_every_tool_set_includes_information_tools(self):
    from tool import tools as tools_lib  # pylint: disable=g-import-not-at-top

    for build in (
        tools_lib.spot_tools,
        tools_lib.spot_sim_tools,
        tools_lib.robot_sim_tools,
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


