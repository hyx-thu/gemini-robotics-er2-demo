"""Handler that executes tool calls from the model.

Subscribes to TOOL_CALL events, executes functions via the embodiment,
sends JSON dict responses back to the Gemini session, and publishes
TOOL_RESULT events.

This Lite version uses JSON dicts for the public BidiGenerateContent API
instead of protobuf messages.
"""

import asyncio
from collections.abc import Callable
import datetime
import json
import logging
import re
import time
from typing import Any

import httpx

from core import action_watchdog
from core import event_bus
from tool import grounding_tools
from tool import scenario_rag


_PICK_CAMERA_STABLE_SECONDS = 3.0
# Supervisor loop cadence for the action watchdog (seconds).
_WATCHDOG_TICK_S = 0.5
# confirm_action is only accepted once the command has had time to take
# effect and at least one camera frame captured after it has been sent.
_MIN_CONFIRM_DWELL_S = 3.0
# retry_step is two-phase: the failure must still be visible on a later
# heartbeat (between these bounds) before the plan is moved back.
_RETRY_RECHECK_MIN_S = 2.0
_RETRY_RECHECK_MAX_S = 30.0
# Plan-level retries (going back to redo steps) allowed per plan.
_MAX_PLAN_RETRIES = 2
_MIN_CONFIRM_FRAME_LAG_S = 1.0

# Spoken / typed commands that cancel the running task (any language).
_USER_STOP_RE = re.compile(
    r"\b(stop|cancel|abort|halt|enough|never mind|nevermind)\b|停止|停下|停一下|"
    r"别动|不要动|终止|取消|算了|不用了|中止|停止任务|やめて|止めて|그만|멈춰",
    re.IGNORECASE,
)


_NEGATED_SIT_PATTERN = re.compile(
    r"(?:\b(?:do\s+not|don't|never)\b[^.!?]{0,40}\bsit\b|\bnot\s+to\s+sit\b"
    # zh / ja / ko
    r"|(?:不要|别|不准|不许|不用|不必|无需)\s*(?:坐|趴)"
    r"|座らないで|すわらないで|앉지\s*마"
    # es / pt / fr / de / it / ru
    r"|\bno\s+te\s+sientes\b|\bnão\s+(?:se\s+)?sente\b|\bne\s+t['’]assieds\s+pas\b"
    r"|\bsetz\s+dich\s+nicht\b|\bnicht\s+(?:hin)?setzen\b|\bnon\s+sederti\b|\bне\s+садись\b)",
    re.IGNORECASE,
)
# Sit requests in any common language. Latin-script words use \b; CJK phrases
# cannot (no word boundaries), and ASR often inserts spaces between CJK
# characters ("坐 下 来"), which user_requested_sit() strips first.
_SIT_PATTERN = re.compile(
    r"\b(?:sit(?:\s+down)?|take\s+a\s+seat)\b"
    r"|坐下|坐好|坐着|坐吧|请坐|坐一下|趴下|蹲下"
    r"|座って|すわって|お座り|おすわり|座れ|座る"
    r"|앉아|앉으"
    r"|\bsi[eé]ntat[e]\b|\bsi[eé]ntese\b|\bsentad[oa]\b"
    r"|\bsenta\b|\bsente-se\b|\bsentar\b"
    r"|\bassieds-toi\b|\basseyez-vous\b|\bassis\b"
    r"|\bsetz\s+dich\b|\bhinsetzen\b|\bsitz\b"
    r"|\bsiediti\b|\bsedut[oa]\b"
    r"|\bсядь\b|\bсадись\b|\bсидеть\b",
    re.IGNORECASE,
)
_CJK_GAP_RE = re.compile(r"(?<=[^\x00-\x7f])\s+(?=[^\x00-\x7f])")


def user_requested_sit(text: str) -> bool:
  """Return whether a user utterance explicitly asks the robot to sit (any language)."""
  text = _CJK_GAP_RE.sub("", text or "")
  return bool(_SIT_PATTERN.search(text)) and not bool(
      _NEGATED_SIT_PATTERN.search(text)
  )


_WAVE_GESTURE_NAMES = frozenset({
    "high_wave",
    "wave",
    "left_hand_wave",
    "right_hand_wave",
    "greeting_crowd",
    "greets_with_both_hands",
    "hello",
    "hello_1",
})

_GREETING_OR_WAVE_RE = re.compile(
    r"\b(?:hello|hi|hey|howdy|greetings|good\s+(?:morning|afternoon|evening|night)"
    r"|bye|goodbye|see\s+(?:you|ya)|farewell|wave|waving"
    r"|who\s+are\s+you|what(?:'s|’s|\s+is)\s+your\s+name|introduce\s+yourself"
    r"|say\s+(?:hi|hello|bye|goodbye))\b"
    r"|你好|您好|嗨|哈喽|哈啰|早上好|早安|下午好|晚上好|晚安|再见|拜拜|回头见"
    r"|你是谁|你叫什么|自我介绍|介绍一下你自己|挥手|挥个手|挥挥手|招手|招个手|打招呼|打个招呼"
    r"|こんにちは|こんばんは|おはよう|さようなら|バイバイ|手を振"
    r"|안녕|반가워|잘\s*가|계세요"
    r"|\b(?:hola|buenos\s+d[ií]as|buenas\s+(?:tardes|noches)|adi[oó]s"
    r"|bonjour|bonsoir|salut|au\s+revoir"
    r"|hallo|guten\s+(?:morgen|tag|abend)|tsch[üu]ss"
    r"|ciao|buongiorno|buonasera|arrivederci"
    r"|ol[aá]|bom\s+dia|boa\s+(?:tarde|noite)|tchau"
    r"|привет|здравствуйте|пока|до\s+свидания)\b",
    re.IGNORECASE,
)

_NEGATED_LOCOMOTION_RE = re.compile(
    r"(?:\b(?:do\s+not|don['’]?t|no\s+need\s+to|never)\b[^.!?]{0,40}"
    r"\b(?:move|walk|go|guide|lead|escort|take\s+me|navigate|drive|come|follow)\b"
    r"|\b(?:stay\s+(?:here|still|put|in\s+place|where\s+you\s+are)"
    r"|just\s+(?:tell|answer|explain)(?:\s+me)?)\b"
    r"|\bno\s*,?\s*thanks?\b"
    r"|(?:不要|别|不用|不需要|不必|无需|先别|先不用).{0,6}"
    r"(?:动|走|移动|带我|带路|领我|领路|陪我|过去|去|导航|跟着)"
    r"|原地(?:待着|不动)|(?:只用|只要|光)?(?:告诉我|说一下|讲一下|回答)(?:就行|就可以|就好)"
    r"|動かないで|案内しなくていい|움직이지\s*마|안내하지\s*마"
    r"|\bno\s+te\s+muevas\b|\bne\s+bouge\s+pas\b|\bbeweg\s+dich\s+nicht\b)",
    re.IGNORECASE,
)

_EXPLICIT_LOCOMOTION_RE = re.compile(
    # English escort / follow / relative body motion commands
    r"\b(?:take\s+me|guide\s+me|lead\s+(?:me|the\s+way)|show\s+me\s+the\s+way"
    r"|walk\s+(?:me|with\s+me)|escort(?:\s+me)?|bring\s+me\s+(?:to|over|there)"
    r"|come\s+(?:here|over|with\s+me|closer)|follow\s+me"
    r"|let['’]?s\s+(?:go|walk|head\s+over)|navigate|head\s+over)\b"
    r"|\bdrive\s+(?:forward|back|backward|left|right|ahead|slowly|a\s+bit)\b"
    r"|\b(?:move|step|walk)\s+(?:forward|back|backward|left|right|closer|away|ahead"
    r"|two\s+steps|a\s+step|a\s+little)\b"
    r"|\bturn\s+(?:left|right|around)\b"
    # Imperative / polite 2nd-person "go to / walk to / head to <place>" (excludes "how do I go to" / "where should I go")
    r"|(?:^|[.!?,;]\s*|\b(?:please|can\s+you|could\s+you|would\s+you|will\s+you|robot|now|first|then)\s+)"
    r"(?:go|walk|head|move)\s+(?:to|over\s+to|back\s+to|straight\s+to|towards?)\b"
    # Chinese escort / navigation / fetch-at-station / relative drive commands
    r"|带我去|带我过去|带我到|带我找|领我去|领我过去|领我到|陪我去|陪我过去"
    r"|帮我带路|给我带路|替我带路|带路吧|带个路|请带路|领路|引路"
    r"|跟我走|跟上我|跟我来|我们走吧|咱们走吧|走吧|出发吧|带过去"
    r"|过来一下|走过来|退后|后退|往前走|向前走|往后退|向后退"
    r"|往左走|往右走|向左走|向右走|左转|右转|转身|走两步|前进一步|导航到|导航去"
    r"|(?:^|[，。！？,!?]\s*|(?:请你?|麻烦你?|你能|你可以|能帮我|帮我))"
    r"(?!(?:怎么|如何|在哪|哪里|哪儿|几楼))(?:去|走到|前往)(?!哪|几楼)[^，。！？?？]{1,15}(?:帮我|拿|取|捡|送)"
    r"|(?:请你?|麻烦你?|你去|你走到|你前往)(?!(?:怎么|如何|在哪|哪里|哪儿|几楼))"
    r"(?:去|走到|前往)?(?!哪|几楼)[^，。！？?？]{2,15}"
    # Multilingual (ja / ko / es / pt / fr / de / it / ru)
    r"|案内して|連れて行って|そこまで行って|ついてきて|前に進んで|後ろに下がって|右に曲がって|左に曲がって"
    r"|안내해\s*줘|안내해\s*주|데려다\s*줘|데려다\s*주|따라와|같이\s*가자|앞으로\s*가"
    r"|\b(?:ll[eé]vame|gu[ií]ame|acomp[aá][ñn]ame|vamos\s+all[aá]"
    r"|leve-me|guie-me|emm[eè]ne-moi|guide-moi|accompagne-moi"
    r"|f[üu]hr\s+mich|bring\s+mich\s+hin|begleite\s+mich"
    r"|portami|accompagnami|отведи\s+меня|проводи\s+меня)\b",
    re.IGNORECASE,
)

_ASSISTANT_ESCORT_OFFER_RE = re.compile(
    r"\b(?:guide\s+you|lead\s+(?:you|the\s+way)|escort\s+you|walk\s+(?:you|with\s+you)"
    r"|show\s+you\s+the\s+way|take\s+you\s+(?:there|over|right|straight|to)"
    r"|(?:want|like)\s+me\s+to\s+(?:guide|lead|take|walk|escort)"
    r"|shall\s+i\s+(?:guide|lead|take|walk|escort)|follow\s+me)\b"
    r"|带你去|带您去|带你过去|带您过去|带您前往|带你前往|领你去|领您去|领你过去|领您过去"
    r"|陪你去|陪您去|为您带路|为你带路|给您带路|给你带路|帮您带路|帮你带路|需要我带|要我带|跟我来"
    r"|案内しま|ご案内いたし|連れて行き|안내해\s*드릴|데려다\s*드릴"
    r"|\b(?:le\s+gu[ií]o|te\s+gu[ií]o|acompa[ñn]arle|guiar\s+voc[eê]"
    r"|vous\s+guider|vous\s+accompagner|sie\s+begleiten|accompagnarla)\b",
    re.IGNORECASE,
)

_SHORT_AFFIRMATIVE_RE = re.compile(
    r"^(?:yes|yeah|yep|yup|sure|ok|okay|please|please\s+do|yes\s*,?\s*please"
    r"|sure\s*,?\s*please|go\s+ahead|lead\s+on|sounds\s+good|let['’]?s\s+do\s+it|alright)"
    r"[!.,\s]*$"
    r"|^(?:好|好的|好呀|好啊|好嘞|可以|可以呀|行|行啊|嗯|嗯嗯|嗯好的|好的谢谢"
    r"|麻烦了|麻烦你了|有劳了|劳驾|太好了|走|去吧)[！!。，,~～\s]*$"
    r"|^(?:はい|お願いします|ええ|네|좋아요|부탁해요|s[ií]|claro|vale"
    r"|oui|d['’]accord|ja|gerne|sim|certo)[!.,\s]*$",
    re.IGNORECASE,
)

_NEW_QUESTION_TOPIC_RE = re.compile(
    r"[?？]|\b(?:where|what|when|how|why|who|which)\b|怎么|哪里|哪儿|什么|多少|几楼|吗",
    re.IGNORECASE,
)

_LEADING_MID_SESSION_GREETING_RE = re.compile(
    r"^\s*(?:"
    r"(?:hello(?:\s+there|\s+again)?|hi(?:\s+there|\s+again)?|hey(?:\s+there)?|greetings)"
    r"\s*[!,.，。！？]+\s*"
    r"(?:welcome\s+to\s+cymbal\s+(?:hospital|bank)[^.!?。！？]*[.!?。！？]\s*)?"
    r"(?:(?:i['’]?m|i\s+am)\s+(?:your\s+)?(?:atrium|smart|flagship)[^.!?。！？]*robot[^.!?。！？]*[.!?。！？]\s*)?"
    r"|(?:你好|您好|嗨|哈喽)\s*[！!，,。.]*\s*"
    r"(?:欢迎来到[^。！？!?.]*[。！？!?.]\s*)?"
    r"(?:我是[^。！？!?.]*(?:机器人|助手)[^。！？!?.]*[。！？!?.]\s*)?"
    r")",
    re.IGNORECASE,
)

_MANIPULATION_STEP_RE = re.compile(
    r"\b(?:detect|pick|place|wait_for_pick_up|open_gripper|close_gripper)\b",
    re.IGNORECASE,
)
_LOCOMOTION_STEP_RE = re.compile(r"\b(?:navigate|drive)\b", re.IGNORECASE)


def user_requested_wave_or_greeting(text: str) -> bool:
  """Return True when the user greeted, said goodbye, asked who the robot is, or asked for a wave."""
  norm = _CJK_GAP_RE.sub("", (text or "").strip())
  if not norm:
    return False
  return bool(_GREETING_OR_WAVE_RE.search(norm))


def assistant_offered_escort_in_text(text: str) -> bool:
  """Return True when the assistant's reply offered to guide/escort the user somewhere."""
  norm = _CJK_GAP_RE.sub("", (text or "").strip())
  if not norm:
    return False
  return bool(_ASSISTANT_ESCORT_OFFER_RE.search(norm))


def user_requested_locomotion(
    text: str, assistant_offered_escort: bool = False
) -> bool:
  """Return True ONLY when the user explicitly commands or confirms robot movement/escort."""
  norm = _CJK_GAP_RE.sub("", (text or "").strip())
  if not norm or norm == "🎤 正在识别语音…":
    return False
  if _NEGATED_LOCOMOTION_RE.search(norm):
    return False
  if _EXPLICIT_LOCOMOTION_RE.search(norm):
    return True
  if (
      assistant_offered_escort
      and len(norm) <= 45
      and not _NEW_QUESTION_TOPIC_RE.search(norm)
      and _SHORT_AFFIRMATIVE_RE.match(norm)
  ):
    return True
  return False


def strip_redundant_mid_session_greeting(
    message: str, already_greeted: bool, user_text: str
) -> str:
  """Strip accidental mid-session re-greetings ('Hello! Welcome to...') on follow-up Q&A turns."""
  msg = (message or "").strip()
  if not msg or not already_greeted or user_requested_wave_or_greeting(user_text):
    return message
  m = _LEADING_MID_SESSION_GREETING_RE.match(msg)
  if not m:
    return message
  remainder = msg[m.end() :].lstrip()
  if len(remainder) < 10:
    return message
  if remainder[0].isascii() and remainder[0].islower():
    remainder = remainder[0].upper() + remainder[1:]
  return remainder


def pick_response_for_model(
    result: Any,
    post_action_frame_sent: bool,
) -> dict[str, Any]:
  """Describe pick completion without claiming visually unverified success."""
  if not isinstance(result, dict):
    result = {"result": result}

  if (
      result.get("executed") is False
      and "required_stable_seconds" in result
  ):
    return {
        "outcome": "blocked_camera_unstable",
        "executed": False,
        "visual_verification_required": False,
        "post_action_frame_sent": post_action_frame_sent,
        "camera_stable_for_seconds": result.get(
            "camera_stable_for_seconds", 0.0
        ),
        "required_stable_seconds": result["required_stable_seconds"],
        "instruction": result.get("retry_instruction", result.get("error", "")),
    }

  failure_reasons = []
  if result.get("error"):
    failure_reasons.append(str(result["error"]))
  if result.get("success") is False:
    failure_reasons.append("backend success diagnostic is false")
  if result.get("holding_item") is False:
    failure_reasons.append("gripper holding-item sensor is false")
  backend_state = str(result.get("state", ""))
  if "FAILED" in backend_state or "NO_SOLUTION" in backend_state:
    failure_reasons.append(f"backend manipulation state is {backend_state}")

  response: dict[str, Any] = {
      "outcome": "failed" if failure_reasons else "unverified",
      "visual_verification_required": not failure_reasons,
      "post_action_frame_sent": post_action_frame_sent,
      "instruction": (
          "Inspect the fresh post-pick camera image. Claim success only if the"
          " requested object is visibly secured by the gripper and has moved"
          " from its original location. Backend completion and gripper sensor"
          " signals are diagnostics, not proof of a successful pick. If the"
          " image is missing or ambiguous, report that the pick is unverified."
      ),
  }
  if backend_state:
    response["backend_state"] = backend_state
  if failure_reasons:
    response["failure_reasons"] = failure_reasons
  if isinstance(result.get("target"), dict):
    response["target"] = result["target"]
  # Watchdog-supervised embodiments: keep the attempt bookkeeping and the
  # explicit next step so the model follows the confirm/retry protocol.
  for key in ("grasp_point", "grasp_part", "trajectory", "watchdog", "next_action"):
    if key in result:
      response[key] = result[key]
  return response


def blocking_tool_names(tools: list[dict[str, Any]] | None) -> frozenset[str]:
  """Return the names of tools whose declarations mark them BLOCKING."""
  names = set()
  for group in tools or []:
    declarations = group.get("functionDeclarations", [group])
    for declaration in declarations:
      if str(declaration.get("behavior", "")).upper() == "BLOCKING":
        name = declaration.get("name")
        if name:
          names.add(name)
  return frozenset(names)


class ToolCallHandler:
  """A handler that executes tool calls and publishes results back to the event bus."""

  def __init__(
      self,
      bus: event_bus.EventBus,
      session_manager: Any,
      embodiment: Any,
      agent_peers: dict[str, str] | None = None,
      peer_name: str = "unknown",
      text_output_callback: Any | None = None,
      clock: Callable[[], datetime.datetime] = datetime.datetime.utcnow,
      enable_send_message_to_user: bool = False,
      blocking_tools: frozenset[str] | set[str] | None = None,
  ):
    self._bus = bus
    self._clock = clock
    self._session_manager = session_manager
    self._embodiment = embodiment
    self._agent_peers = agent_peers or {}
    self._peer_name = peer_name
    # Optional TTS callback so send_message can speak the message
    # out loud before delivering it to the peer agent.
    self._text_output_callback = text_output_callback
    self._enable_send_message_to_user = enable_send_message_to_user
    self._blocking_tools = frozenset(blocking_tools or ())
    self._blocking_execution_count = 0
    self._sit_authorized = False
    self._locomotion_authorized = False
    self._plan_allows_locomotion = False
    self._locomotion_done_hb_count: int | None = None
    self._assistant_offered_escort = False
    self._prev_turn_offered_escort = False
    self._assistant_turn_buf = ""
    self._assistant_has_replied = False
    self._greeted_prior_turn = False
    self._current_utterance_id: int | None = None
    self._wave_count_in_session = 0
    self._user_transcript = ""
    # Last accepted update_plan (goal/steps/current_step/status); used to gate
    # advancing past an unconfirmed action and to mark the plan failed when
    # the watchdog aborts a task.
    self._plan: dict[str, Any] | None = None
    # Plan stickiness bookkeeping (see _execute_update_plan).
    self._last_user_input_time = 0.0
    self._plan_started_at = 0.0
    self._plan_finished_at = 0.0
    # retry_step bookkeeping (reset with every new plan).
    self._plan_retries = 0
    self._retry_intent: tuple[int, float] | None = None
    # Command -> visual confirmation -> retry -> abort supervisor. Enabled by
    # embodiments that set ``verify_actions_visually = True``.
    self._watchdog = self._build_watchdog(embodiment)
    self._watchdog_task: asyncio.Task | None = None
    # Set while any blocking tool is executing. The heartbeat
    # loop checks this and skips sending prompts to avoid re-triggering
    # the model before the tool result is in context.
    self.tool_executing = asyncio.Event()
    bus.subscribe([event_bus.EventType.TOOL_CALL], self._handle_tool_call)
    bus.subscribe(
        [event_bus.EventType.TEXT_INPUT, event_bus.EventType.USER_TRANSCRIPT],
        self._record_user_instruction,
    )
    bus.subscribe(
        [event_bus.EventType.GEMINI_TEXT],
        self._record_assistant_text,
    )

  @staticmethod
  def _build_watchdog(embodiment: Any) -> action_watchdog.ActionWatchdog | None:
    if not getattr(embodiment, "verify_actions_visually", False):
      return None
    return action_watchdog.ActionWatchdog(
        timeout_s=float(
            getattr(embodiment, "action_timeout_s", action_watchdog.DEFAULT_TIMEOUT_S)
        ),
        max_attempts=int(
            getattr(
                embodiment, "action_max_attempts", action_watchdog.DEFAULT_MAX_ATTEMPTS
            )
        ),
        verified_actions=frozenset(
            getattr(
                embodiment, "verified_actions", action_watchdog.DEFAULT_VERIFIED_ACTIONS
            )
        ),
    )

  @property
  def watchdog(self) -> action_watchdog.ActionWatchdog | None:
    return self._watchdog

  def task_active(self) -> bool:
    """True while a plan is in progress or an action awaits confirmation."""
    if self._plan and self._plan.get("status") in ("planning", "in_progress"):
      return True
    wd = self._watchdog
    return bool(wd is not None and getattr(wd, "pending", None) is not None)

  def heartbeat_context(self) -> str:
    """Extra heartbeat instructions: plan progress + watchdog state."""
    parts: list[str] = []
    plan = self._plan
    if plan and plan.get("status") in ("planning", "in_progress"):
      steps = [str(s) for s in plan.get("steps") or []]
      cur = int(plan.get("current_step", 0) or 0)
      cur = max(0, min(cur, len(steps) - 1)) if steps else 0
      cur_text = steps[cur] if steps else ""
      nxt_text = steps[cur + 1] if cur + 1 < len(steps) else "(none - goal reached)"
      pending = self._watchdog.pending if self._watchdog is not None else None
      goal = str(plan.get("goal") or "").strip()
      if pending is not None:
        rule = (
            f"The {pending.name} command of this step is awaiting VISUAL"
            " confirmation (see [TARGET] and [WATCHDOG] below). Remember: any"
            " hand, gripper, or actuator in the camera IS our robot arm"
            " executing our command. Perform this 3-way check on the latest"
            " frame: (1) If the frame shows its effect (for pick: a hand is"
            " grasping, lifting, or holding the object off the table; for"
            " place: the object rests at the destination) - OR shows that"
            " this and later steps are already done (e.g. the object is"
            " already at its destination) - call confirm_action, then"
            " update_plan jumping current_step to the first step whose effect"
            " is NOT yet visible (NEVER call detect or pick once a hand/arm"
            " has entered the pick process or is holding the object!)."
            " (2) If NO hand or arm has entered the pick/grasp process (the"
            " object is still lying untouched on the table surface) and the"
            " target object (for pick) or destination (for place) has moved"
            " away from its [TARGET] position (>~60 units or outside box_2d),"
            " IMMEDIATELY call detect on the current frame with its new"
            f" coordinates and re-issue {pending.name} with the updated"
            " position (do NOT wait for the 30s timeout and do NOT just call"
            " ack). (3) Otherwise, if the target is still at its [TARGET] spot"
            " and the command is still in progress, call ack once."
        )
      else:
        rule = (
            "FOLLOW THIS PLAN; do not restart or rewrite it. Judge progress"
            " against the GOAL and the latest camera frame. Any hand or"
            " gripper visible in the camera IS our robot arm obeying our"
            " commands. (a) If the scene shows this step's effect AND"
            " possibly later steps already achieved (e.g. for pick/detect: a"
            " hand is already holding or lifting the object in the air, or"
            " the object is already at its destination), call update_plan"
            " with the SAME steps and current_step set to the first step"
            " whose effect is NOT yet visible (or status='done' if the goal"
            " is reached), then immediately call that step's tool - NEVER"
            " redo steps whose physical effect is already achieved and NEVER"
            " detect or pick an object that is already held or being grasped"
            " by a hand! (b) If this step's tool has not been called yet:"
            " before calling pick or place, check [TARGET] against the LATEST"
            " camera frame - if the unpicked target object still lying"
            " untouched on the table (with no hand/arm touching or holding it)"
            " or the placement destination has moved since your last detect"
            " (>~60 units or outside box_2d), you MUST IMMEDIATELY call detect"
            " again on the current frame with its NEW coordinates FIRST, and"
            " then call pick or place with the new position (an object already"
            " picked moves with the hand, which is expected and never a reason"
            " to re-detect or re-pick it); only if the target has NOT moved"
            " from [TARGET], call this step's tool now. (c) If it is still in"
            " progress, call ack once and keep watching. (d) Small deviations"
            " (slightly different spot, object nudged) are NOT failures -"
            " carry on with the next step. Only if the scene makes the goal"
            " impossible as planned (object dropped far away, grasp clearly"
            " missed after the watchdog says so, target blocked) re-issue the"
            " tool or update_plan keeping the same goal and completed steps"
            " and revising ONLY the remaining steps. Never invent a new goal."
        )
      parts.append(
          f"[PLAN] goal = {goal!r}; step {cur + 1}/{len(steps)} ="
          f" {cur_text!r}; next = {nxt_text!r}. {rule}"
      )
    target_fn = getattr(self._embodiment, "target_context", None)
    if callable(target_fn):
      try:
        target_text = target_fn()
      except Exception:  # pylint: disable=broad-except
        logger.exception("target_context failed")
        target_text = ""
      if target_text:
        parts.append(target_text)
    if self._watchdog is not None:
      wd_text = self._watchdog.heartbeat_text()
      if wd_text:
        parts.append(wd_text)
    return "\n".join(parts)

  def shutdown(self) -> None:
    """Cancel the watchdog supervisor loop (session teardown)."""
    orch = getattr(self._session_manager, "er_orchestrator", None)
    if orch is not None:
      orch.cancel()
    if self._watchdog_task is not None and not self._watchdog_task.done():
      self._watchdog_task.cancel()
    self._watchdog_task = None

  def set_session_manager(self, session_manager: Any):
    """Updates the session manager reference."""
    self._session_manager = session_manager

  def _notify_ui(self, payload: dict[str, Any]) -> None:
    """Send an immediate UI-only status payload without queuing on EventBus."""
    sm = self._session_manager
    notify = getattr(sm, "notify_ui", None) or getattr(sm, "_notify_ui", None)
    if callable(notify):
      try:
        notify(payload)
      except Exception:  # pylint: disable=broad-except
        logging.debug("notify_ui failed", exc_info=True)

  def set_embodiment(self, embodiment: Any):
    """Updates the embodiment reference."""
    self._embodiment = embodiment
    self.shutdown()
    self._watchdog = self._build_watchdog(embodiment)


  async def _handle_tool_call(self, event: event_bus.Event) -> None:
    """Handle a TOOL_CALL event by executing all function calls and sending one batched toolResponse."""
    tool_call = event.data
    function_calls = tool_call.get("functionCalls", [])
    if len(function_calls) <= 1:
      for fc in function_calls:
        try:
          await self._execute_single_tool(fc)
        except Exception:  # pylint: disable=broad-except
          logging.exception('Tool call %s failed', fc.get("name", "unknown"))
      return

    batched_responses: list[dict[str, Any]] = []

    async def collect(fr: dict[str, Any]) -> None:
      batched_responses.append(fr)

    for fc in function_calls:
      try:
        await self._execute_single_tool(fc, sink=collect)
      except Exception:  # pylint: disable=broad-except
        logging.exception('Tool call %s failed', fc.get("name", "unknown"))
    if batched_responses:
      if self._session_manager is None:
        raise RuntimeError('Session manager is not initialized')
      await self._session_manager.send_message(
          {"toolResponse": {"functionResponses": batched_responses}}
      )

  async def execute_for_agent(self, fc: dict) -> dict[str, Any]:
    """Run a tool for a non-Live sub-agent (e.g. the ER2 orchestrator).

    Same execution path as a Live tool call (watchdog, overlays, UI events),
    but the FunctionResponse is returned instead of sent to the Live stream.
    """
    box: dict[str, Any] = {}

    async def sink(function_response: dict[str, Any]) -> None:
      box["fr"] = function_response

    await self._execute_single_tool(fc, sink=sink)
    return box.get("fr") or {
        "id": str(fc.get("id", "")), "name": fc.get("name", ""),
        "response": {"error": "tool produced no response"},
    }

  async def _await_inflight_user_transcript_if_needed(
      self, func_name: str
  ) -> None:
    """If server VAD just started a new user utterance, wait briefly for its transcript."""
    if func_name not in ("navigate", "drive", "make_gesture", "update_plan"):
      return
    sm = self._session_manager
    dm = getattr(sm, "decision_making", None)
    speech_start = float(getattr(dm, "last_user_speech_start", 0.0) or 0.0)
    now = time.monotonic()
    if (
        speech_start > 0.0
        and 0.0 <= (now - speech_start) < 4.0
        and speech_start > self._last_user_input_time + 0.05
    ):
      for _ in range(8):
        await asyncio.sleep(0.03)
        if self._last_user_input_time >= speech_start - 0.05:
          break

  def _already_greeted_in_prior_turn(self) -> bool:
    sm = self._session_manager
    return bool(
        self._greeted_prior_turn
        or int(getattr(sm, "user_turn_count", 0) or 0) >= 2
    )

  def _should_block_unrequested_locomotion(
      self, is_subagent: bool = False
  ) -> bool:
    """Block unrequested navigate/drive during informational Q&A or idle heartbeats."""
    if is_subagent:
      return False
    if self._plan_allows_locomotion and self.task_active():
      return False
    sm = self._session_manager
    hb_count = int(getattr(sm, "_heartbeat_sent_count", 0) or 0)
    if self._locomotion_authorized:
      if (
          self._locomotion_done_hb_count is not None
          and not self.task_active()
          and hb_count > self._locomotion_done_hb_count
      ):
        return True
      return False
    has_user_turn = (
        bool(self._user_transcript.strip())
        or int(getattr(sm, "user_turn_count", 0) or 0) > 0
    )
    has_live_heartbeat = hb_count > 0
    if not has_user_turn and not has_live_heartbeat:
      return False
    return True

  def _precheck_suppressed_tool(
      self,
      func_name: str,
      args: dict[str, Any],
      is_subagent: bool = False,
  ) -> dict[str, Any] | None:
    """Suppress unrequested Q&A locomotion or mid-session wave spam before executing."""
    if func_name in ("navigate", "drive") and self._should_block_unrequested_locomotion(
        is_subagent=is_subagent
    ):
      logging.info(
          "Suppressed unrequested %s during Q&A / heartbeat (user_transcript=%r)",
          func_name,
          self._user_transcript,
      )
      return {
          "executed": False,
          "status": "stayed_in_place",
          "_suppressed_ui": True,
          "note": (
              "STAY STATIONARY: The visitor only asked a question (or no"
              " movement was requested) and did NOT ask you to move or"
              " guide/escort them yet. Answer their question verbally right"
              " where you are (you may briefly offer to lead them there if"
              " helpful), and call navigate/drive ONLY after the visitor"
              " explicitly asks or confirms they want you to go or guide them."
          ),
      }
    if (
        func_name == "update_plan"
        and str(args.get("status", "in_progress")) in ("planning", "in_progress")
        and not self.task_active()
        and self._should_block_unrequested_locomotion(is_subagent=is_subagent)
    ):
      steps_joined = " ".join(str(s) for s in (args.get("steps") or []))
      if _LOCOMOTION_STEP_RE.search(steps_joined) and not _MANIPULATION_STEP_RE.search(
          steps_joined
      ):
        logging.info(
            "Suppressed unrequested navigation-only update_plan during Q&A / heartbeat"
        )
        return {
            "status": "ignored",
            "executed": False,
            "_suppressed_ui": True,
            "note": (
                "STAY STATIONARY: Do NOT start a navigation plan on a Q&A turn"
                " or heartbeat when the visitor did not request movement or"
                " escort. Answer verbally in place (or call ack if this was a"
                " heartbeat)."
            ),
        }
    if func_name == "make_gesture":
      gesture_name = str(args.get("gesture") or "").strip().lower()
      if gesture_name in _WAVE_GESTURE_NAMES:
        sm = self._session_manager
        has_context = (
            bool(self._user_transcript.strip())
            or int(getattr(sm, "_heartbeat_sent_count", 0) or 0) > 0
        )
        already_waved_or_greeted = (
            self._wave_count_in_session >= 1
            or self._already_greeted_in_prior_turn()
            or bool(getattr(sm, "_greeted_in_session", False))
        )
        if (
            already_waved_or_greeted
            and has_context
            and not user_requested_wave_or_greeting(self._user_transcript)
        ):
          logging.info(
              "Suppressed redundant mid-session %s(%r) during ongoing Q&A",
              func_name,
              gesture_name,
          )
          return {
              "executed": False,
              "status": "skipped",
              "_suppressed_ui": True,
              "note": (
                  "Already greeted earlier in this session; skipped redundant"
                  " wave gesture during ongoing Q&A. Continue answering the"
                  " visitor's question naturally without re-greeting ('Hello' /"
                  " 'Hi' / 'Welcome to...')."
              ),
          }
    return None

  async def _execute_single_tool(self, fc: dict, sink: Any = None) -> None:
    """Execute a single function call and publish the result."""
    func_name = fc.get("name", "")
    args = dict(fc.get("args", {}) or {})
    call_id = str(fc.get("id", ""))
    is_subagent = bool(fc.get("_agent"))
    is_blocking = func_name in self._blocking_tools
    if is_blocking:
      self._blocking_execution_count += 1
      self.tool_executing.set()
    suppressed_ui = False
    try:
      await self._await_inflight_user_transcript_if_needed(func_name)
      suppressed_result = self._precheck_suppressed_tool(
          func_name, args, is_subagent=is_subagent
      )
      if suppressed_result is not None:
        suppressed_ui = bool(suppressed_result.get("_suppressed_ui"))
        result = suppressed_result
      else:
        if (
            func_name == "send_message"
            and args.get("target") == "user"
            and isinstance(args.get("message"), str)
        ):
          cleaned_msg = strip_redundant_mid_session_greeting(
              args["message"],
              already_greeted=self._already_greeted_in_prior_turn(),
              user_text=self._user_transcript,
          )
          if cleaned_msg != args["message"]:
            args["message"] = cleaned_msg
            fc = {**fc, "args": args}
        if func_name and func_name not in event_bus.IDLE_TOOL_NAMES:
          if (
              func_name not in {"update_plan", "confirm_action", "retry_step"}
              and self._plan
              and self._plan.get("status") in ("done", "failed")
              and self._last_user_input_time > self._plan_finished_at
          ):
            self._plan = None
          self._notify_ui({
              "type": "tool_started",
              "id": call_id,
              "name": func_name,
              "args": args,
              "agent": fc.get("_agent") or "",
          })
        watchdog_block = (
            self._watchdog.blocks(func_name) if self._watchdog is not None else None
        )
        if watchdog_block is not None:
          # Another command is still awaiting visual confirmation.
          result = watchdog_block
        elif func_name in {"detect", "pick", "place"}:
          if func_name == "detect":
            await self._clear_pick_target()
          stability_error = self._camera_stability_error(func_name)
          if stability_error is not None:
            if func_name in {"pick", "place"}:
              robot = getattr(self._embodiment, "robot", None)
              clear_target = getattr(robot, "clear_detected_target", None)
              if clear_target is not None:
                clear_target()
            result = stability_error
          else:
            result = await self._invoke_tool(func_name, args)
            if func_name == "detect":
              await self._publish_detected_target(result)
              # If the model is re-detecting after a timeout notice, give its
              # follow-up pick/place a full grace window so _auto_reissue does
              # not fire with stale coordinates between detect and pick/place.
              if (
                  self._watchdog is not None
                  and self._watchdog.pending is not None
                  and self._watchdog.pending.timeout_notified_at is not None
              ):
                self._watchdog.pending.timeout_notified_at = (
                    self._watchdog._clock()  # pylint: disable=protected-access
                )
        else:
          result = await self._invoke_tool(func_name, args)
          if func_name == "make_gesture":
            gesture_name = str(args.get("gesture") or "").strip().lower()
            if gesture_name in _WAVE_GESTURE_NAMES:
              self._wave_count_in_session += 1
          elif (
              func_name == "navigate"
              and isinstance(result, dict)
              and result.get("executed") is not False
              and not result.get("error")
              and not self.task_active()
          ):
            self._locomotion_done_hb_count = int(
                getattr(self._session_manager, "_heartbeat_sent_count", 0) or 0
            )
        if (
            self._watchdog is not None
            and watchdog_block is None
            and self._watchdog.is_verified_action(func_name)
        ):
          result = await self._arm_watchdog(func_name, args, result)
        # Embodiments without a robot-side renderer hand us UI overlay
        # payloads (draw_points / clear_overlay); publish and strip them.
        result = await self._publish_ui_overlay(result)

      # A blocking FunctionResponse resumes model generation. Send the latest
      # visual context first instead of creating a competing heartbeat turn.
      post_action_frame_sent = False
      if (
          not suppressed_ui
          and is_blocking
          and sink is None
          and func_name
          not in (
              "talk_motion",
              scenario_rag.SCENARIO_RAG_TOOL_NAME,
          )
      ):
        send_frame = getattr(
            self._session_manager, "send_latest_video_frame", None
        )
        if send_frame is not None:
          post_action_frame_sent = bool(await send_frame())

      # Ensure result is a dict for the JSON response.
      if not isinstance(result, dict):
        result_dict = {'result': result}
      else:
        result_dict = dict(result)
      result_dict.pop("_suppressed_ui", None)
      if func_name == "pick":
        result_dict = pick_response_for_model(result, post_action_frame_sent)

      # Idle acknowledgements must not wake the model up again: deliver the
      # response with SILENT scheduling so it is only appended to the
      # context. Without this, each ack response re-triggers generation and
      # the model acks again in a tight loop (~2 Hz), never completing the
      # turn and burning ~6K prompt tokens per round.
      function_response: dict[str, Any] = {
          "id": call_id,
          "name": func_name,
          "response": result_dict,
      }
      if func_name in event_bus.IDLE_TOOL_NAMES:
        # v1beta FunctionResponse.scheduling (top level) is what the Live
        # API honours; omit for models that reject function response scheduling
        # with WebSocket 1007 (e.g. gemini-3.8-live-extended-thinking).
        if getattr(self._session_manager, "supports_blocking_tools", True):
          function_response["scheduling"] = "SILENT"
          result_dict.setdefault("scheduling", "SILENT")
      else:
        sm = self._session_manager
        dm = getattr(sm, "decision_making", None)
        speech_start = float(getattr(dm, "last_user_speech_start", 0.0) or 0.0)
        last_lang_ts = float(getattr(sm, "_last_user_input_time", 0.0) or 0.0)
        # If server VAD detected a brand-new user utterance and toolCall raced
        # ahead of inputTranscription, do not stamp a stale previous-turn
        # language lock onto this tool response.
        speech_newer_than_transcript = bool(
            speech_start > 0.0 and speech_start > last_lang_ts + 0.05
        )
        user_lang = (
            None
            if speech_newer_than_transcript
            else getattr(sm, "user_language_name", None)
        )
        if user_lang == "English":
          rule = (
              "The user is speaking English. Your reply (voice and text) MUST"
              " be 100% in English ONLY. Do NOT reply in Mandarin Chinese,"
              " Portuguese, Spanish, German, or any other non-English language,"
              " even if earlier turns in this conversation were in another"
              " language or the speaker has a non-native accent."
          )
        elif user_lang:
          rule = (
              f"The user is speaking {user_lang}. Your reply (voice and text)"
              f" MUST be entirely in {user_lang}. Do NOT switch to English,"
              " Portuguese, German, Spanish, or any other language because this"
              " tool response is written in English."
          )
        else:
          rule = (
              "Speak to the user in the EXACT language of the WORDS in the"
              " user's MOST RECENT utterance (reply 100% in English if the user"
              " spoke English words, even with an accent or after earlier"
              " Chinese turns; reply in Mandarin Chinese if the user spoke"
              " Chinese words). Never reply in Portuguese, Spanish, or another"
              " unrequested language."
          )
        result_dict.setdefault("reply_language_rule", rule)
        if self._already_greeted_in_prior_turn() and not user_requested_wave_or_greeting(
            self._user_transcript
        ):
          result_dict.setdefault(
              "conversation_continuity_rule",
              (
                  "Ongoing mid-session conversation (already greeted on an"
                  " earlier turn): jump straight into answering the user's"
                  " question naturally — do NOT start with 'Hello', 'Hi',"
                  " 'Welcome to...', or re-introduce yourself."
              ),
          )

      # Send tool response back to the Gemini session as a JSON dict.
      msg = {"toolResponse": {"functionResponses": [function_response]}}

      if sink is not None:
        await sink(function_response)
      else:
        if self._session_manager is None:
          raise RuntimeError('Session manager is not initialized')
        await self._session_manager.send_message(msg)

      if not suppressed_ui:
        # Publish result for UI and logging.
        action_data = json.dumps(fc)

        await self._bus.publish(
            event_bus.Event(
                type=event_bus.EventType.TOOL_RESULT,
                source=event_bus.EventSource.USER,
                data={
                    "type": "tool_call",
                    "id": call_id,
                    "name": func_name,
                    "args": args,
                    "result": result,
                    "action_data": action_data,
                    # Which model issued the call (composite mode: "er2").
                    "agent": fc.get("_agent") or "",
                },
            )
        )
      logging.info(
          'Tool call %s id=%s completed with result: %s',
          func_name,
          call_id,
          result,
      )
    finally:
      if func_name in {"pick", "place"} and not getattr(
          self._embodiment, "keep_target_overlay", False
      ):
        await self._clear_pick_target()
      if is_blocking:
        self._blocking_execution_count = max(
            0, self._blocking_execution_count - 1
        )
        if self._blocking_execution_count == 0:
            self.tool_executing.clear()

  async def _record_assistant_text(self, event: event_bus.Event) -> None:
    """Track assistant replies for ongoing-greeting state and escort offers."""
    data = event.data
    text = data.get("text", "") if isinstance(data, dict) else str(data or "")
    cleaned = text.strip()
    if not cleaned:
      return
    self._assistant_has_replied = True
    sm = self._session_manager
    if sm is not None and (
        bool(self._user_transcript.strip())
        or int(getattr(sm, "user_turn_count", 0) or 0) >= 1
    ):
      sm._greeted_in_session = True  # pylint: disable=protected-access
    self._assistant_turn_buf = f"{self._assistant_turn_buf} {cleaned}".strip()[
        -600:
    ]
    if assistant_offered_escort_in_text(self._assistant_turn_buf):
      self._assistant_offered_escort = True

  async def _record_user_instruction(self, event: event_bus.Event) -> None:
    """Track user requests: sit/locomotion authorisation, greeting state, stop/cancel."""
    is_replace = False
    is_new_utterance = False
    if event.type == event_bus.EventType.TEXT_INPUT:
      text = str(event.data or "")
      stripped = text.lstrip()
      if stripped.startswith(
          ("[HEARTBEAT]", "[ROBOT]", "[ROBOT_TASK_RESULT]", "[USER COMMAND]")
      ):
        return
      is_new_utterance = True
      self._current_utterance_id = None
      self._user_transcript = text.strip()
    else:
      data = event.data or {}
      text = data.get("text", "") if isinstance(data, dict) else str(data)
      full_text = data.get("full_text") if isinstance(data, dict) else None
      uid = data.get("utterance_id") if isinstance(data, dict) else None
      is_replace = bool(isinstance(data, dict) and data.get("replace"))
      if text.strip() == "🎤 正在识别语音…":
        return
      if uid is not None:
        try:
          uid_int = int(uid)
        except (TypeError, ValueError):
          uid_int = None
      else:
        uid_int = None
      if uid_int is not None:
        if uid_int != self._current_utterance_id:
          is_new_utterance = True
          self._current_utterance_id = uid_int
          self._user_transcript = str(
              full_text if full_text is not None else text
          ).strip()
        else:
          if full_text is not None:
            self._user_transcript = str(full_text).strip()
          elif is_replace:
            self._user_transcript = text.strip()
          else:
            self._user_transcript = f"{self._user_transcript} {text}".strip()
      else:
        is_new_utterance = True
        self._current_utterance_id = None
        self._user_transcript = text.strip()

    if is_new_utterance and self._user_transcript:
      sm = self._session_manager
      if (
          self._assistant_has_replied
          or self._wave_count_in_session >= 1
          or bool(getattr(sm, "_greeted_in_session", False))
      ):
        self._greeted_prior_turn = True
      self._prev_turn_offered_escort = self._assistant_offered_escort
      self._assistant_offered_escort = False
      self._assistant_turn_buf = ""
      self._locomotion_done_hb_count = None

    if text.strip() and text.strip() != "🎤 正在识别语音…":
      # Any genuine user input re-opens the right to (re)plan and dismisses
      # a finished plan from a previous turn.
      self._last_user_input_time = time.monotonic()
      if (
          not is_replace
          and self._plan
          and self._plan.get("status") in ("done", "failed")
      ):
        self._plan = None
        self._plan_allows_locomotion = False
        self._notify_ui({"type": "plan_clear"})
    self._sit_authorized = user_requested_sit(self._user_transcript)
    self._locomotion_authorized = user_requested_locomotion(
        self._user_transcript,
        assistant_offered_escort=self._prev_turn_offered_escort,
    )
    if self.task_active() and _USER_STOP_RE.search(text or ""):
      await self.cancel_task("user said stop")

  async def cancel_task(self, reason: str = "cancelled by user") -> dict[str, Any]:
    """Stop the current task on user request (UI button or spoken command)."""
    orch = getattr(self._session_manager, "er_orchestrator", None)
    if orch is not None:
      orch.cancel()
    wd = self._watchdog
    pending_name = wd.pending.name if (wd is not None and wd.pending) else None
    if wd is not None:
      wd.reset()
    hook = getattr(self._embodiment, "on_task_aborted", None)
    if hook is not None:
      try:
        hook(pending_name)
      except Exception:  # pylint: disable=broad-except
        logging.exception("on_task_aborted hook failed")
    stop = getattr(self._embodiment, "execute_action", None)
    if stop is not None:
      try:
        await stop("stop")
      except Exception:  # pylint: disable=broad-except
        logging.debug("stop() not available on this embodiment")
    had_plan = bool(self._plan and self._plan.get("status") in ("planning", "in_progress"))
    self._plan_allows_locomotion = False
    self._locomotion_authorized = False
    if had_plan:
      self._plan["status"] = "failed"
      self._plan_finished_at = time.monotonic()
      plan_args = {**self._plan, "note": f"⏹ {reason}"}
      await self._bus.publish(
          event_bus.Event(
              type=event_bus.EventType.TOOL_RESULT,
              source=event_bus.EventSource.ASSISTANT,
              data={
                  "type": "tool_call",
                  "id": "user-cancel",
                  "name": "update_plan",
                  "args": plan_args,
                  "result": {"status": "ok", "plan_status": "failed", "by": "user", "reason": reason},
                  "source": "watchdog",
              },
          )
      )
    await self._publish_action_status({"state": "cleared"})
    await self._clear_pick_target()
    # Tell the model (as a user turn, so it answers and goes idle).
    sm = self._session_manager
    send = getattr(sm, "send_text_with_fresh_video", None) or getattr(sm, "send_message", None)
    notice = (
        "[USER COMMAND] Stop. The user cancelled the current task"
        f" ({reason}). Do NOT call any more action tools for it; the plan has"
        " been marked failed/cancelled. Tell the user in ONE short sentence, in"
        " the user's language, that the task was stopped, then stay idle (only"
        " ack on heartbeats) until the user gives a new instruction."
    )
    try:
      if send is not None and send.__name__ == "send_text_with_fresh_video":
        await send(notice)
      elif send is not None:
        await send({
            "clientContent": {
                "turns": [{"role": "user", "parts": [{"text": notice}]}],
                "turnComplete": True,
            }
        })
    except Exception:  # pylint: disable=broad-except
      logging.exception("Failed to notify the model about task cancellation")
    logging.info("Task cancelled (%s); had_plan=%s pending=%s", reason, had_plan, pending_name)
    return {"status": "cancelled", "had_plan": had_plan, "pending": pending_name}

  def _camera_stability_error(self, action_name: str) -> dict[str, Any] | None:
    """Reject pixel-grounded manipulation until the camera is stable."""
    poller = getattr(self._embodiment, "poller", None)
    is_stable_for = getattr(poller, "is_stable_for", None)
    if is_stable_for is None or is_stable_for(_PICK_CAMERA_STABLE_SECONDS):
      return None
    stable_for = float(getattr(poller, "stable_for_seconds", 0.0))
    if action_name in {"pick", "place"}:
      next_action = action_name
    else:
      next_action = "pick or place"
    return {
        "error": (
            f"Camera view is not yet stable; {action_name} was not executed."
        ),
        "executed": False,
        "camera_stable_for_seconds": round(stable_for, 2),
        "required_stable_seconds": _PICK_CAMERA_STABLE_SECONDS,
        "retry_instruction": (
            "Wait until the hand-camera view has remained stable for three"
            " seconds and inspect the latest frame again. Call detect with an"
            f" exact target description; after detection succeeds, call {next_action}"
            " with no arguments and without moving the camera."
        ),
    }

  async def _publish_detected_target(self, result: Any) -> None:
    """Overlay the detect target (backend- or model-selected) on the UI."""
    if not isinstance(result, dict):
      return
    target = result.get("target")
    if not isinstance(target, dict):
      return
    try:
      x = float(target["normalized_x"])
      y = float(target["normalized_y"])
    except (KeyError, TypeError, ValueError):
      return
    if not 0 <= x <= 1000 or not 0 <= y <= 1000:
      return
    label = str(
        target.get("label") or result.get("label") or "detected target"
    )
    grasp_part = str(target.get("grasp_part") or result.get("grasp_part") or "").strip()
    point: dict[str, Any] = {"x": x, "y": y, "label": label, "kind": "target"}
    if grasp_part:
      point["grasp_part"] = grasp_part
    box = target.get("box_2d")
    if isinstance(box, (list, tuple)) and len(box) == 4:
      try:
        point["box"] = [float(v) for v in box]
      except (TypeError, ValueError):
        pass
    await self._bus.publish(
        event_bus.Event(
            type=event_bus.EventType.TOOL_RESULT,
            source=event_bus.EventSource.ASSISTANT,
            data={"type": "draw_points", "layer": "target", "points": [point]},
        )
    )

  async def _publish_ui_overlay(self, result: Any) -> Any:
    """Forward an embodiment-provided ``ui_overlay`` payload to the UI.

    Embodiments without a robot-side renderer (e.g. the simulated Spot)
    return ``{"ui_overlay": {"type": "draw_points"|"clear_overlay", ...}}``
    from draw_points / clear_overlay. The payload is published to the UI and
    stripped from the result sent to the model.
    """
    if not isinstance(result, dict):
      return result
    overlay = result.get("ui_overlay")
    if not isinstance(overlay, dict) or overlay.get("type") not in (
        "draw_points",
        "clear_overlay",
    ):
      return result
    await self._bus.publish(
        event_bus.Event(
            type=event_bus.EventType.TOOL_RESULT,
            source=event_bus.EventSource.ASSISTANT,
            data=dict(overlay),
        )
    )
    stripped = dict(result)
    stripped.pop("ui_overlay", None)
    return stripped

  async def _clear_pick_target(self) -> None:
    """Remove the pick marker (target layer only; model annotations stay)."""
    await self._bus.publish(
        event_bus.Event(
            type=event_bus.EventType.TOOL_RESULT,
            source=event_bus.EventSource.ASSISTANT,
            data={"type": "clear_overlay", "layer": "target"},
        )
    )


  async def _invoke_tool(self, func_name: str, args: dict[str, Any]) -> Any:
    try:
      if func_name == scenario_rag.SCENARIO_RAG_TOOL_NAME:
        sm = self._session_manager
        return await asyncio.to_thread(
            scenario_rag.run_rag_tool,
            args,
            getattr(sm, "scenario_id", None),
            getattr(sm, "scenario_custom_kb", ""),
        )
      if func_name in grounding_tools.GROUNDING_TOOL_NAMES:
        sm = self._session_manager
        return await asyncio.to_thread(
            grounding_tools.run_grounding_tool,
            func_name,
            args,
            getattr(sm, "api_key", None),
            getattr(sm, "client_geolocation", None),
            getattr(sm, "user_language_name", None),
        )
      if func_name == "sit":
        if not self._sit_authorized:
          return {
              "executed": False,
              "error": "Sit rejected: the user did not explicitly ask the robot to sit.",
          }
        self._sit_authorized = False
      if func_name == "send_message":
        return await self._execute_send_message(args)
      if func_name == "update_plan":
        return await self._execute_update_plan(args)
      if func_name == "confirm_action":
        return await self._execute_confirm_action(args)
      if func_name == "retry_step":
        return await self._execute_retry_step(args)
      if func_name == "robot_task":
        return await self._execute_robot_task(args)
      return await self._embodiment.execute_action(func_name, **args)
    except Exception as exc:  # pylint: disable=broad-except
      logging.error('Tool execution error for %s: %s', func_name, exc)
      return f'Error: {exc}'


  async def _execute_update_plan(self, args: dict[str, Any]) -> dict[str, Any]:
    """Validate and echo the plan; the UI renders it from the tool_call event.

    With an action watchdog active, the plan may not advance past the step
    whose command is still awaiting visual confirmation (the model must call
    ``confirm_action`` first). Marking the plan failed is always allowed.
    """
    steps = args.get("steps") or []
    if not isinstance(steps, list) or not steps:
      return {"error": "steps must be a non-empty list of strings"}
    try:
      current = int(args.get("current_step", 0))
    except (TypeError, ValueError):
      current = 0
    current = max(0, min(current, len(steps) - 1))
    status = str(args.get("status", "in_progress"))

    # ---- Plan stickiness (minimise spontaneous replans) --------------------
    # A plan may only change goal / restart from step 0 when the USER asked
    # for something after the current plan was started (or finished). Without
    # a new user request the original plan is kept and the model is told to
    # continue from where it is; progress never regresses.
    prev = self._plan
    clamped_from: int | None = None
    user_spoke_since_plan = (
        self._last_user_input_time > self._plan_started_at
    )
    new_goal = str(args.get("goal") or "").strip()
    if prev and status in ("planning", "in_progress") and not user_spoke_since_plan:
      prev_status = prev.get("status")
      prev_goal = str(prev.get("goal") or "").strip()
      prev_steps = [str(s) for s in prev.get("steps") or []]
      prev_cur = int(prev.get("current_step", 0) or 0)
      goal_changed = bool(new_goal) and new_goal != prev_goal
      if prev_status in ("done", "failed") and (
          time.monotonic() - self._plan_finished_at < 120.0
      ):
        # Task just ended and the user has not asked for anything new: do not
        # start a self-invented task.
        logging.info("update_plan ignored: new task without a user request")
        return {
            "status": "ignored",
            "plan_status": prev_status,
            "reason": (
                "The previous task is already"
                f" {prev_status} and the user has not given a new instruction."
                " Do not start tasks on your own."
            ),
            "next_action": "Stay idle: call ack on heartbeats until the user asks for something.",
        }
      if prev_status in ("planning", "in_progress"):
        if goal_changed:
          # Only reject as kept_original_plan when the model tries to start a
          # brand-new plan at step 0 ("planning" or current_step == 0). When
          # the model is advancing an in-progress plan (status == "in_progress"
          # and current > 0) and merely translated or rephrased the goal text,
          # silently preserve prev_goal and accept the step advancement!
          if status == "planning" or current == 0:
            logging.info(
                "update_plan kept original goal %r (model proposed %r)",
                prev_goal, new_goal,
            )
            return {
                "status": "kept_original_plan",
                "plan_status": prev_status,
                "goal": prev_goal,
                "current_step": prev_cur,
                "current_step_text": prev_steps[prev_cur] if prev_steps else "",
                "reason": (
                    "The user did not ask for a different task, so the goal"
                    f" stays {prev_goal!r}. Continue with the current step; only"
                    " revise the REMAINING steps of this goal if the scene truly"
                    " requires it."
                ),
            }
          logging.info(
              "update_plan preserved original goal %r across in-progress step"
              " update (model wrote %r)",
              prev_goal, new_goal,
          )
          new_goal = prev_goal
        # If the model shortened the step list by dropping already-completed
        # prefix steps (e.g. passing only the remaining 8 of 12 steps), restore
        # the completed prefix so step indices never shift or regress.
        if 0 < len(steps) < len(prev_steps):
          offset = len(prev_steps) - len(steps)
          if offset <= prev_cur:
            steps = prev_steps[:offset] + [str(s) for s in steps]
            current = min(len(steps) - 1, current + offset)
        if current < prev_cur:
          # Never move backwards without a user request; keep progress.
          logging.info(
              "update_plan clamped current_step %d -> %d (no regression)",
              current, prev_cur,
          )
          clamped_from = current
          current = prev_cur
          if current >= len(steps):
            steps = prev_steps
            current = min(prev_cur, len(steps) - 1)
    if not prev or prev.get("status") in ("done", "failed"):
      self._plan_started_at = time.monotonic()
      self._plan_retries = 0
      self._retry_intent = None

    wd = self._watchdog
    if wd is not None and wd.pending is not None and status != "failed":
      p = wd.pending
      # The plan may mark the pending action's own step as current (e.g. move
      # from "detect" to "pick"), but not move beyond it or finish.
      limit = self._action_step_index(p.name, p.plan_step, [str(s) for s in steps])
      advancing = status == "done" or (limit is not None and current > limit)
      if advancing:
        return {
            "error": (
                f"Plan not advanced: {p.name} (attempt {p.attempt}/"
                f"{wd.max_attempts}) has not been visually confirmed."
            ),
            "plan_status": self._plan["status"] if self._plan else "in_progress",
            "next_action": (
                "Look at the latest camera frame. If you can SEE that the"
                f" {p.name} completed, call confirm_action(observation=...)"
                " first and then update_plan again. If nothing changed, call"
                " ack and keep watching; the watchdog will tell you when to"
                f" re-issue {p.name}."
            ),
            "watchdog": wd.status(),
        }

    goal_str = new_goal or (str(prev.get("goal", "") or "").strip() if prev else "")
    self._plan = {
        "goal": goal_str,
        "steps": [str(s) for s in steps],
        "current_step": current,
        "status": status,
    }
    # If the model created the initial plan in a non-English language (e.g.
    # Chinese heard directly from raw audio) while inputAudioTranscription had
    # auto-translated the transcript to English, sync the language lock to the
    # plan's language so heartbeats do not force a mid-task language switch.
    sm = self._session_manager
    if (
        sm is not None
        and current == 0
        and goal_str
        and getattr(sm, "user_language", None) == "en"
        and _CJK_GAP_RE.pattern  # CJK check
        and re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", goal_str)
    ):
      note_lang = getattr(sm, "note_user_language", None)
      if callable(note_lang):
        note_lang(goal_str)
    dm = getattr(self._session_manager, "decision_making", None)
    if (
        dm is not None
        and getattr(dm, "last_transcript_garbled", False)
        and goal_str
        and current == 0
    ):
      dm.last_transcript_garbled = False
      uid = getattr(dm, "last_garbled_utterance_id", None)
      payload: dict[str, Any] = {"text": goal_str, "replace": True}
      if uid is not None:
        payload["utterance_id"] = uid
      await self._bus.publish(
          event_bus.Event(
              type=event_bus.EventType.USER_TRANSCRIPT,
              source=event_bus.EventSource.USER,
              data=payload,
          )
      )
    if status in ("planning", "in_progress") and (
        not prev or prev.get("status") in ("done", "failed")
    ):
      self._plan_allows_locomotion = bool(self._locomotion_authorized)
    if status in ("done", "failed"):
      self._plan_finished_at = time.monotonic()
      self._plan_allows_locomotion = False
    if status in ("done", "failed") and wd is not None:
      if status == "failed" and wd.pending is not None:
        hook = getattr(self._embodiment, "on_task_aborted", None)
        if hook is not None:
          try:
            hook(wd.pending.name)
          except Exception:  # pylint: disable=broad-except
            logging.exception("on_task_aborted hook failed")
      wd.reset()
      await self._publish_action_status({"state": "cleared"})
    result_payload: dict[str, Any] = {
        "status": "ok",
        "plan_status": status,
        "total_steps": len(steps),
        "current_step": current,
        "current_step_text": steps[current],
        "instruction": (
            "Plan displayed to the user. Now execute the current step with its"
            " action tool, then visually confirm the result before calling"
            " update_plan again to advance."
            if status in ("planning", "in_progress")
            else "Plan status recorded."
        ),
    }
    if clamped_from is not None:
      result_payload["note"] = (
          f"current_step was NOT moved back to {clamped_from}: progress is"
          f" kept at step {current}. Note: if the target object on the table"
          " (with no hand/arm grasping it) or the destination moved, you do"
          " NOT need to move current_step back via update_plan — directly call"
          " detect() on the current frame and then call pick() or place()"
          " while staying on the current step."
      )
    return result_payload

  # ---- Action watchdog (command -> visual confirmation -> retry -> abort) ----

  @staticmethod
  def _action_step_index(
      name: str, issued_step: int | None, steps: list[str]
  ) -> int | None:
    """Index of the plan step that performs ``name`` (search from issued_step).

    Steps are free text such as "pick() - grab the red block"; the first step
    at or after the one current when the command was issued that mentions the
    action name is taken as the action's own step. Falls back to issued_step.
    """
    start = max(0, issued_step or 0)
    pattern = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
    for idx in range(start, len(steps)):
      if pattern.search(steps[idx]):
        return idx
    return issued_step

  async def _arm_watchdog(
      self, func_name: str, args: dict[str, Any], result: Any
  ) -> Any:
    """Register a verified command with the watchdog and annotate the result."""
    wd = self._watchdog
    if wd is None or not isinstance(result, dict) or result.get("error"):
      return result
    plan_step = self._plan["current_step"] if self._plan else None
    effective_args = dict(args or {})
    if (
        func_name == "pick"
        and "grasp_x" not in effective_args
        and isinstance(result.get("grasp_point"), (list, tuple))
        and len(result["grasp_point"]) == 2
    ):
      effective_args["grasp_x"], effective_args["grasp_y"] = result["grasp_point"]
    elif (
        func_name == "place"
        and "x" not in effective_args
        and isinstance(result.get("place_point"), (list, tuple))
        and len(result["place_point"]) == 2
    ):
      effective_args["x"], effective_args["y"] = result["place_point"]
    info = wd.on_issued(
        func_name, effective_args, plan_step=plan_step, source="model"
    )
    if info.get("exhausted"):
      reason = (
          f"{func_name} was not executed after {info['max_attempts']} attempts"
          f" (no visible change within {wd.timeout_s:.0f}s each)."
      )
      await self._abort_task(reason)
      return {
          "executed": False,
          "error": reason,
          "task_status": "aborted",
          "next_action": (
              "The task was aborted. Tell the user in one short sentence,"
              " then stay idle and wait for a new instruction."
          ),
      }
    annotated = dict(result)
    annotated["watchdog"] = info
    if info.get("retargeted"):
      annotated["next_action"] = (
          f"{func_name} re-targeted to the new position (attempt"
          f" {info['attempt']}/{info['max_attempts']}, confirmation window"
          f" restarted: {info['confirm_within_s']:.0f}s). Keep watching the"
          f" camera: call confirm_action(observation=...) when you SEE the"
          f" {func_name} completed, otherwise ack."
      )
      await self._publish_action_status({**wd.status(), "state": "pending"})
      return annotated
    if info.get("merged"):
      annotated["next_action"] = (
          f"{func_name} attempt {info['attempt']}/{info['max_attempts']} is"
          f" already in flight ({info['confirm_within_s']}s left in its"
          " confirmation window); this call did not start a new attempt (any"
          " new coordinates it carried were applied). Keep"
          " watching the camera: call confirm_action(observation=...) when you"
          f" SEE the {func_name} completed, otherwise ack."
      )
      return annotated
    annotated["next_action"] = (
        f"Command sent (attempt {info['attempt']}/{info['max_attempts']})."
        " Watch the next camera frames; the arm/hand you see is the robot"
        f" executing it. On each heartbeat: (1) when the {func_name} has"
        " VISIBLY completed (for pick: a hand/gripper is grasping, lifting,"
        " or holding the object off the table; for place: object rests at"
        " destination) call confirm_action(observation=...), then update_plan"
        " to advance; (2) if NO hand/arm has entered the pick/grasp process"
        " yet and the untouched target on the table (or destination for place)"
        " has moved away from its detected position (>~60 units or outside"
        " box_2d), IMMEDIATELY call detect on the current frame and re-issue"
        f" {func_name} with the new coordinates without waiting for timeout;"
        f" (3) if the target has not moved and nothing changed within"
        f" {wd.timeout_s:.0f}s the watchdog will ask you to re-issue the"
        " command (call ack while waiting). Never ask the user to do it by"
        " hand."
    )
    await self._publish_action_status(
        {**wd.status(), "state": "retry" if info["retry"] else "pending"}
    )
    self._ensure_watchdog_loop()
    return annotated

  async def _execute_robot_task(self, args: dict[str, Any]) -> dict[str, Any]:
    """Composite mode: hand a robot task to the ER orchestrator (async)."""
    orch = getattr(self._session_manager, "er_orchestrator", None)
    if orch is None:
      return {"error": "robot_task is not available in this mode."}
    instruction = str(args.get("instruction", "")).strip()
    if not instruction:
      return {"error": "instruction is required."}
    return orch.start(instruction)

  async def _execute_retry_step(self, args: dict[str, Any]) -> dict[str, Any]:
    """Escalation level 2: redo earlier step(s) of the SAME plan.

    Used only when the in-flight pick/place can no longer succeed (object
    dropped, scene changed a lot). Two-phase: the first call only records the
    intent; a second call on a later heartbeat (problem still visible)
    cancels the in-flight command, rolls back the embodiment and moves the
    plan back. At most _MAX_PLAN_RETRIES per plan; after that the model must
    keep the plan or replan the remaining steps.
    """
    plan = self._plan
    if not plan or plan.get("status") not in ("planning", "in_progress"):
      return {"status": "not_applicable", "note": "No task in progress."}
    steps = [str(x) for x in plan.get("steps") or []]
    cur = int(plan.get("current_step", 0) or 0)
    try:
      step = int(args.get("step"))
    except (TypeError, ValueError):
      return {"error": "step must be the 0-based index of the plan step to redo."}
    if step < 0 or step > cur:
      return {
          "error": f"step must be between 0 and the current step ({cur}).",
          "next_action": "Keep the plan; call ack if nothing needs redoing.",
      }
    observation = str(args.get("observation", "")).strip()
    now = time.monotonic()
    intent = self._retry_intent
    confirmed = (
        intent is not None
        and intent[0] == step
        and _RETRY_RECHECK_MIN_S <= now - intent[1] <= _RETRY_RECHECK_MAX_S
    )
    if not confirmed:
      self._retry_intent = (step, now)
      logging.info("retry_step(%d) noted, awaiting recheck: %s", step, observation)
      return {
          "status": "awaiting_recheck",
          "step": step,
          "next_action": (
              "Nothing changed yet - the plan and the command in flight are"
              " kept. Look again on the NEXT heartbeat: only if the problem is"
              " still clearly visible, call retry_step again with the same"
              " step; if the scene is fine after all, just continue (ack /"
              " confirm_action)."
          ),
      }
    self._retry_intent = None
    wd = self._watchdog
    pending = wd.pending.name if wd is not None and wd.pending is not None else None
    if pending is not None:
      wd.reset()
      hook = getattr(self._embodiment, "on_retry", None) or getattr(
          self._embodiment, "on_task_aborted", None
      )
      if hook is not None:
        try:
          hook(pending)
        except Exception:  # pylint: disable=broad-except
          logging.exception("on_retry hook failed")
      await self._bus.publish(
          event_bus.Event(
              type=event_bus.EventType.TOOL_RESULT,
              source=event_bus.EventSource.ASSISTANT,
              data={"type": "clear_overlay", "layer": "trajectory"},
          )
      )
      await self._publish_action_status({"state": "cleared"})
    if self._plan_retries >= _MAX_PLAN_RETRIES:
      logging.info("retry_step(%d) refused: budget used; cancelled %s", step, pending)
      return {
          "status": "retry_budget_exhausted",
          "cancelled_command": pending,
          "next_action": (
              f"{_MAX_PLAN_RETRIES} retries were already used for this plan."
              " If the original remaining steps can still reach the goal,"
              " continue with the current step. Only if they truly cannot,"
              " replan: update_plan with the SAME goal, keep the completed"
              " steps and revise ONLY the remaining ones (or status='failed'"
              " if the goal is impossible), and tell the user in one sentence."
          ),
      }
    self._plan_retries += 1
    plan["current_step"] = step
    plan["status"] = "in_progress"
    note = observation or "retry"
    logging.info(
        "retry_step %d -> %d (retry %d/%d, cancelled %s): %s",
        cur, step, self._plan_retries, _MAX_PLAN_RETRIES, pending, observation,
    )
    # Let the UI plan panel follow (it renders update_plan tool events).
    await self._bus.publish(
        event_bus.Event(
            type=event_bus.EventType.TOOL_RESULT,
            source=event_bus.EventSource.ASSISTANT,
            data={
                "type": "tool_call",
                "id": f"retry-step-{self._plan_retries}",
                "name": "update_plan",
                "args": {
                    "goal": plan.get("goal", ""),
                    "steps": steps,
                    "current_step": step,
                    "status": "in_progress",
                    "note": note,
                },
                "result": {"status": "ok", "current_step": step,
                           "plan_status": "in_progress"},
                "source": "retry",
            },
        )
    )
    return {
        "status": "retrying",
        "current_step": step,
        "current_step_text": steps[step] if step < len(steps) else "",
        "retries_used": self._plan_retries,
        "max_retries": _MAX_PLAN_RETRIES,
        "cancelled_command": pending,
        "next_action": (
            f"Same plan, same goal. Redo step {step + 1} now on the CURRENT"
            " frame (e.g. detect the object again, then pick it), then"
            " continue with the remaining steps unchanged. Tell the user in"
            " one short sentence that you are retrying."
        ),
    }

  async def _execute_confirm_action(self, args: dict[str, Any]) -> dict[str, Any]:
    wd = self._watchdog
    if wd is None:
      return {
          "status": "not_applicable",
          "note": "This embodiment does not use confirm_action; call update_plan.",
      }
    observation = str(args.get("observation", "")).strip()
    p = wd.pending
    if p is not None:
      # A confirmation is only credible once the robot had time to act AND
      # the model has seen at least one frame captured after the command.
      elapsed = wd.elapsed()
      min_dwell = min(_MIN_CONFIRM_DWELL_S, 0.2 * wd.timeout_s)
      obs = getattr(self._session_manager, "observation", None)
      last_frame = float(getattr(obs, "last_frame_sent_time", 0.0) or 0.0)
      frame_after_cmd = (
          obs is None
          or last_frame >= p.issued_at + min(_MIN_CONFIRM_FRAME_LAG_S, min_dwell)
      )
      if elapsed < min_dwell or not frame_after_cmd:
        logging.info(
            "confirm_action(%s) too early: elapsed=%.1fs frame_after_cmd=%s",
            p.name, elapsed, frame_after_cmd,
        )
        return {
            "status": "not_confirmed",
            "action": p.name,
            "elapsed_s": round(elapsed, 1),
            "reason": (
                f"Only {elapsed:.1f}s since the {p.name} command was sent and"
                " no camera frame captured after it shows the result yet. A"
                " tool result is not evidence - the physical effect must be"
                " visible."
            ),
            "next_action": (
                "Call ack now and keep watching the next camera frames. Call"
                f" confirm_action again ONLY when you can SEE that the {p.name}"
                " happened (pick: a hand/gripper is grasping, lifting, or"
                " holding the object off the table; place: object resting at"
                " the target and the gripper empty). If no hand/arm has"
                " grasped the object and it moved to a different position on"
                " the table (or the destination moved), call detect on the"
                f" new position and re-issue {p.name} with the updated"
                " coordinates."
            ),
            "watchdog": wd.status(),
        }
    pending_plan_step = p.plan_step if p is not None else None
    result = wd.confirm(observation)
    if result.get("status") == "confirmed":
      hook = getattr(self._embodiment, "on_action_confirmed", None)
      if hook is not None:
        try:
          hook(result["action"])
        except Exception:  # pylint: disable=broad-except
          logging.exception("on_action_confirmed hook failed")
      # Automatically advance the stored plan past the confirmed pick/place
      # step so the plan never stays stuck on a completed detect/pick step even
      # if the model omits or botches update_plan.
      plan = self._plan
      if plan and plan.get("status") in ("planning", "in_progress"):
        steps = [str(s) for s in plan.get("steps") or []]
        cur = int(plan.get("current_step", 0) or 0)
        act_idx = self._action_step_index(
            str(result["action"]),
            pending_plan_step if pending_plan_step is not None else cur,
            steps,
        )
        if act_idx is not None and cur <= act_idx and act_idx + 1 < len(steps):
          plan["current_step"] = act_idx + 1
          plan["status"] = "in_progress"
          result["current_step"] = act_idx + 1
          result["current_step_text"] = steps[act_idx + 1]
          await self._bus.publish(
              event_bus.Event(
                  type=event_bus.EventType.TOOL_RESULT,
                  source=event_bus.EventSource.ASSISTANT,
                  data={
                      "type": "tool_call",
                      "id": f"confirm-advance-{act_idx + 1}",
                      "name": "update_plan",
                      "args": {
                          "goal": plan.get("goal", ""),
                          "steps": steps,
                          "current_step": act_idx + 1,
                          "status": "in_progress",
                          "note": observation or f"{result['action']} confirmed",
                      },
                      "result": {
                          "status": "ok",
                          "current_step": act_idx + 1,
                          "plan_status": "in_progress",
                      },
                      "source": "confirm_action",
                  },
              )
          )
      result["next_action"] = (
          "Call update_plan with current_step advanced to the next step"
          " (same steps list, status='in_progress') and continue."
      )
      await self._publish_action_status({
          "state": "confirmed",
          "action": result["action"],
          "attempt": result["attempts_used"],
          "max_attempts": wd.max_attempts,
          "observation": observation,
      })
    return result

  def _ensure_watchdog_loop(self) -> None:
    if self._watchdog_task is None or self._watchdog_task.done():
      self._watchdog_task = asyncio.create_task(self._watchdog_loop())

  async def _watchdog_loop(self) -> None:
    """Supervise the pending command until confirmed, retried out, or aborted."""
    wd = self._watchdog
    if wd is None:
      return
    last_notified: tuple[str, int] | None = None
    ticks = 0
    try:
      while wd.pending is not None:
        await asyncio.sleep(_WATCHDOG_TICK_S)
        ticks += 1
        p = wd.pending
        if p is None:
          break
        phase = wd.phase()
        if phase == action_watchdog.EXHAUSTED:
          await self._abort_task(
              f"{p.name} was not executed after {p.attempt} attempts"
              f" (no visible change within {wd.timeout_s:.0f}s each)."
          )
          break
        if phase == action_watchdog.TIMED_OUT:
          key = (p.name, p.attempt)
          if last_notified != key:
            last_notified = key
            # Start the grace clock here too, so the policy terminates even
            # if heartbeats (which carry the notice to the model) are delayed.
            wd.mark_timeout_notified()
            logging.info(
                "[watchdog] %s attempt %d/%d timed out; waiting for the model to"
                " re-issue", p.name, p.attempt, wd.max_attempts,
            )
            await self._publish_action_status({**wd.status(), "state": "timed_out"})
        elif phase == action_watchdog.OVERDUE:
          # The model did not re-issue within the grace period: re-send the
          # command on its behalf so the policy stays deterministic.
          await self._auto_reissue(p.name, p.args)
        elif ticks % 4 == 0:
          # Periodic sync (every ~2 s) so the UI countdown stays honest.
          await self._publish_action_status(wd.status())
    except asyncio.CancelledError:
      pass
    except Exception:  # pylint: disable=broad-except
      logging.exception("[watchdog] supervisor loop failed")

  async def _auto_reissue(self, name: str, args: dict[str, Any]) -> None:
    wd = self._watchdog
    if wd is None:
      return
    try:
      result = await self._embodiment.execute_action(name, **args)
    except Exception as exc:  # pylint: disable=broad-except
      result = {"error": f"re-issue failed: {exc}"}
    plan_step = self._plan["current_step"] if self._plan else None
    info = wd.on_issued(name, args, plan_step=plan_step, source="watchdog")
    if info.get("exhausted"):
      await self._abort_task(
          f"{name} was not executed after {info['max_attempts']} attempts"
          f" (no visible change within {wd.timeout_s:.0f}s each)."
      )
      return
    logging.info(
        "[watchdog] auto re-issued %s (attempt %d/%d)", name, info["attempt"],
        info["max_attempts"],
    )
    ui_result = dict(result) if isinstance(result, dict) else {"result": result}
    ui_result.pop("ui_overlay", None)
    ui_result["watchdog"] = info
    await self._bus.publish(
        event_bus.Event(
            type=event_bus.EventType.TOOL_RESULT,
            source=event_bus.EventSource.ASSISTANT,
            data={
                "type": "tool_call",
                "id": f"watchdog-{name}-{info['attempt']}",
                "name": name,
                "args": {"watchdog_retry": info["attempt"]},
                "result": ui_result,
                "source": "watchdog",
            },
        )
    )
    await self._publish_action_status({**wd.status(), "state": "retry"})

  async def _abort_task(self, reason: str) -> None:
    """Watchdog gave up: mark the plan failed, notify UI, return to idle."""
    wd = self._watchdog
    if wd is None:
      return
    info = wd.fail(reason)
    logging.warning("[watchdog] aborting task: %s", reason)
    hook = getattr(self._embodiment, "on_task_aborted", None)
    if hook is not None:
      try:
        hook(info.get("action"))
      except Exception:  # pylint: disable=broad-except
        logging.exception("on_task_aborted hook failed")
    await self._publish_action_status({
        "state": "failed",
        "action": info.get("action"),
        "attempt": info.get("attempts"),
        "max_attempts": info.get("max_attempts"),
        "reason": reason,
    })
    if self._plan and self._plan.get("status") in ("planning", "in_progress"):
      self._plan["status"] = "failed"
      self._plan_finished_at = time.monotonic()
      # Point the FAILED marker at the step that performs the aborted action
      # (the model may not have advanced the plan past the preceding step).
      steps = [str(s) for s in (self._plan.get("steps") or [])]
      failed_idx = self._action_step_index(
          str(info.get("action") or ""), self._plan.get("current_step"), steps
      )
      if failed_idx is not None and 0 <= failed_idx < len(steps):
        self._plan["current_step"] = failed_idx
      plan_args = {**self._plan, "note": f"watchdog: {reason}"}
      await self._bus.publish(
          event_bus.Event(
              type=event_bus.EventType.TOOL_RESULT,
              source=event_bus.EventSource.ASSISTANT,
              data={
                  "type": "tool_call",
                  "id": "watchdog-abort",
                  "name": "update_plan",
                  "args": plan_args,
                  "result": {
                      "status": "ok",
                      "plan_status": "failed",
                      "by": "watchdog",
                      "reason": reason,
                  },
                  "source": "watchdog",
              },
          )
      )
    await self._clear_pick_target()

  async def _publish_action_status(self, status: dict[str, Any]) -> None:
    await self._bus.publish(
        event_bus.Event(
            type=event_bus.EventType.TOOL_RESULT,
            source=event_bus.EventSource.ASSISTANT,
            data={"type": "action_status", **status},
        )
    )

  async def _execute_send_message(self, args: dict[str, Any]) -> str:
    """Send a message to another robot agent via HTTP POST.

    Waits for any in-flight TTS/audio playback to finish before sending,
    so the receiving agent doesn't start speaking over the sender.
    """
    target = args.get("target", "")
    message = args.get("message", "")

    if not target:
      return "Error: 'target' is required"
    if not message:
      return "Error: 'message' is required"

    if target == "user":
      if not self._enable_send_message_to_user:
        return (
            "Error: sending messages to user is not enabled."
            " Use inline text instead."
        )
      self._assistant_has_replied = True
      sm = self._session_manager
      if sm is not None and (
          bool(self._user_transcript.strip())
          or int(getattr(sm, "user_turn_count", 0) or 0) >= 1
      ):
        sm._greeted_in_session = True  # pylint: disable=protected-access
      self._assistant_turn_buf = f"{self._assistant_turn_buf} {message}".strip()[
          -600:
      ]
      if assistant_offered_escort_in_text(self._assistant_turn_buf):
        self._assistant_offered_escort = True
      dm = getattr(self._session_manager, "decision_making", None)
      if dm is not None and hasattr(dm, "_start_tts_task"):
        logging.info(
            "send_message to user: speaking message via DecisionMaking TTS task"
        )
        await dm._start_tts_task(message)
      elif self._text_output_callback:

        async def _run_tts():
          try:
            logging.info(
                "send_message to user: speaking message via TTS (non-blocking)"
            )
            await self._text_output_callback(message)
          except Exception as e:  # pylint: disable=broad-except
            logging.exception("TTS for send_message to user failed: %s", e)

        asyncio.create_task(_run_tts())
      return f"Message delivered to user: {message}"

    target_url = self._agent_peers.get(target)
    if not target_url:
      available = list(self._agent_peers.keys())
      return f"Error: unknown target '{target}'. Available peers: {available}"

    # Speak the message out loud via TTS before sending to peer,
    # so the user hears what is being communicated and the message
    # arrives after playback finishes.
    if self._text_output_callback:
      try:
        logging.info(
            "send_message to %s: speaking message via TTS first", target
        )
        await self._text_output_callback(message)
      except Exception as e:  # pylint: disable=broad-except
        logging.error("TTS for send_message failed: %s", e)

    try:
      async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{target_url}/api/send",
            json={"text": message, "source": self._peer_name},
            timeout=10.0,
        )
        if resp.status_code == 200:
          return f"Message delivered to {target}: {message}"
        else:
          return (
              f"Error sending to {target}: HTTP {resp.status_code} {resp.text}"
          )
    except httpx.ConnectError:
      return f"Error: could not connect to {target} at {target_url}"
    except Exception as e:  # pylint: disable=broad-except
      return f"Error sending to {target}: {e}"
