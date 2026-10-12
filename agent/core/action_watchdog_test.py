"""Tests for the command -> confirm -> retry -> abort state machine."""

from __future__ import annotations

import unittest

from core import action_watchdog


class FakeClock:

  def __init__(self):
    self.t = 100.0

  def __call__(self):
    return self.t

  def advance(self, s):
    self.t += s


class ActionWatchdogTest(unittest.TestCase):

  def setUp(self):
    self.clock = FakeClock()
    self.wd = action_watchdog.ActionWatchdog(
        timeout_s=10.0, max_attempts=3, grace_s=4.0, clock=self.clock
    )

  def test_idle_has_no_text_and_blocks_nothing(self):
    self.assertIsNone(self.wd.phase())
    self.assertEqual("", self.wd.heartbeat_text())
    self.assertIsNone(self.wd.blocks("pick"))
    self.assertEqual({"state": "idle"}, self.wd.status())

  def test_first_issue_starts_attempt_one(self):
    info = self.wd.on_issued("pick", {}, plan_step=1)
    self.assertEqual(1, info["attempt"])
    self.assertFalse(info["retry"])
    self.assertEqual(action_watchdog.WAITING, self.wd.phase())
    self.assertEqual("pending", self.wd.status()["state"])
    self.assertIn("pick attempt 1/3", self.wd.heartbeat_text())
    self.assertIn("confirm_action", self.wd.heartbeat_text())
    self.assertEqual(1, self.wd.pending.plan_step)

  def test_other_verified_action_is_blocked_while_pending(self):
    self.wd.on_issued("pick", {})
    block = self.wd.blocks("place")
    self.assertIsNotNone(block)
    self.assertIn("pick", block["error"])
    # Same action and non-verified tools are not blocked.
    self.assertIsNone(self.wd.blocks("pick"))
    self.assertIsNone(self.wd.blocks("detect"))
    self.assertIsNone(self.wd.blocks("ack"))

  def test_early_reissue_is_merged_not_counted(self):
    self.wd.on_issued("pick", {})
    self.clock.advance(3.0)
    info = self.wd.on_issued("pick", {})
    self.assertTrue(info["merged"])
    self.assertEqual(1, info["attempt"])
    self.assertEqual(1, self.wd.pending.attempt)
    # Deadline is unchanged: 7 s remain.
    self.assertAlmostEqual(7.0, self.wd.remaining(), places=3)

  def test_timeout_then_reissue_counts_as_retry(self):
    self.wd.on_issued("pick", {})
    self.clock.advance(10.0)
    self.assertEqual(action_watchdog.TIMED_OUT, self.wd.phase())
    text = self.wd.heartbeat_text()
    self.assertIn("TIMED OUT", text)
    self.assertIn("re-issue pick()", text)
    self.assertIn("LATEST frame", text)
    self.assertIn("attempt 2/3", text)
    # The notice starts the grace period.
    self.assertIsNotNone(self.wd.pending.timeout_notified_at)
    self.clock.advance(1.0)
    info = self.wd.on_issued("pick", {})
    self.assertTrue(info["retry"])
    self.assertEqual(2, info["attempt"])
    self.assertEqual(action_watchdog.WAITING, self.wd.phase())

  def test_overdue_after_grace_without_reissue(self):
    self.wd.on_issued("pick", {})
    self.clock.advance(10.0)
    self.wd.heartbeat_text()  # notice delivered at t+10
    self.clock.advance(3.9)
    self.assertEqual(action_watchdog.TIMED_OUT, self.wd.phase())
    self.clock.advance(0.2)
    self.assertEqual(action_watchdog.OVERDUE, self.wd.phase())

  def test_exhausted_after_last_attempt_times_out(self):
    self.wd.on_issued("pick", {})
    for expected in (2, 3):
      self.clock.advance(10.0)
      info = self.wd.on_issued("pick", {})
      self.assertEqual(expected, info["attempt"])
    self.clock.advance(10.0)
    self.assertEqual(action_watchdog.EXHAUSTED, self.wd.phase())
    # A fourth issue is refused instead of re-arming.
    info = self.wd.on_issued("pick", {})
    self.assertTrue(info["exhausted"])
    self.assertEqual(3, self.wd.pending.attempt)

  def test_confirm_clears_pending(self):
    self.wd.on_issued("place", {}, plan_step=3)
    self.clock.advance(4.0)
    result = self.wd.confirm("the block rests next to the ball")
    self.assertEqual("confirmed", result["status"])
    self.assertEqual("place", result["action"])
    self.assertEqual(1, result["attempts_used"])
    self.assertIsNone(self.wd.pending)
    self.assertEqual("nothing_pending", self.wd.confirm()["status"])

  def test_fail_queues_two_notices_then_goes_quiet(self):
    self.wd.on_issued("pick", {})
    info = self.wd.fail("pick was not executed after 3 attempts.")
    self.assertEqual("pick", info["action"])
    self.assertIsNone(self.wd.pending)
    first = self.wd.heartbeat_text()
    self.assertIn("TASK ABORTED", first)
    self.assertIn("stay idle", first)
    second = self.wd.heartbeat_text()
    self.assertIn("TASK ABORTED", second)
    self.assertEqual("", self.wd.heartbeat_text())

  def test_reset_clears_everything(self):
    self.wd.on_issued("pick", {})
    self.wd.fail("x")
    self.wd.reset()
    self.assertIsNone(self.wd.pending)
    self.assertEqual("", self.wd.heartbeat_text())


if __name__ == "__main__":
  unittest.main()
