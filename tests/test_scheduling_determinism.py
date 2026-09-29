"""④ MessageCenter 同时间戳动作必须确定性结算（离线，无 LLM 调用）。

背景：CONTACT 阶段所有 agent 并发执行，`MessageCenter._msgs` 的入队顺序由
线程调度决定。同一个角色在同一个 slot 里可以发多次动作（上限
`contact.n_action_per_slot`，默认 10），它们的时间戳完全相同——
旧代码在这些位置只比较时间戳：

- `responses_all.sort` 的键完全相同 → 依赖插入顺序；
- "同一 responder 只保留最后一条响应" 用 `<=` 比较后按迭代顺序覆盖；
- 冲突解决 `_sort_key = (priority, msg_time)` 同票 → 依赖列表构造顺序。

于是"同一 slot 里先答应后反悔"到底算不算数，取决于线程时序。
修复后每个动作携带 RoleAgent 分配的单调整数 `seq`，作为最终裁决依据。

这些测试会打乱消息入队顺序，要求结算结果逐字节一致。
"""

from __future__ import annotations

import random
import unittest
from typing import Dict, List, Tuple

from src.world.clock import Clock, Stage
from src.world.scheduling import MessageCenter

PROPOSER = "甲"
INVITEE = "乙"
ACTIVITY = "爬山"
ACTIVITY_TIME = "Y2020-W01-activity-D2"
# A proposal is answered in a *later* slot: responses must be strictly after
# the proposal they answer, so only the responses share a timestamp here.
PROPOSE_SLOT = "Y2020-W01-contact-S1"
RESPOND_SLOT = "Y2020-W01-contact-S2"
OTHER_ACTIVITY = "看展"


def _clock() -> Clock:
    clock = Clock(start_year=2020, start_week=1)
    clock.set_stage(Stage.CONTACT)
    clock.set_slot(2)
    return clock


def _propose(
    clock: Clock, *, proposer: str, invited: List[str], name: str = ACTIVITY, seq: int = 1
) -> Dict:
    return {
        "time": PROPOSE_SLOT,
        "seq": seq,
        "from": proposer,
        "type": "propose_joint_activity",
        "activity_name": name,
        "invited_persons": invited,
        "required_participants": invited,
        "activity_time": ACTIVITY_TIME,
        "location": "山脚",
        "raw_action": f'propose("{name}")',
        "message": "一起去？",
        "proposal": "周末爬山",
    }


def _respond(
    clock: Clock,
    *,
    responder: str,
    proposer: str,
    decision: str,
    name: str = ACTIVITY,
    seq: int = 1,
) -> Dict:
    return {
        "time": RESPOND_SLOT,
        "seq": seq,
        "from": responder,
        "to": proposer,
        "type": "respond_invitation",
        "activity_name": name,
        "decision": decision,
        "raw_action": f'respond("{name}", {decision})',
        "message": "好" if decision == "yes" else "不行",
    }


def _digest(mc: MessageCenter, people: List[str]) -> Tuple:
    """Canonical, order-independent view of one settlement outcome."""
    activities = sorted(
        (s.activity_id, s.status, tuple(sorted(s.participants or [])))
        for s in mc.created_activities
    )
    per_person = {
        name: sorted(
            (s.activity_id, s.status) for s in mc.get_scheduling_result(name)
        )
        for name in people
    }
    notifications = {name: sorted(mc.get_notifications(name)) for name in people}
    return activities, per_person, notifications


class SchedulingDeterminismTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = _clock()
        self.people = [PROPOSER, INVITEE]

    def _settle(self, msgs: List[Dict]) -> MessageCenter:
        mc = MessageCenter(world_name="regression_world", clock=self.clock)
        for m in msgs:
            mc.add(m)
        mc.confirm_schedule()
        return mc

    def _assert_order_independent(self, msgs: List[Dict], *, expected_digest=None) -> MessageCenter:
        baseline = self._settle(msgs)
        baseline_digest = _digest(baseline, self.people)
        if expected_digest is not None:
            self.assertEqual(baseline_digest, expected_digest)

        rng = random.Random(20240925)
        for _ in range(25):
            shuffled = list(msgs)
            rng.shuffle(shuffled)
            mc = self._settle(shuffled)
            self.assertEqual(
                _digest(mc, self.people),
                baseline_digest,
                "settlement depends on message insertion order",
            )
        return baseline

    # -- the core case: same responder, same slot, both decisions ----------
    def test_contradicting_responses_resolve_by_seq(self) -> None:
        msgs = [
            _propose(self.clock, proposer=PROPOSER, invited=[INVITEE], seq=1),
            _respond(self.clock, responder=INVITEE, proposer=PROPOSER, decision="no", seq=2),
            _respond(self.clock, responder=INVITEE, proposer=PROPOSER, decision="yes", seq=3),
        ]
        mc = self._assert_order_independent(msgs)

        created = [s for s in mc.created_activities if s.status == "created"]
        self.assertEqual(len(created), 1, "the later 'yes' must win")
        self.assertEqual(sorted(created[0].participants), sorted(self.people))

    def test_contradicting_responses_the_other_way_round(self) -> None:
        msgs = [
            _propose(self.clock, proposer=PROPOSER, invited=[INVITEE], seq=1),
            _respond(self.clock, responder=INVITEE, proposer=PROPOSER, decision="yes", seq=2),
            _respond(self.clock, responder=INVITEE, proposer=PROPOSER, decision="no", seq=3),
        ]
        mc = self._assert_order_independent(msgs)

        created = [s for s in mc.created_activities if s.status == "created"]
        self.assertEqual(created, [], "the later 'no' must win")

    # -- same person, same slot, conflicting arrangements ------------------
    def test_conflicting_arrangements_resolve_by_seq(self) -> None:
        """Two proposers invite the same person to the same time slot."""
        third = "丙"
        msgs = [
            _propose(self.clock, proposer=PROPOSER, invited=[INVITEE], seq=1),
            _propose(
                self.clock,
                proposer=third,
                invited=[INVITEE],
                name=OTHER_ACTIVITY,
                seq=1,
            ),
            # The invitee accepts both; the later acceptance must survive, and
            # the earlier one is turned into a 'no' with a notification.
            _respond(self.clock, responder=INVITEE, proposer=PROPOSER, decision="yes", seq=2),
            _respond(
                self.clock,
                responder=INVITEE,
                proposer=third,
                decision="yes",
                name=OTHER_ACTIVITY,
                seq=3,
            ),
        ]
        self.people = [PROPOSER, INVITEE, third]
        mc = self._assert_order_independent(msgs)

        created = sorted(
            s.activity_name for s in mc.created_activities if s.status == "created"
        )
        self.assertEqual(created, [OTHER_ACTIVITY])
        self.assertNotEqual(mc.get_notifications(INVITEE), [])

    # -- messages without seq stay deterministic ---------------------------
    def test_missing_seq_falls_back_to_zero_and_stays_deterministic(self) -> None:
        msgs = [
            {k: v for k, v in _propose(self.clock, proposer=PROPOSER, invited=[INVITEE]).items() if k != "seq"},
            {k: v for k, v in _respond(self.clock, responder=INVITEE, proposer=PROPOSER, decision="yes").items() if k != "seq"},
        ]
        mc = self._assert_order_independent(msgs)
        self.assertEqual(len([s for s in mc.created_activities if s.status == "created"]), 1)

    # -- the failure mode stays deterministic ------------------------------
    def test_duplicate_proposals_raise_regardless_of_order(self) -> None:
        msgs = [
            _propose(self.clock, proposer=PROPOSER, invited=[INVITEE], seq=1),
            _propose(self.clock, proposer=PROPOSER, invited=[INVITEE], seq=2),
        ]
        rng = random.Random(7)
        for _ in range(5):
            rng.shuffle(msgs)
            with self.assertRaises(RuntimeError):
                self._settle(list(msgs))

    # -- cancel still voids the proposal -----------------------------------
    def test_cancel_voids_the_proposal_deterministically(self) -> None:
        cancel = {
            "time": RESPOND_SLOT,
            "seq": 3,
            "from": PROPOSER,
            "type": "cancel_joint_activity",
            "activity_name": ACTIVITY,
            "invited_persons": [INVITEE],
            "message": "取消了",
            "raw_action": f'cancel("{ACTIVITY}")',
        }
        msgs = [
            _propose(self.clock, proposer=PROPOSER, invited=[INVITEE], seq=1),
            _respond(self.clock, responder=INVITEE, proposer=PROPOSER, decision="yes", seq=2),
            cancel,
        ]
        mc = self._assert_order_independent(msgs)
        self.assertEqual([s for s in mc.created_activities if s.status == "created"], [])
        self.assertEqual(
            {s.status for s in mc.get_scheduling_result(PROPOSER)}, {"canceled"}
        )


if __name__ == "__main__":
    unittest.main()
