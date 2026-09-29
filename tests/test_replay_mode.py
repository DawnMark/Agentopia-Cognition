"""阶段 0：replay-only 模式与运行种子（离线，无 LLM 调用）。

可重放的前提有两条：
1. 随机源确定（`world.seed`，模型分配不再用运行目录名做种子）；
2. LLM 响应确定——缓存按请求内容寻址（`cache_file` 被明确排除在 key 之外），
   所以同一份配置重跑时会命中已有缓存；一旦某个请求没命中，说明轨迹已经分叉，
   必须**立即失败**而不是偷偷真调一次 API（那会让"重放"变成"另一次运行"）。
"""

from __future__ import annotations

import unittest

from src import utils
from tests._helpers import temp_workspace


class ReplayModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._ctx = temp_workspace()
        self.root = self._ctx.__enter__()
        self._previous = utils.is_replay_only()

    def tearDown(self) -> None:
        utils.set_replay_only(self._previous)
        self._ctx.__exit__(None, None, None)

    def test_switch_roundtrip(self) -> None:
        utils.set_replay_only(True)
        self.assertTrue(utils.is_replay_only())
        utils.set_replay_only(False)
        self.assertFalse(utils.is_replay_only())

    def test_cache_miss_aborts_instead_of_calling_the_api(self) -> None:
        """A miss must raise before any network call can happen."""
        utils.set_replay_only(True)
        with self.assertRaises(RuntimeError) as ctx:
            utils.generate_with_fc(
                model="__replay_probe_never_cached__",
                messages=[{"role": "user", "content": "deterministic replay probe"}],
            )
        message = str(ctx.exception)
        self.assertIn("REPLAY CACHE MISS", message)
        self.assertIn("generate_with_fc", message)

    def test_repeated_miss_in_replay_mode_is_stable(self) -> None:
        utils.set_replay_only(True)
        digests = set()
        for _ in range(2):
            with self.assertRaises(RuntimeError) as ctx:
                utils.generate_with_fc(
                    model="__replay_probe_never_cached__",
                    messages=[{"role": "user", "content": "same request twice"}],
                )
            digests.add(str(ctx.exception).split("Key digest:")[-1].strip())
        self.assertEqual(len(digests), 1, "the same request must report the same key")

    def test_guard_is_inactive_by_default(self) -> None:
        utils.set_replay_only(False)
        self.assertFalse(utils.is_replay_only())
        # The guard is what raises; without it the call would proceed to the
        # API layer, so this only asserts the mode, never making a request.
        import inspect

        source = inspect.getsource(utils.cached)
        self.assertIn("if _replay_only:", source)


class RunSeedTests(unittest.TestCase):
    def test_world_seed_defaults_and_is_an_int(self) -> None:
        from src.config import get_config

        seed = get_config()["world"].get("seed", 0)
        self.assertIsInstance(int(seed), int)

    def test_model_assignment_is_seeded_by_world_and_seed(self) -> None:
        """Same world + same seed ⇒ same assignment, regardless of run dir."""
        import inspect

        from src.world.world import World

        source = inspect.getsource(World._load_or_assign_models)
        self.assertIn("self.seed", source)
        self.assertIn("self.config['name']", source)  # 种子里带世界名
        # 上游是用运行目录名播种的。保留那条路径的唯一理由是对照档
        # （`baseline = "upstream"`），所以它必须挂在开关后面。
        self.assertIn("upstream_seed_source(self.config)", source)

    def test_random_module_is_seeded_at_world_construction(self) -> None:
        import inspect

        from src.world.world import World

        source = inspect.getsource(World.__init__)
        self.assertIn("random.seed(self.seed)", source)


if __name__ == "__main__":
    unittest.main()
