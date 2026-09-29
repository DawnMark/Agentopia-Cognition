"""默认运行必须能构造出来：`RoleAgent.__init__` 的属性顺序守卫。

背景（2026-09-27，阶段 5 验收后）：`method_hints` 变成默认开启后，一次**不带任何
`--cognition-*` 开关**的运行立刻崩了：

    AttributeError: 'RoleAgent' object has no attribute 'hint_config'

原因不在新代码，而在阶段 4 埋下的一行——活动观察者接线用
`if self.shadow.enabled or self.hint_config.enabled:` 判断要不要挂观察者，而
`self.hint_config` 要到 `__init__` 后面才构造。此前每次真跑都带着 `--cognition-shadow`，
`or` 在第一个条件就短路了，所以这条路径从来没被走到；**关闭全部认知开关的默认运行**
才会踩到它。

这类"属性先用后赋值"的错误对行为测试不可见（要构造一个完整 RoleAgent 才能触发），
所以这里用静态检查把整类问题钉死：在 `__init__` 自己的作用域里，任何 `self.X` 的读取
都必须出现在同一次 `__init__` 中 `self.X = ...` 的赋值之后。
"""

from __future__ import annotations

import ast
import io
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ROLE_AGENT = REPO_ROOT / "src" / "agents" / "role_agent.py"


def _loads_before_assignment(
    class_name: str, method_name: str, *, source: str | None = None
) -> list[str]:
    """`self.X` reads that happen before the first `self.X = ...` in one method."""
    text = source if source is not None else io.open(ROLE_AGENT, encoding="utf-8").read()
    tree = ast.parse(text)
    target = None
    class_level: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                # Methods/properties exist on the class, not on the instance:
                # calling `self._persona_traits()` is never an ordering problem.
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    class_level.add(item.name)
                elif isinstance(item, ast.Assign):
                    class_level.update(_self_attrs_written(item.targets))
                    class_level.update(
                        t.id for t in item.targets if isinstance(t, ast.Name)
                    )
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    class_level.add(item.target.id)
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    target = item
    assert target is not None, f"{class_name}.{method_name} not found"

    assigned: set[str] = set()
    offenders: list[str] = []

    class Visitor(ast.NodeVisitor):
        """Walk only this method's own statements, not nested functions."""

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            return  # nested defs run later, or not at all

        visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]
        visit_Lambda = visit_FunctionDef  # type: ignore[assignment]

        def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
            for self_attr in _self_attrs_written(node.targets):
                assigned.add(self_attr)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
            for self_attr in _self_attrs_written([node.target]):
                assigned.add(self_attr)
            self.generic_visit(node)

        def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and isinstance(node.ctx, ast.Load)
                and node.attr not in assigned
                and node.attr not in class_level
            ):
                offenders.append(node.attr)
            self.generic_visit(node)

    # Visit the body statement by statement: `target` is itself a FunctionDef,
    # and the visitor deliberately ignores nested function definitions.
    visitor = Visitor()
    for statement in target.body:
        visitor.visit(statement)
    return offenders


def _self_attrs_written(targets: list[ast.expr]) -> list[str]:
    out: list[str] = []
    for target in targets:
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
            if target.value.id == "self":
                out.append(target.attr)
        elif isinstance(target, (ast.Tuple, ast.List)):
            out.extend(_self_attrs_written(list(target.elts)))
    return out


class RoleAgentInitOrderTests(unittest.TestCase):
    def test_no_attribute_is_read_before_it_is_assigned(self) -> None:
        offenders = _loads_before_assignment("RoleAgent", "__init__")
        self.assertEqual(
            offenders,
            [],
            "RoleAgent.__init__ 在赋值之前读了这些属性（默认配置的运行会直接崩）："
            f"{sorted(set(offenders))}",
        )

    def test_the_guard_actually_detects_the_bug_it_was_written_for(self) -> None:
        """反向验证：同样的顺序错误必须被检查报出来（用合成源码，不动真文件）。"""
        broken = """
class RoleAgent:
    def _observe(self):
        return None

    def __init__(self, config):
        self.shadow = build_shadow(config)
        if self.shadow.enabled or self.hint_config.enabled:
            self.observer = self._observe
        self.hint_config = build_hints(config)
"""
        self.assertIn(
            "hint_config", _loads_before_assignment("RoleAgent", "__init__", source=broken)
        )
        fixed = """
class RoleAgent:
    def _observe(self):
        return None

    def __init__(self, config):
        self.shadow = build_shadow(config)
        self.hint_config = build_hints(config)
        if self.shadow.enabled or self.hint_config.enabled:
            self.observer = self._observe
"""
        self.assertEqual(
            _loads_before_assignment("RoleAgent", "__init__", source=fixed), []
        )

    def test_the_real_init_constructs_the_hint_provider_before_wiring_observers(self) -> None:
        source = io.open(ROLE_AGENT, encoding="utf-8").read()
        assigned = source.index("self.hint_config = HintConfig.from_world_config")
        wired = source.index("if self.shadow.enabled or self.hint_config.enabled:")
        self.assertLess(assigned, wired)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
