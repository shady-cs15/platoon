from __future__ import annotations

import inspect
from pathlib import Path

from platoon.agents.actions.common import finish
from platoon.envs.base import Task
from platoon.envs.codeact import ForkableCodeExecutor, IPythonCodeExecutor, safe_asyncio

from platoon.deepplanning.shoppingplanning.toolbox import ShoppingPlanningToolBox


class ShoppingPlanningCodeExecutor(IPythonCodeExecutor, ForkableCodeExecutor):
    """Code executor for ShoppingPlanning.

    Exposes ShoppingPlanningToolBox methods as Python-callable actions inside the sandbox.
    """

    def __init__(
        self,
        task: Task,
        case_dir: Path,
    ) -> None:
        self.case_dir = Path(case_dir)
        self.toolbox = ShoppingPlanningToolBox(self.case_dir)

        actions = (
            finish,
            # toolbox actions (must be plain callables)
            self.toolbox.search_products,
            self.toolbox.filter_by_brand,
            self.toolbox.filter_by_color,
            self.toolbox.filter_by_size,
            self.toolbox.filter_by_applicable_coupons,
            self.toolbox.filter_by_range,
            self.toolbox.sort_products,
            self.toolbox.get_product_details,
            self.toolbox.calculate_transport_time,
            self.toolbox.get_user_info,
            self.toolbox.add_product_to_cart,
            self.toolbox.delete_product_from_cart,
            self.toolbox.get_cart_info,
            self.toolbox.add_coupon_to_cart,
            self.toolbox.delete_coupon_from_cart,
            safe_asyncio,
        )

        super().__init__(task, actions=actions)

    async def describe_action_space(self) -> str:
        """Return a string description of the available tool functions.

        CodeAct agents will see this in the prompt (via env.reset()).
        """
        fns = [
            self.toolbox.search_products,
            self.toolbox.filter_by_brand,
            self.toolbox.filter_by_color,
            self.toolbox.filter_by_size,
            self.toolbox.filter_by_applicable_coupons,
            self.toolbox.filter_by_range,
            self.toolbox.sort_products,
            self.toolbox.get_product_details,
            self.toolbox.calculate_transport_time,
            self.toolbox.get_user_info,
            self.toolbox.add_product_to_cart,
            self.toolbox.delete_product_from_cart,
            self.toolbox.get_cart_info,
            self.toolbox.add_coupon_to_cart,
            self.toolbox.delete_coupon_from_cart,
            finish,
        ]

        lines = ["Available actions (Python functions you can call):"]
        for fn in fns:
            sig = inspect.signature(fn)
            doc = (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else ""
            lines.append(f"- {fn.__name__}{sig}" + (f": {doc}" if doc else ""))

        lines.append("")
        lines.append("Use finish('...') when you are done.")
        return "\n".join(lines)

    async def reset(self) -> ShoppingPlanningCodeExecutor:
        return self

    async def fork(self, task: Task) -> ShoppingPlanningCodeExecutor:
        # New executor instance, same case_dir unless you want per-fork isolation
        return ShoppingPlanningCodeExecutor(task=task, case_dir=self.case_dir)