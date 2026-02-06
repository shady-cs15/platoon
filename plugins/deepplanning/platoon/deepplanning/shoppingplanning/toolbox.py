# Tests ToolBox for shopping planning tasks.
# Cmd to run tests:
# PYTHONPATH=/root/platoon/plugins/deepplanning uv run pytest -q \
# platoon/deepplanning/shoppingplanning/tests/test_toolbox.py

import sys
import json
import importlib

from pathlib import Path
from typing import Any


class ShoppingPlanningToolBox:
    def __init__(self, case_dir: Path):
        case_dir = Path(case_dir)
        if not case_dir.exists():
            raise FileNotFoundError(f"case_dir not found: {case_dir}")

        # This file is: shoppingplanning/tools.py
        # Tool modules live in: shoppingplanning/tools/*.py
        tools_dir = Path(__file__).resolve().parent / "tools"
        if not tools_dir.exists():
            raise FileNotFoundError(f"tools dir not found: {tools_dir}")

        # IMPORTANT: tool modules do `from base_shopping_tool import ...`
        # so we need tools_dir on sys.path so `import base_shopping_tool` works.
        tools_dir_str = str(tools_dir)
        if tools_dir_str not in sys.path:
            sys.path.insert(0, tools_dir_str)

        # Import registry holder
        base = importlib.import_module("base_shopping_tool")

        # Import every tool module to trigger @register_tool decorators.
        # (Skip __init__.py and base_shopping_tool.py)
        for p in tools_dir.glob("*.py"):
            if p.name in {"__init__.py", "base_shopping_tool.py"}:
                continue
            importlib.import_module(p.stem)

        tool_registry = getattr(base, "TOOL_REGISTRY", None)
        if not tool_registry:
            raise RuntimeError("TOOL_REGISTRY is empty; tool modules may not have imported correctly")

        cfg = {"database_path": str(case_dir)}

        # Instantiate all tools
        self._tools = {}
        for tool_name, tool_cls in tool_registry.items():
            self._tools[tool_name] = tool_cls(cfg=cfg)
    
    def _call(self, name: str, **kwargs: Any) -> Any:
        tool = self._tools[name]
        out = tool.call(json.dumps(kwargs, ensure_ascii=False))
        # tools often return JSON strings; parse if possible
        try:
            return json.loads(out)
        except Exception:
            return out

    # ---- thin method wrappers (one per tool) ----
    def search_products(self, query: str, limit: int = 20) -> Any:
        """Semantic search over the product catalog using a natural language query."""
        return self._call("search_products", query=query, limit=limit)

    def add_product_to_cart(self, product_id: str, quantity: int = 1) -> Any:
        """Add a product to the cart (validates existence and stock; updates cart summary)."""
        return self._call("add_product_to_cart", product_id=product_id, quantity=quantity)

    def get_cart_info(self) -> Any:
        """Return the current shopping cart items and summary (total count / total price)."""
        return self._call("get_cart_info")

    def filter_by_brand(self, brand_names: list[str], product_ids: list[str] | None = None) -> Any:
        """Filter product IDs by brand name(s) (OR logic across brands)."""
        return self._call("filter_by_brand", brand_names=brand_names, product_ids=product_ids)

    def filter_by_color(self, colors: list[str], product_ids: list[str] | None = None) -> Any:
        """Filter product IDs by color(s) (OR logic across colors)."""
        return self._call("filter_by_color", colors=colors, product_ids=product_ids)

    def filter_by_size(self, sizes: list[str], product_ids: list[str] | None = None) -> Any:
        """Filter product IDs by size(s) (OR logic across sizes)."""
        return self._call("filter_by_size", sizes=sizes, product_ids=product_ids)

    def filter_by_applicable_coupons(self, coupon_names: list[str], product_ids: list[str] | None = None) -> Any:
        """Filter product IDs by requiring all provided coupon names to be applicable."""
        return self._call("filter_by_applicable_coupons", coupon_names=coupon_names, product_ids=product_ids)

    def filter_by_range(
        self,
        condition_key: str, 
        operator: str, 
        value: float, 
        product_ids: list[str] | None = None,
    ) -> Any:
        """Filter product IDs by numeric condition: (condition_key operator value)."""
        return self._call(
            "filter_by_range",
            condition_key=condition_key,
            operator=operator,
            value=value,
            product_ids=product_ids,
        )

    def sort_products(self, sort_by: str, product_ids: list[str] | None = None, order: str = "desc") -> Any:
        """Sort product IDs by a feature dimension and order ('asc' or 'desc')."""
        return self._call("sort_products", sort_by=sort_by, product_ids=product_ids, order=order)

    def get_product_details(self, product_ids: list[str]) -> Any:
        """Fetch full product details for the given product IDs."""
        return self._call("get_product_details", product_ids=product_ids)

    def calculate_transport_time(
        self,
        product_id: str,
        destination_address: str, 
        provider: str | None = None,
    ) -> Any:
        """Calculate estimated delivery time (days) for a product to a destination address."""
        return self._call(
            "calculate_transport_time",
            product_id=product_id,
            destination_address=destination_address,
            provider=provider,
        )

    def get_user_info(self, user_id: str | None = None) -> Any:
        """Retrieve user profile info (address, body measurements, etc.); optional user_id."""
        return self._call("get_user_info", user_id=user_id)

    def delete_product_from_cart(self, product_id: str, quantity: int = 1) -> Any:
        """Remove/decrease a product in the cart; updates cart summary (defaults to 1)."""
        return self._call("delete_product_from_cart", product_id=product_id, quantity=quantity)

    def add_coupon_to_cart(self, coupon_name: str, quantity: int = 1) -> Any:
        """Add a coupon to the cart (validates ownership/threshold); defaults quantity=1."""
        return self._call("add_coupon_to_cart", coupon_name=coupon_name, quantity=quantity)

    def delete_coupon_from_cart(self, coupon_name: str, quantity: int = 1) -> Any:
        """Remove/decrease a coupon in the cart; updates discounted summary (defaults to 1)."""
        return self._call("delete_coupon_from_cart", coupon_name=coupon_name, quantity=quantity)