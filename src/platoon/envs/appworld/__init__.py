from munch import Munch

# Monkey path deepcopy for munch.
Munch.__deepcopy__ = lambda self, memo: self.copy()

from appworld.common.safety_guard import ALLOWED_MODULE_NAMES
# Monkey patch for async tool support in appworld.
ALLOWED_MODULE_NAMES.add("asyncio")