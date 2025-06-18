from pathlib import Path
from typing import NamedTuple, Optional

from .registry import ToolsRegistry

# Create singleton instance
registry = ToolsRegistry()

# Export commonly used functions at module level
get_tool = registry.get_tool
list_tools = registry.list_tools
get_tool_path = registry.get_tool_path
get_tool_version = registry.get_tool_version
get_tools_by_feature = registry.get_tools_by_feature
get_tool_tutorials_folder = registry.get_tool_tutorials_folder
get_tool_prompt_template = registry.get_tool_prompt_template

# Export new registration functions
register_tool = registry.register_tool
unregister_tool = registry.unregister_tool
update_tool = registry.update_tool
add_tool_tutorials = registry.add_tool_tutorials


class TutorialInfo(NamedTuple):
    """Stores information about a tutorial"""

    path: Path
    title: str
    summary: str
    score: Optional[float] = None
    content: Optional[str] = None
