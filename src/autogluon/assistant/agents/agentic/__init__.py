"""Agentic (multi-turn, tool-using) agents backed by the Claude Agent SDK (P3).

This subpackage is an OPTIONAL extension. Nothing here is imported at MLZero start-up;
the SDK is lazy-imported only when an agent is explicitly configured to use it, so the
default Bedrock install is unaffected and importing the base package never requires the
`claude-agent-sdk` dependency.
"""
