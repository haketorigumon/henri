# Complications

Case studies of features that were more complex than expected.

## Status Updates

**Goal:** Show a spinner when the agent is preparing to use a tool.

**The problem:** Ollama doesn't signal when tool use begins. It streams text, goes silent for ~7 seconds while generating the tool call JSON, then sends the complete tool call. No early signal exists.

**What didn't work:**
- Timeout detection in the provider - broke async iteration
- Show status after each text chunk - Rich's spinner overwrites streamed text
- Show status after newlines - text often doesn't end with newlines

**Solution:** Timer at the agent level. After each text chunk, schedule "Pondering..." to appear in 500ms. Cancel when more text arrives. If nothing comes, the spinner shows on a new line.

**Complexity added:** asyncio task scheduling, cancel logic, newline management.

**To remove this feature, delete:**
- `StreamEvent.tool_use_started` in `providers/base.py`
- `yield StreamEvent(tool_use_started=True)` in all providers
- `_pondering_task`, `_schedule_pondering`, `_cancel_pondering` in `agent.py`
- `_status`, `_show_status`, `_hide_status` in `agent.py`
- All calls to the above methods in `agent.py`
- `import asyncio` and `from rich.status import Status` in `agent.py` (if unused elsewhere)
