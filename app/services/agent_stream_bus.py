# -*- coding: utf-8 -*-
"""兼容层：统一转发到 runtime.core.live_stream_bus。"""

from harness.core.live_stream_bus import AgentStreamBus, live_stream_bus

agent_stream_bus = live_stream_bus

__all__ = ["AgentStreamBus", "agent_stream_bus"]
