import time
import threading
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.messages import AIMessage


class StatsCallbackHandler(BaseCallbackHandler):
    """Callback handler that tracks LLM calls, tool calls, token usage, and timing."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.llm_calls = 0
        self.tool_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.total_llm_seconds = 0.0
        self._llm_start_times: Dict[str, float] = {}
        self.current_agent: Optional[str] = None
        self.llm_timings: Dict[str, float] = {}

    def set_current_agent(self, name: str) -> None:
        with self._lock:
            self.current_agent = name

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        run_id = str(kwargs.get("run_id", ""))
        with self._lock:
            self.llm_calls += 1
            self._llm_start_times[run_id] = time.perf_counter()

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        **kwargs: Any,
    ) -> None:
        run_id = str(kwargs.get("run_id", ""))
        with self._lock:
            self.llm_calls += 1
            self._llm_start_times[run_id] = time.perf_counter()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", ""))
        elapsed = 0.0
        with self._lock:
            start = self._llm_start_times.pop(run_id, None)
            if start is not None:
                elapsed = time.perf_counter() - start
                self.total_llm_seconds += elapsed
                if self.current_agent:
                    self.llm_timings[self.current_agent] = (
                        self.llm_timings.get(self.current_agent, 0.0) + elapsed
                    )

        try:
            generation = response.generations[0][0]
        except (IndexError, TypeError):
            return

        usage_metadata = None
        if hasattr(generation, "message"):
            message = generation.message
            if isinstance(message, AIMessage) and hasattr(message, "usage_metadata"):
                usage_metadata = message.usage_metadata

        if usage_metadata:
            with self._lock:
                self.tokens_in += usage_metadata.get("input_tokens", 0)
                self.tokens_out += usage_metadata.get("output_tokens", 0)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", ""))
        with self._lock:
            self._llm_start_times.pop(run_id, None)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            self.tool_calls += 1

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "llm_calls": self.llm_calls,
                "tool_calls": self.tool_calls,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "total_llm_seconds": self.total_llm_seconds,
                "llm_timings": dict(self.llm_timings),
            }
