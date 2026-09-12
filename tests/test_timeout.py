"""Tests for two-tier timeout mechanism."""

from __future__ import annotations

import threading
import time

import pytest

import rocq_mcp.server as _server
from rocq_mcp import jobs
from rocq_mcp.interactive import (
    _is_timeout_eligible,
    _compute_hard_timeout,
    _PET_TIMEOUT_GRACE,
)
from rocq_mcp.server import _run_with_pet
from tests.conftest import (
    make_lifespan_state,
    mock_pet as _mock_pet,
    patch_psutil_rss as _patch_psutil_rss,
)

# ---------------------------------------------------------------------------
# _is_timeout_eligible
# ---------------------------------------------------------------------------


class TestIsTimeoutEligible:
    """Test tactic eligibility for Rocq Timeout wrapping."""

    def test_normal_tactic(self):
        assert _is_timeout_eligible("auto.") is True

    def test_tactic_with_spaces(self):
        assert _is_timeout_eligible("  auto.  ") is True

    def test_bullet_dash(self):
        assert _is_timeout_eligible("- auto.") is False

    def test_bullet_plus(self):
        assert _is_timeout_eligible("+ auto.") is False

    def test_bullet_star(self):
        assert _is_timeout_eligible("* auto.") is False

    def test_no_dot(self):
        assert _is_timeout_eligible("auto") is False

    def test_brace_open(self):
        # "{ auto. }" does not end with "." — not eligible
        assert _is_timeout_eligible("{ auto. }") is False

    def test_brace_close(self):
        assert _is_timeout_eligible("}") is False

    def test_intros(self):
        assert _is_timeout_eligible("intros.") is True

    def test_complex_tactic(self):
        assert _is_timeout_eligible("rewrite IH; reflexivity.") is True

    def test_empty(self):
        assert _is_timeout_eligible("") is False

    def test_only_dot(self):
        assert _is_timeout_eligible(".") is True

    def test_numbered_goal(self):
        assert _is_timeout_eligible("1: auto.") is True

    def test_double_bullet(self):
        assert _is_timeout_eligible("-- auto.") is False

    def test_whitespace_before_bullet(self):
        assert _is_timeout_eligible("  - auto.") is False

    def test_semicolon_chain(self):
        assert _is_timeout_eligible("split; auto.") is True


# ---------------------------------------------------------------------------
# _compute_hard_timeout
# ---------------------------------------------------------------------------


class TestComputeHardTimeout:
    """Test hard timeout computation."""

    def test_default_grace(self):
        assert _compute_hard_timeout(30.0) == 30.0 + _PET_TIMEOUT_GRACE

    def test_small_timeout(self):
        assert _compute_hard_timeout(1.0) == 1.0 + _PET_TIMEOUT_GRACE

    def test_zero(self):
        assert _compute_hard_timeout(0.0) == _PET_TIMEOUT_GRACE


# ---------------------------------------------------------------------------
# Timeout error message — actionable retry hint
# ---------------------------------------------------------------------------


class TestTimeoutErrorHint:
    """The pet timeout error string must include an actionable retry hint
    that names the per-call ``timeout=`` arg and the env-var cap."""

    @pytest.fixture(autouse=True)
    def _reset_pet_state(self, monkeypatch):
        _server._pet_semaphore = None
        monkeypatch.setattr(_server, "_pet_lock", threading.Lock())
        yield
        _server._pet_semaphore = None

    @pytest.fixture(autouse=True)
    def _fast_watchdog(self, monkeypatch):
        monkeypatch.setattr(_server, "_MEMORY_WATCHDOG_INTERVAL", 0.01)

    @pytest.mark.asyncio
    async def test_timeout_error_includes_retry_hint(self, monkeypatch):
        """When _run_with_pet times out, the error string includes
        ``Retry with`` so agents see the actionable knob name."""
        monkeypatch.setattr(_server, "ROCQ_MAX_PET_RSS_MB", 1_000_000)
        _patch_psutil_rss(monkeypatch, 1)

        pet = _mock_pet()
        ls = make_lifespan_state(pet_timeout=0.05, full=True)
        ls["pet_client"] = pet
        monkeypatch.setattr(_server, "_ensure_pet", lambda lstate: pet)
        monkeypatch.setattr(
            _server,
            "_invalidate_pet",
            lambda lstate: lstate.update(pet_client=None),
        )

        def fn_slow(p):
            time.sleep(1.0)
            return {"success": True}

        result = await _run_with_pet(fn_slow, ls, "rocq_check")
        # Envelope contract:
        assert result["success"] is False
        assert isinstance(result.get("error"), str)
        assert result["reason"] == "timeout"
        assert result["pet_restarted"] is True
        # Actionable hint substrings (load-bearing pieces, not the
        # illustrative timeout value):
        assert "Retry with" in result["error"]
        assert "rocq_check(..., timeout=" in result["error"]
        assert "ROCQ_PET_TIMEOUT" in result["error"]
        assert "ROCQ_QUERY_TIMEOUT_CAP" in result["error"]


class TestCheckTimeoutConfig:
    """``_check_timeout_config`` warns on either misordering of the three
    timeout knobs, and on both at once."""

    def test_silent_when_ordered_correctly(self):
        """Pet timeout above the soft deadline and below the cap: no warning."""
        assert _server._check_timeout_config(120.0, 300, 45.0) is None

    def test_warns_when_pet_timeout_exceeds_cap(self):
        """The pre-existing check: fallback above the documented cap."""
        msg = _server._check_timeout_config(400.0, 300, 0.0)
        assert msg is not None
        assert "ROCQ_QUERY_TIMEOUT_CAP" in msg

    def test_warns_when_pet_timeout_not_above_soft_deadline(self):
        """A pet timeout below the soft deadline leaves the handoff
        unreachable, so it must warn and name the knob that fixes it."""
        msg = _server._check_timeout_config(30.0, 300, 45.0)
        assert msg is not None
        assert "ROCQ_SOFT_DEADLINE" in msg
        assert "timeout=" in msg

    def test_warns_when_equal(self):
        """Equal is still wrong: the pet timeout must be strictly above, or
        the two race with no guarantee the handoff wins."""
        assert _server._check_timeout_config(45.0, 300, 45.0) is not None

    def test_silent_when_handoff_disabled(self):
        """``ROCQ_SOFT_DEADLINE=0`` disables the handoff, so the ordering is
        irrelevant and must not warn."""
        assert _server._check_timeout_config(30.0, 300, 0.0) is None

    def test_reports_both_misorderings_at_once(self):
        """A pet timeout that is both above the cap and below the soft
        deadline names both problems rather than masking one."""
        msg = _server._check_timeout_config(40.0, 30, 45.0)
        assert msg is not None
        assert "ROCQ_QUERY_TIMEOUT_CAP" in msg
        assert "ROCQ_SOFT_DEADLINE" in msg

    def test_warns_when_coqc_timeout_not_above_soft_deadline(self):
        """rocq_compile_file is bounded by ROCQ_COQC_TIMEOUT, not the pet
        timeout, so that knob needs its own guard."""
        msg = _server._check_timeout_config(120.0, 300, 45.0, 15.0)
        assert msg is not None
        assert "ROCQ_COQC_TIMEOUT" in msg
        assert "rocq_compile_file" in msg

    def test_coqc_timeout_omitted_is_not_checked(self):
        """The argument is optional; omitting it checks only the pet side."""
        assert _server._check_timeout_config(120.0, 300, 45.0) is None

    def test_names_each_offending_knob_separately(self):
        """Both tool timeouts below the soft deadline: both are named, so the
        operator does not fix one and still have a broken handoff."""
        msg = _server._check_timeout_config(10.0, 300, 45.0, 15.0)
        assert msg is not None
        assert "ROCQ_PET_TIMEOUT" in msg
        assert "ROCQ_COQC_TIMEOUT" in msg
        assert "rocq_start" in msg
        assert "rocq_compile_file" in msg

    def test_shipped_defaults_are_ordered_correctly(self):
        """The defaults must not warn.

        This is the regression guard for the pair actually shipped: the
        handoff was introduced with a soft deadline *above* the default pet
        timeout, which left it unreachable out of the box.  Anything that
        moves either default back into that ordering fails here.
        """
        assert (
            _server._check_timeout_config(
                _server.ROCQ_PET_TIMEOUT,
                _server.ROCQ_QUERY_TIMEOUT_CAP,
                jobs.ROCQ_SOFT_DEADLINE,
                _server.ROCQ_COQC_TIMEOUT,
            )
            is None
        )
