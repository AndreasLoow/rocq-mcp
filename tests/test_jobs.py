"""Tests for the background job registry (``rocq_mcp.jobs``).

The registry exists so a cold ``rocq_start`` / ``rocq_compile_file`` is not
destroyed by the MCP client's per-call deadline.  Two properties carry that
promise and are what most of these tests pin:

- work outlives both the soft deadline and the caller's cancellation;
- an identical call attaches to the running job instead of queueing a
  second copy of the same replay behind it on the pet lock.

No pet and no Rocq: the jobs layer is deliberately ignorant of what it is
running, so the work is plain coroutines here.
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest

from rocq_mcp import compile_enrichment
from rocq_mcp import jobs
from rocq_mcp import server as _server


@pytest.fixture(autouse=True)
def _clean_registry():
    """The registry is process-global; no test may see another's jobs."""
    jobs.reset()
    yield
    jobs.reset()


async def _sleep_then(value, delay):
    await asyncio.sleep(delay)
    return value


# ---------------------------------------------------------------------------
# Soft deadline
# ---------------------------------------------------------------------------


class TestSoftDeadline:
    async def test_fast_work_returns_its_real_result(self):
        result = await jobs.run_with_soft_deadline(
            tool="rocq_start",
            key="k",
            factory=lambda: _sleep_then({"success": True, "state_id": 7}, 0),
            soft_deadline=5,
        )
        assert result == {"success": True, "state_id": 7}

    async def test_slow_work_yields_a_pending_envelope(self):
        result = await jobs.run_with_soft_deadline(
            tool="rocq_start",
            key="k",
            factory=lambda: _sleep_then({"success": True}, 10),
            soft_deadline=0.05,
        )
        assert result["success"] is False
        assert result["status"] == "pending"
        assert result["reason"] == "warming"
        assert result["job_id"]
        assert result["retry_after_s"] >= 1
        # The envelope has to talk the agent out of the fallback that
        # prompted this whole mechanism.
        assert "NOT LOST" in result["error"]
        assert "rocq_poll" in result["error"]

    async def test_pending_work_keeps_running_and_is_collectable(self):
        pending = await jobs.run_with_soft_deadline(
            tool="rocq_start",
            key="k",
            factory=lambda: _sleep_then({"success": True, "state_id": 126}, 0.2),
            soft_deadline=0.01,
        )
        assert pending["status"] == "pending"

        collected = await jobs.poll_job(pending["job_id"], wait=5)
        assert collected == {"success": True, "state_id": 126}

    async def test_zero_deadline_disables_the_handoff(self):
        """``ROCQ_SOFT_DEADLINE=0`` restores await-to-completion."""
        result = await jobs.run_with_soft_deadline(
            tool="rocq_start",
            key="k",
            factory=lambda: _sleep_then({"success": True}, 0.05),
            soft_deadline=0,
        )
        assert result == {"success": True}


# ---------------------------------------------------------------------------
# Survival
# ---------------------------------------------------------------------------


class TestWorkSurvives:
    async def test_caller_cancellation_does_not_cancel_the_work(self):
        """A client hanging up must not destroy minutes of replay.

        This is the exact shape of the bug the registry answers: the
        request dies, and the work has to still be there afterwards.
        """
        started = asyncio.Event()

        async def _work():
            started.set()
            await asyncio.sleep(0.3)
            return {"success": True, "state_id": 1}

        waiter = asyncio.create_task(
            jobs.run_with_soft_deadline(
                tool="rocq_start", key="k", factory=_work, soft_deadline=30
            )
        )
        await started.wait()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        # The job is still there, and still finishes.
        live = [j for j in jobs.snapshot() if j["status"] == "running"]
        assert len(live) == 1
        collected = await jobs.poll_job(live[0]["job_id"], wait=5)
        assert collected == {"success": True, "state_id": 1}

    async def test_background_failure_becomes_an_envelope(self):
        """A raise in detached work must not surface as an unretrieved task."""

        async def _boom():
            raise RuntimeError("pet died")

        pending = await jobs.run_with_soft_deadline(
            tool="rocq_start", key="k", factory=_boom, soft_deadline=0
        )
        assert pending["success"] is False
        assert pending["reason"] == "crashed"
        assert "pet died" in pending["error"]


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    async def test_identical_calls_share_one_job(self):
        calls = 0

        def _factory():
            nonlocal calls
            calls += 1
            return _sleep_then({"success": True}, 0.2)

        first, second = await asyncio.gather(
            jobs.run_with_soft_deadline(
                tool="rocq_start", key="same", factory=_factory, soft_deadline=5
            ),
            jobs.run_with_soft_deadline(
                tool="rocq_start", key="same", factory=_factory, soft_deadline=5
            ),
        )
        assert calls == 1
        assert first == second == {"success": True}

    async def test_a_retry_after_pending_attaches_rather_than_restarting(self):
        calls = 0

        def _factory():
            nonlocal calls
            calls += 1
            return _sleep_then({"success": True, "state_id": 5}, 0.3)

        pending = await jobs.run_with_soft_deadline(
            tool="rocq_start", key="same", factory=_factory, soft_deadline=0.01
        )
        assert pending["status"] == "pending"

        retry = await jobs.run_with_soft_deadline(
            tool="rocq_start", key="same", factory=_factory, soft_deadline=5
        )
        assert calls == 1
        assert retry == {"success": True, "state_id": 5}

    async def test_different_keys_do_not_share(self):
        calls = 0

        def _factory():
            nonlocal calls
            calls += 1
            return _sleep_then({"success": True}, 0)

        await jobs.run_with_soft_deadline(
            tool="rocq_start", key="a", factory=_factory, soft_deadline=5
        )
        await jobs.run_with_soft_deadline(
            tool="rocq_start", key="b", factory=_factory, soft_deadline=5
        )
        assert calls == 2

    async def test_a_finished_job_is_not_served_to_a_later_call(self):
        """Dedup covers concurrency, not caching.

        Attaching a *new* call to an already-finished job would hand back
        a proof state captured at an unknown earlier time, so a finished
        result is reachable only through its own ``job_id``.
        """
        calls = 0

        def _factory():
            nonlocal calls
            calls += 1
            return _sleep_then({"success": True}, 0)

        await jobs.run_with_soft_deadline(
            tool="rocq_start", key="same", factory=_factory, soft_deadline=5
        )
        await jobs.run_with_soft_deadline(
            tool="rocq_start", key="same", factory=_factory, soft_deadline=5
        )
        assert calls == 2


# ---------------------------------------------------------------------------
# Polling
# ---------------------------------------------------------------------------


class TestPoll:
    async def test_unknown_job_is_a_named_failure(self):
        result = await jobs.poll_job("job-does-not-exist")
        assert result["success"] is False
        assert result["reason"] == "not_found"
        assert result["live_jobs"] == []

    async def test_poll_without_wait_returns_pending_immediately(self):
        pending = await jobs.run_with_soft_deadline(
            tool="rocq_start",
            key="k",
            factory=lambda: _sleep_then({"success": True}, 5),
            soft_deadline=0.01,
        )
        again = await jobs.poll_job(pending["job_id"])
        assert again["status"] == "pending"
        assert again["job_id"] == pending["job_id"]

    async def test_wait_is_clamped_to_the_soft_deadline(self, monkeypatch):
        """A poll must not outlive the client deadline it exists to dodge."""
        monkeypatch.setattr(jobs, "ROCQ_SOFT_DEADLINE", 0.05)
        pending = await jobs.run_with_soft_deadline(
            tool="rocq_start",
            key="k",
            factory=lambda: _sleep_then({"success": True}, 30),
            soft_deadline=0.01,
        )
        loop = asyncio.get_running_loop()
        before = loop.time()
        result = await jobs.poll_job(pending["job_id"], wait=600)
        assert result["status"] == "pending"
        assert loop.time() - before < 5


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------


class TestBookkeeping:
    async def test_snapshot_reports_running_then_done(self):
        pending = await jobs.run_with_soft_deadline(
            tool="rocq_compile_file",
            key="k",
            factory=lambda: _sleep_then({"success": True}, 0.1),
            soft_deadline=0.01,
            detail="theories/Big.v",
        )
        snap = jobs.snapshot()
        assert snap[0]["tool"] == "rocq_compile_file"
        assert snap[0]["detail"] == "theories/Big.v"
        assert snap[0]["status"] == "running"

        await jobs.poll_job(pending["job_id"], wait=5)
        assert jobs.snapshot()[0]["status"] == "done"

    async def test_finished_jobs_expire(self, monkeypatch):
        pending = await jobs.run_with_soft_deadline(
            tool="rocq_start",
            key="k",
            factory=lambda: _sleep_then({"success": True}, 0),
            soft_deadline=5,
        )
        assert pending == {"success": True}
        monkeypatch.setattr(jobs, "ROCQ_JOB_RETENTION", -1)
        assert jobs.snapshot() == []

    async def test_running_jobs_are_never_evicted(self, monkeypatch):
        """The size cap must not orphan a task by dropping its record."""
        monkeypatch.setattr(jobs, "ROCQ_MAX_JOBS", 1)
        for i in range(4):
            await jobs.run_with_soft_deadline(
                tool="rocq_start",
                key=f"k{i}",
                factory=lambda: _sleep_then({"success": True}, 30),
                soft_deadline=0.01,
            )
        assert len([j for j in jobs.snapshot() if j["status"] == "running"]) == 4


# ---------------------------------------------------------------------------
# Key construction (server side)
# ---------------------------------------------------------------------------


class TestJobKey:
    def test_same_call_same_key(self, tmp_path):
        f = tmp_path / "A.v"
        f.write_text("Theorem t : True. Proof. exact I. Qed.\n")
        a = _server._job_key("rocq_start", str(tmp_path), file="A.v", line=1)
        b = _server._job_key("rocq_start", str(tmp_path), file="A.v", line=1)
        assert a == b

    def test_editing_the_file_changes_the_key(self, tmp_path):
        """An edit must never be answered by the pre-edit job."""
        f = tmp_path / "A.v"
        f.write_text("Theorem t : True.\n")
        before = _server._job_key("rocq_start", str(tmp_path), file="A.v", line=1)
        f.write_text("Theorem t : True. Proof. exact I. Qed.\n")
        os.utime(f, (0, 0))
        after = _server._job_key("rocq_start", str(tmp_path), file="A.v", line=1)
        assert before != after

    def test_different_positions_are_different_work(self, tmp_path):
        f = tmp_path / "A.v"
        f.write_text("Theorem t : True.\n")
        a = _server._job_key("rocq_start", str(tmp_path), file="A.v", line=1)
        b = _server._job_key("rocq_start", str(tmp_path), file="A.v", line=2)
        assert a != b

    def test_a_missing_file_still_yields_a_key(self, tmp_path):
        key = _server._job_key("rocq_start", str(tmp_path), file="nope.v")
        assert key.endswith("?")


class TestCompileDoesNotBlockTheEventLoop:
    """The compile wrappers must run coqc/dune off the event loop.

    ``run_compile_file`` blocks in ``subprocess.communicate`` until the
    compile returns.  Called directly from its ``async def`` wrapper it
    froze the whole loop for that long, and the symptom that matters here
    is that ``jobs``' soft-deadline timer could not fire: the ``pending``
    envelope arrived only once the compile had *already finished* (observed
    in the field at 158s against ``ROCQ_SOFT_DEADLINE=20``), far too late
    to beat the client's own per-call deadline.
    """

    @staticmethod
    def _blocking_compile(*args, **kwargs):
        time.sleep(0.6)
        return {"success": True, "stdout": "", "stderr": ""}

    @pytest.mark.asyncio
    async def test_loop_stays_responsive_during_a_compile(self, monkeypatch, tmp_path):
        """Another coroutine must get scheduled while the compile runs."""
        monkeypatch.setattr(
            compile_enrichment, "run_compile_file", self._blocking_compile
        )
        (tmp_path / "a.v").write_text("(* x *)\n")

        ticks = 0

        async def _ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        tick_task = asyncio.create_task(_ticker())
        try:
            await compile_enrichment.run_compile_file_with_state(
                file="a.v", workspace=str(tmp_path), timeout=30
            )
        finally:
            tick_task.cancel()

        # A blocked loop yields zero ticks; a free one yields many.
        assert ticks > 5, f"event loop was blocked during the compile ({ticks} ticks)"

    @pytest.mark.asyncio
    async def test_soft_deadline_fires_while_a_compile_is_still_running(
        self, monkeypatch, tmp_path
    ):
        """The regression: pending must arrive at the deadline, not at the end.

        With the compile on the loop this returned the finished result after
        the full 0.6s instead of a pending envelope at 0.1s.
        """
        monkeypatch.setattr(
            compile_enrichment, "run_compile_file", self._blocking_compile
        )
        (tmp_path / "a.v").write_text("(* x *)\n")

        started = time.monotonic()
        result = await jobs.run_with_soft_deadline(
            tool="rocq_compile_file",
            key="k-compile-blocking",
            soft_deadline=0.1,
            factory=lambda: compile_enrichment.run_compile_file_with_state(
                file="a.v", workspace=str(tmp_path), timeout=30
            ),
        )
        elapsed = time.monotonic() - started

        assert result["status"] == "pending"
        assert result["reason"] == "warming"
        assert elapsed < 0.5, f"pending envelope took {elapsed:.2f}s to arrive"

        # And the work really did keep running: it lands via the job.
        collected = await jobs.poll_job(result["job_id"], wait=2.0)
        assert collected["success"] is True
