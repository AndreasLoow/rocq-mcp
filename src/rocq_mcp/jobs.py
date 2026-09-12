"""Background jobs — keep a cold call alive past the client's deadline.

The first ``rocq_start`` / ``rocq_compile_file`` against a file has to
replay that file's prefix, which on a large development takes minutes.
MCP clients impose their own per-call deadline (typically 60s) *above*
the server, so such a call is killed by the client no matter what
``timeout=`` the caller passed or what ``ROCQ_PET_TIMEOUT`` is set to.

The work itself is never lost when that happens — the request dies but
the thread holding the pet lock runs to completion and Fleche keeps the
document warm — so the caller's *second* identical call returns
promptly.  Relying on that is a trap all the same: the agent sees a
tool timeout, has no way to tell "still warming" from "broken", and
usually falls back to a whole-file build.

This module turns that accident into a contract.  Work is started as a
detached :class:`asyncio.Task` registered here, and the caller awaits it
only up to a *soft deadline* set below the client's.  If it finishes in
time the caller gets the real result and the job retires immediately.
If it does not, the caller gets a ``status: "pending"`` envelope naming
a ``job_id``; the task keeps running, and either ``rocq_poll(job_id)``
or an identical re-call picks up where it left off.

Two properties matter and are what the tests here pin down:

- **A cancelled request does not cancel the work.**  Callers await
  through :func:`asyncio.shield`, so both the soft deadline firing and
  the client hanging up leave the underlying task running.
- **An identical call attaches rather than queueing.**  Concurrent calls
  sharing a *key* share one job, so an agent that retries a timed-out
  call does not end up behind its own first call on the pet lock.

Keys are built by the caller (see ``server._job_key``) and include the
file's mtime, so editing the file yields a different key: an edit is
never served the pre-edit job's result.
"""

from __future__ import annotations

import asyncio
import itertools
import os
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

# Seconds a caller waits before being handed a ``pending`` envelope.
# Two ceilings bound this, and the default sits below both:
#   - the MCP client's own per-call deadline (60s in the clients
#     measured), with enough headroom for the response to travel;
#   - the effective pet timeout (a per-call ``timeout=``, else
#     ``server.ROCQ_PET_TIMEOUT``, default 30), which is what actually
#     kills the work.  Above that, the work dies before the handoff can
#     happen and this mechanism is unreachable -- ``server``'s
#     ``_check_timeout_config`` warns at startup when the pair is
#     ordered that way.
# 0 disables the mechanism: callers then await the work to completion
# and behave exactly as they did before this module existed.
ROCQ_SOFT_DEADLINE: float = float(os.environ.get("ROCQ_SOFT_DEADLINE", "20"))

# Seconds a finished job's result stays retrievable by ``rocq_poll``.
ROCQ_JOB_RETENTION: float = float(os.environ.get("ROCQ_JOB_RETENTION", "900"))

# Hard cap on tracked jobs.  Finished jobs are evicted oldest-first;
# running jobs are never evicted (their task would outlive its record).
ROCQ_MAX_JOBS: int = int(os.environ.get("ROCQ_MAX_JOBS", "64"))

_job_counter = itertools.count(1)


class _Pending:
    """Sentinel: the job outlived the wait, nothing to return yet."""


_PENDING = _Pending()


@dataclass
class _Job:
    """One unit of deferred work and everything ``rocq_poll`` reports."""

    job_id: str
    tool: str
    key: str
    started_at: float
    task: asyncio.Task
    detail: str = ""
    finished_at: float | None = None
    waiters: int = 0
    # Set once a caller has been handed the result, so the job stops
    # being offered to new identical calls (see _live_job).
    delivered: bool = False


# job_id -> _Job, insertion-ordered so eviction is oldest-first.
_jobs: dict[str, _Job] = {}


def _now() -> float:
    return time.monotonic()


def reset() -> None:
    """Drop every tracked job.  For tests and ``force_restart``."""
    for job in list(_jobs.values()):
        if not job.task.done():
            job.task.cancel()
    _jobs.clear()


def _reap() -> None:
    """Evict finished jobs past retention, then over the size cap."""
    now = _now()
    for job_id, job in list(_jobs.items()):
        if job.finished_at is not None and now - job.finished_at > ROCQ_JOB_RETENTION:
            del _jobs[job_id]
    if len(_jobs) <= ROCQ_MAX_JOBS:
        return
    for job_id, job in list(_jobs.items()):
        if len(_jobs) <= ROCQ_MAX_JOBS:
            return
        if job.task.done():
            del _jobs[job_id]


def _live_job(key: str) -> _Job | None:
    """Return a still-running job under *key*, if one exists.

    Only *running* jobs are offered: a finished result is reachable
    solely through its ``job_id``.  A later identical call re-runs
    instead of being served a result computed at an unknown earlier
    time, which keeps this dedup from becoming an unbounded cache.
    """
    for job in _jobs.values():
        if job.key == key and not job.task.done():
            return job
    return None


async def _guard(tool: str, coro: Awaitable[Any]) -> Any:
    """Run *coro*, turning any escape into a failure envelope.

    A job outlives the request that started it, so there is frequently
    nobody awaiting the task when it finishes.  An exception escaping
    here would surface only as asyncio's "exception was never retrieved"
    noise on the server's stderr and leave a later ``rocq_poll`` raising
    instead of answering, so failures are converted to the same envelope
    shape every other failure path returns.
    """
    try:
        return await coro
    except asyncio.CancelledError:
        raise
    except BaseException as exc:  # noqa: BLE001 - deliberately total
        return {
            "success": False,
            "reason": "crashed",
            "error": f"{tool} failed in the background: {exc!r}",
        }


def _register(tool: str, key: str, detail: str, coro: Awaitable[Any]) -> _Job:
    job_id = f"job-{next(_job_counter)}"
    task = asyncio.ensure_future(_guard(tool, coro))
    job = _Job(
        job_id=job_id,
        tool=tool,
        key=key,
        started_at=_now(),
        task=task,
        detail=detail,
    )

    def _mark_done(_task: asyncio.Task) -> None:
        job.finished_at = _now()

    task.add_done_callback(_mark_done)
    _jobs[job_id] = job
    _reap()
    return job


def _retry_after(job: _Job) -> int:
    """How long to suggest waiting before polling again.

    Scaled off elapsed time: a job that has already run for minutes is
    unlikely to land in the next few seconds, and polling it every 5s
    just burns turns.
    """
    elapsed = _now() - job.started_at
    return int(min(60.0, max(10.0, elapsed * 0.4)))


def pending_envelope(job: _Job) -> dict[str, Any]:
    """The response handed back when a job outlives the soft deadline."""
    elapsed = round(_now() - job.started_at, 1)
    return {
        "success": False,
        "status": "pending",
        "reason": "warming",
        "job_id": job.job_id,
        "tool": job.tool,
        "elapsed_s": elapsed,
        "retry_after_s": _retry_after(job),
        "error": (
            f"{job.tool} is still processing after {elapsed}s and has been "
            f"handed off to background job {job.job_id}. THE WORK IS NOT "
            f"LOST and is still running — do not fall back to a whole-file "
            f"build. Call rocq_poll(job_id='{job.job_id}') to collect the "
            f"result; an identical {job.tool} call also re-attaches to this "
            f"same job rather than starting a second one."
        ),
    }


async def run_with_soft_deadline(
    *,
    tool: str,
    key: str,
    factory: Callable[[], Awaitable[Any]],
    soft_deadline: float | None = None,
    detail: str = "",
) -> Any:
    """Await *factory()* for at most the soft deadline.

    Returns the real result if the work lands in time, otherwise a
    :func:`pending_envelope`.  An identical *key* that is still running
    is attached to rather than started again.

    *factory* is only called when no live job matches, so the caller can
    build the coroutine unconditionally without risking an un-awaited
    coroutine warning on the attach path.
    """
    deadline = ROCQ_SOFT_DEADLINE if soft_deadline is None else soft_deadline

    job = _live_job(key)
    if job is None:
        job = _register(tool, key, detail, factory())

    if deadline <= 0:
        # Mechanism disabled: await to completion, but still shield so a
        # client hang-up does not destroy the work.
        return await _await_job(job, timeout=None)

    result = await _await_job(job, timeout=deadline)
    if result is _PENDING:
        return pending_envelope(job)
    return result


async def _await_job(job: _Job, *, timeout: float | None) -> Any:
    """Wait on *job* without ever cancelling it.

    ``asyncio.shield`` keeps the underlying task alive both when the
    wait times out and when the caller itself is cancelled — the latter
    being the case that matters, since that is what an MCP client
    hanging up looks like from in here.
    """
    job.waiters += 1
    try:
        if timeout is None:
            result = await asyncio.shield(job.task)
        else:
            try:
                result = await asyncio.wait_for(
                    asyncio.shield(job.task), timeout=timeout
                )
            except asyncio.TimeoutError:
                return _PENDING
    finally:
        job.waiters -= 1
    job.delivered = True
    return result


async def poll_job(job_id: str, wait: float = 0.0) -> dict[str, Any]:
    """Collect a job's result, optionally waiting up to *wait* seconds.

    Returns the finished result, a fresh ``pending`` envelope, or a
    ``not_found`` failure naming the jobs that do exist.
    """
    _reap()
    job = _jobs.get(job_id)
    if job is None:
        live = [j.job_id for j in _jobs.values() if not j.task.done()]
        return {
            "success": False,
            "reason": "not_found",
            "error": (
                f"No job {job_id!r}. It either never existed or its result "
                f"expired (results are kept {int(ROCQ_JOB_RETENTION)}s after "
                f"completion). Re-issue the original call; if the document is "
                f"warm by now it will return promptly."
            ),
            "live_jobs": live,
        }

    wait = max(0.0, min(float(wait), ROCQ_SOFT_DEADLINE or float(wait)))
    result = await _await_job(job, timeout=wait if wait > 0 else 0.0)
    if result is _PENDING:
        return pending_envelope(job)
    return result


def snapshot() -> list[dict[str, Any]]:
    """Job list for ``rocq_diag``.  Read-only, never spawns anything."""
    _reap()
    now = _now()
    out: list[dict[str, Any]] = []
    for job in _jobs.values():
        done = job.task.done()
        out.append(
            {
                "job_id": job.job_id,
                "tool": job.tool,
                "detail": job.detail,
                "status": "done" if done else "running",
                "age_seconds": round(now - job.started_at, 1),
                "waiters": job.waiters,
                "delivered": job.delivered,
            }
        )
    return out
