"""Tests for preamble / import failures surfacing instead of being swallowed.

``get_state_at_pos`` returns a state even when coq-lsp rejected a command in
the document, and the petanque protocol carries no diagnostics — so a
``Require`` against a stale ``.vo`` used to produce a *successful* start over
an environment missing every requested import.  The production code re-issues
the library-loading sentences on the state they produced and reports what Coq
raises; these tests pin that behaviour, its failure envelope, and the cases
where the replay must stay quiet.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import rocq_mcp.interactive as _interactive
import rocq_mcp.server as _server
from tests.conftest import make_lifespan_state

# The message Coq emits for the "recompiled A, did not recompile B" state.
INCONSISTENT = (
    "Compiled library Repro.B (in file /ws/B.vo) makes inconsistent "
    "assumptions over library Repro.A"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def petanque_error(monkeypatch):
    """Install a ``pytanque`` stand-in whose ``PetanqueError`` we can raise.

    Installed unconditionally (over a real pytanque, if present) so the
    error's constructor is the same shape on every machine.  ``interactive``
    binds ``_PetanqueError`` at import time, so that has to be repointed too.
    """
    module = types.ModuleType("pytanque")

    class _PetanqueError(Exception):
        def __init__(self, message: str = "") -> None:
            self.message = message
            super().__init__(message)

    module.PetanqueError = _PetanqueError
    monkeypatch.setitem(sys.modules, "pytanque", module)
    monkeypatch.setattr(_interactive, "_PetanqueError", _PetanqueError)
    return _PetanqueError


@pytest.fixture(autouse=True)
def clean_caches():
    """Drop the import cache and the pet semaphore around every test."""
    _interactive._import_cache.clear()
    _server._pet_semaphore = None
    yield
    _interactive._import_cache.clear()
    _server._pet_semaphore = None


@pytest.fixture
def lifespan_state():
    state = make_lifespan_state(full=True)
    state["current_workspace"] = None
    return state


class _FakePet:
    """Minimal pytanque stand-in that fails exactly the commands asked for.

    *failures* maps a sentence to the error message ``run`` should raise for
    it; every other sentence is accepted and recorded in ``commands``.
    """

    def __init__(self, failures: dict[str, str] | None = None, alive: bool = True):
        self.failures = failures or {}
        self.commands: list[str] = []
        self.states_built = 0
        self.start_error: str | None = None
        self.toc_names: list[str] = []
        self.process = MagicMock()
        self.process.poll.return_value = None if alive else 1
        self._own_pgrp = False
        self.set_workspace = MagicMock()

    # -- construction ----------------------------------------------------
    def _state(self):
        return SimpleNamespace(st=self.states_built, proof_finished=False, feedback=[])

    def get_state_at_pos(self, file, line, character):
        self.states_built += 1
        return self._state()

    def get_root_state(self, file):
        return self._state()

    def start(self, file, theorem):
        if self.start_error is not None:
            raise sys.modules["pytanque"].PetanqueError(self.start_error)
        return self._state()

    # -- execution -------------------------------------------------------
    def run(self, state, cmd, timeout=None):
        self.commands.append(cmd)
        message = self.failures.get(cmd.strip())
        if message is not None:
            raise sys.modules["pytanque"].PetanqueError(message)
        return self._state()

    def complete_goals(self, state):
        return SimpleNamespace(goals=[], stack=[], shelf=[], given_up=[])

    def toc(self, file):
        return [
            (
                "",
                [
                    SimpleNamespace(
                        name=SimpleNamespace(v=n),
                        detail="Lemma",
                        range=None,
                        children=[],
                    )
                    for n in self.toc_names
                ],
            )
        ]


async def _run_start(pet, **kwargs):
    """Drive ``run_start`` against *pet* with ``_ensure_pet`` patched out."""
    with patch.object(_server, "_ensure_pet", return_value=pet):
        return await _interactive.run_start(**kwargs)


async def _run_query(pet, **kwargs):
    with patch.object(_server, "_ensure_pet", return_value=pet):
        return await _interactive.run_query(**kwargs)


# ---------------------------------------------------------------------------
# _get_or_create_import_state: validation of a freshly built import state
# ---------------------------------------------------------------------------


class TestImportStateValidation:
    def test_rejected_require_raises(self, tmp_path, lifespan_state):
        """A Require Coq rejects must not come back as a usable state."""
        pet = _FakePet(failures={"Require Import Repro.B.": INCONSISTENT})
        lifespan_state["pet_client"] = pet
        with pytest.raises(_interactive._PreambleError) as excinfo:
            _interactive._get_or_create_import_state(
                pet,
                str(tmp_path),
                ["Require Import Repro.A.", "Require Import Repro.B."],
                lifespan_state,
            )
        failure = excinfo.value.failure
        assert failure.command == "Require Import Repro.B."
        assert failure.message == INCONSISTENT

    def test_rejected_preamble_is_not_cached(self, tmp_path, lifespan_state):
        """The broken state must not be served to the next caller: the
        remedy is a rebuild, and a cached failure would outlive it."""
        pet = _FakePet(failures={"Require Import Repro.B.": INCONSISTENT})
        lifespan_state["pet_client"] = pet
        cmds = ["Require Import Repro.B."]
        for _ in range(2):
            with pytest.raises(_interactive._PreambleError):
                _interactive._get_or_create_import_state(
                    pet, str(tmp_path), cmds, lifespan_state
                )
        assert _interactive._import_cache == {}
        assert pet.states_built == 2

    def test_healthy_preamble_is_replayed_then_cached(self, tmp_path, lifespan_state):
        """A preamble that loads is replayed once and cached; the second
        call touches neither ``get_state_at_pos`` nor ``run``."""
        pet = _FakePet()
        cmds = ["From Coq Require Import Lia.", "Open Scope Z_scope."]
        first = _interactive._get_or_create_import_state(
            pet, str(tmp_path), cmds, lifespan_state
        )
        assert pet.commands == cmds
        second = _interactive._get_or_create_import_state(
            pet, str(tmp_path), cmds, lifespan_state
        )
        assert second is first
        assert pet.commands == cmds
        assert pet.states_built == 1

    def test_replay_stops_at_the_first_non_import(self, tmp_path, lifespan_state):
        """A ``Definition`` would fail on replay ("already exists") for a
        reason that has nothing to do with the imports, so the replay ends
        there and can never produce a false failure."""
        pet = _FakePet(failures={"Definition x := 3.": "x already exists."})
        lifespan_state["pet_client"] = pet
        state = _interactive._get_or_create_import_state(
            pet,
            str(tmp_path),
            ["Require Import Lia.", "Definition x := 3."],
            lifespan_state,
        )
        assert state is not None
        assert pet.commands == ["Require Import Lia."]

    def test_replay_stops_before_a_section(self, tmp_path, lifespan_state):
        """Everything after ``Section S.`` runs *inside* that section, and
        Rocq rejects a ``Require`` there.  Replaying it at top level would
        report a failure the original run never had."""
        pet = _FakePet(
            failures={"Require Import B.": "Require is not allowed inside a section."}
        )
        lifespan_state["pet_client"] = pet
        state = _interactive._get_or_create_import_state(
            pet,
            str(tmp_path),
            ["Require Import A.", "Section S.", "Require Import B."],
            lifespan_state,
        )
        assert state is not None
        assert pet.commands == ["Require Import A."]

    def test_leading_comment_does_not_end_the_prefix(self, tmp_path, lifespan_state):
        """Sentence splitting glues a file's header comment onto the first
        vernacular; the prefix must look past it."""
        pet = _FakePet(failures={"(* header *)\nRequire Import Repro.B.": INCONSISTENT})
        lifespan_state["pet_client"] = pet
        with pytest.raises(_interactive._PreambleError):
            _interactive._get_or_create_import_state(
                pet,
                str(tmp_path),
                ["(* header *)\nRequire Import Repro.B."],
                lifespan_state,
            )

    def test_replay_skipped_when_pet_has_no_root_state(self, tmp_path, lifespan_state):
        """Validation is best-effort: an older petanque without the
        ``get_root_state`` route degrades to the old behaviour rather than
        reporting a failure nobody observed."""
        pet = _FakePet(failures={"Require Import Repro.B.": INCONSISTENT})
        lifespan_state["pet_client"] = pet

        def _no_route(file):
            raise sys.modules["pytanque"].PetanqueError("unknown route")

        pet.get_root_state = _no_route
        state = _interactive._get_or_create_import_state(
            pet, str(tmp_path), ["Require Import Repro.B."], lifespan_state
        )
        assert state is not None
        assert pet.commands == []

    def test_vo_rebuild_retires_the_cached_state(self, tmp_path, lifespan_state):
        """A compile through this server that rewrote a .vo advances the
        workspace epoch; the cached state froze the libraries loaded before
        it, so it must be rebuilt rather than handed out."""
        pet = _FakePet()
        cmds = ["Require Import Repro.B."]
        _interactive._get_or_create_import_state(
            pet, str(tmp_path), cmds, lifespan_state
        )
        assert pet.states_built == 1

        _server._bump_vo_epoch_if_rebuilt(
            lifespan_state, str(tmp_path), {"B.vo": 1.0}, {"B.vo": 2.0}
        )
        _interactive._get_or_create_import_state(
            pet, str(tmp_path), cmds, lifespan_state
        )
        assert pet.states_built == 2

    def test_dead_pet_during_replay_propagates(self, tmp_path, lifespan_state):
        """A pet that died mid-replay is a transport failure, not a
        preamble failure — the PetanqueError has to reach _run_with_pet."""
        pet = _FakePet(failures={"Require Import Lia.": "boom"}, alive=False)
        lifespan_state["pet_client"] = pet
        with pytest.raises(sys.modules["pytanque"].PetanqueError):
            _interactive._get_or_create_import_state(
                pet, str(tmp_path), ["Require Import Lia."], lifespan_state
            )


# ---------------------------------------------------------------------------
# rocq_start, preamble mode
# ---------------------------------------------------------------------------


class TestStartPreambleMode:
    @pytest.mark.asyncio
    async def test_stale_vo_reported_instead_of_empty_success(
        self, tmp_path, lifespan_state
    ):
        pet = _FakePet(failures={"Require Import Repro.B.": INCONSISTENT})
        lifespan_state["pet_client"] = pet
        result = await _run_start(
            pet,
            file="",
            theorem="",
            workspace=str(tmp_path),
            lifespan_state=lifespan_state,
            preamble="Require Import Repro.A.\nRequire Import Repro.B.",
        )
        assert result["success"] is False
        assert result["reason"] == "preamble_failed"
        assert result["error"] == INCONSISTENT
        assert result["failed_command"] == "Require Import Repro.B."
        assert "state_id" not in result
        # The remedy is the whole point of reporting the error at all.
        assert "force_restart" in result["hint"]
        recorded = list(lifespan_state["recent_errors"])
        assert recorded[-1]["tool"] == "rocq_start"
        assert recorded[-1]["reason"] == "preamble_failed"

    @pytest.mark.asyncio
    async def test_non_library_failure_carries_no_stale_vo_hint(
        self, tmp_path, lifespan_state
    ):
        """The rebuild / force_restart advice is wrong for, say, a scope
        typo — report the error without it."""
        pet = _FakePet(failures={"Open Scope Nonsuch_scope.": "Scope Nonsuch_scope"})
        lifespan_state["pet_client"] = pet
        result = await _run_start(
            pet,
            file="",
            theorem="",
            workspace=str(tmp_path),
            lifespan_state=lifespan_state,
            preamble="Open Scope Nonsuch_scope.",
        )
        assert result["reason"] == "preamble_failed"
        assert "hint" not in result

    @pytest.mark.asyncio
    async def test_healthy_preamble_still_starts(self, tmp_path, lifespan_state):
        pet = _FakePet()
        lifespan_state["pet_client"] = pet
        result = await _run_start(
            pet,
            file="",
            theorem="",
            workspace=str(tmp_path),
            lifespan_state=lifespan_state,
            preamble="Require Import Lia.",
        )
        assert result["success"] is True
        assert result["file"] == "<preamble>"
        assert isinstance(result["state_id"], int)


# ---------------------------------------------------------------------------
# rocq_query, preamble mode
# ---------------------------------------------------------------------------


class TestQueryPreambleMode:
    @pytest.mark.asyncio
    async def test_query_reports_the_preamble_failure(self, tmp_path, lifespan_state):
        """A query against a half-built environment answers the wrong
        question; report the import failure instead of running it."""
        pet = _FakePet(failures={"Require Import Repro.B.": INCONSISTENT})
        lifespan_state["pet_client"] = pet
        result = await _run_query(
            pet,
            command="Search (_ + _).",
            preamble="Require Import Repro.B.",
            workspace=str(tmp_path),
            lifespan_state=lifespan_state,
        )
        assert result["success"] is False
        assert result["reason"] == "preamble_failed"
        assert result["failed_command"] == "Require Import Repro.B."
        assert "Search (_ + _)." not in pet.commands


# ---------------------------------------------------------------------------
# rocq_start, theorem mode: root-cause enrichment
# ---------------------------------------------------------------------------

_NOT_FOUND = (
    "Theorem_not_found: [find_thm] Theorem found but failed with Coq error:\n"
    " The reference b was not found in the current environment.!"
)


def _theorem_file(tmp_path):
    vfile = tmp_path / "C.v"
    vfile.write_text(
        "Require Import Repro.A Repro.B.\n\n"
        "Lemma repro : b = 0.\n"
        "Proof.\n"
        "  reflexivity.\n"
        "Qed.\n"
    )
    return vfile


class TestTheoremModeRootCause:
    @pytest.mark.asyncio
    async def test_library_error_replaces_the_downstream_symptom(
        self, tmp_path, lifespan_state
    ):
        """ "The reference b was not found" names neither library nor
        remedy; the file's own Require does both."""
        _theorem_file(tmp_path)
        pet = _FakePet(failures={"Require Import Repro.A Repro.B.": INCONSISTENT})
        pet.start_error = _NOT_FOUND
        pet.toc_names = ["repro"]
        lifespan_state["pet_client"] = pet
        result = await _run_start(
            pet,
            file="C.v",
            theorem="repro",
            workspace=str(tmp_path),
            lifespan_state=lifespan_state,
        )
        assert result["success"] is False
        assert result["reason"] == "preamble_failed"
        assert result["error"] == INCONSISTENT
        assert result["failed_command"] == "Require Import Repro.A Repro.B."
        # The error pet actually reported is kept, not discarded.
        assert result["lookup_error"] == _NOT_FOUND
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_typo_does_not_pay_for_a_library_probe(
        self, tmp_path, lifespan_state
    ):
        """A name that is not in the file is a typo, not a broken import:
        replaying the file's imports to explain it would be a wasted library
        load on the commonest error on this path."""
        _theorem_file(tmp_path)
        pet = _FakePet()
        pet.start_error = "Theorem_not_found: reprp"
        pet.toc_names = ["repro"]
        lifespan_state["pet_client"] = pet
        await _run_start(
            pet,
            file="C.v",
            theorem="reprp",
            workspace=str(tmp_path),
            lifespan_state=lifespan_state,
        )
        assert pet.commands == []

    @pytest.mark.asyncio
    async def test_typo_still_reports_not_found(self, tmp_path, lifespan_state):
        """When the file's imports are fine, a bad name is still a bad
        name — the enrichment must not change that envelope."""
        _theorem_file(tmp_path)
        pet = _FakePet()
        pet.start_error = "Theorem_not_found: reprp"
        pet.toc_names = ["repro"]
        lifespan_state["pet_client"] = pet
        result = await _run_start(
            pet,
            file="C.v",
            theorem="reprp",
            workspace=str(tmp_path),
            lifespan_state=lifespan_state,
        )
        assert result["success"] is False
        assert result["reason"] == "not_found"
        assert result["available_in_file"] == ["repro"]

    @pytest.mark.asyncio
    async def test_unrelated_replay_error_does_not_mask_the_real_one(
        self, tmp_path, lifespan_state
    ):
        """Only a library-level replay error is trusted to override what
        pet reported; anything else is likelier an artefact of the replay."""
        _theorem_file(tmp_path)
        pet = _FakePet(
            failures={"Require Import Repro.A Repro.B.": "Some unrelated error."}
        )
        pet.start_error = _NOT_FOUND
        pet.toc_names = ["repro"]
        lifespan_state["pet_client"] = pet
        result = await _run_start(
            pet,
            file="C.v",
            theorem="repro",
            workspace=str(tmp_path),
            lifespan_state=lifespan_state,
        )
        assert result["reason"] == "not_found"
        assert result["error"] == _NOT_FOUND

    @pytest.mark.asyncio
    async def test_missing_library_is_also_a_library_error(
        self, tmp_path, lifespan_state
    ):
        _theorem_file(tmp_path)
        missing = "Cannot find library Repro.B in loadpath"
        pet = _FakePet(failures={"Require Import Repro.A Repro.B.": missing})
        pet.start_error = _NOT_FOUND
        lifespan_state["pet_client"] = pet
        result = await _run_start(
            pet,
            file="C.v",
            theorem="repro",
            workspace=str(tmp_path),
            lifespan_state=lifespan_state,
        )
        assert result["reason"] == "preamble_failed"
        assert result["error"] == missing
