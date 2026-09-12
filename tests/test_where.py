"""Tests for position resolution: ``where=`` and the ``resolved`` report.

Petanque resolves a cursor by rounding *forward* through the sentence it
lies in, so the state before a sentence is addressed by pointing at the
whitespace ahead of it.  Getting that column wrong reads back as
``goals: ""`` with ``proof_finished: true``, which is indistinguishable
from a broken session unless the response says where it looked.

``where="before"`` computes the column, and ``resolved`` reports the
position actually used whenever the goals come back empty.  Both are
pure source-text operations, so none of this needs pet.
"""

from __future__ import annotations

import collections

import pytest

from rocq_mcp.interactive import (
    _position_for_where,
    _resolved_report,
    run_start,
)


@pytest.fixture
def proof_file(tmp_path):
    """A proof whose tactics are indented and whose keywords are not."""
    path = tmp_path / "P.v"
    path.write_text(
        "Theorem t : True /\\ True.\n"  # line 0
        "Proof.\n"  # line 1
        "  split.\n"  # line 2
        "  - exact I.\n"  # line 3
        "  - exact I.\n"  # line 4
        "Qed.\n"  # line 5
    )
    return str(path)


# ---------------------------------------------------------------------------
# where="before"
# ---------------------------------------------------------------------------


class TestPositionForWhere:
    def test_after_is_left_alone(self, proof_file):
        assert _position_for_where(proof_file, 2, 4, "after") == (2, 4)

    def test_indented_line_uses_column_zero(self, proof_file):
        """Column 0 of an indented line is already whitespace-before."""
        assert _position_for_where(proof_file, 2, 7, "before") == (2, 0)

    def test_character_is_ignored_for_before(self, proof_file):
        """Whichever column the caller guessed, "before" is the same place."""
        assert _position_for_where(proof_file, 3, 0, "before") == _position_for_where(
            proof_file, 3, 11, "before"
        )

    def test_unindented_line_uses_the_end_of_the_previous_line(self, proof_file):
        """Whitespace after the previous period is the same state.

        ``Qed.`` starts at column 0, so there is no whitespace on its own
        line to point at; the end of line 4 is after that sentence's
        period, which is the state before ``Qed.``.
        """
        assert _position_for_where(proof_file, 5, 0, "before") == (
            4,
            len("  - exact I."),
        )

    def test_first_line_has_no_before(self, proof_file):
        assert _position_for_where(proof_file, 0, 3, "before") == (0, 3)

    def test_position_past_the_end_is_left_alone(self, proof_file):
        assert _position_for_where(proof_file, 99, 0, "before") == (99, 0)

    def test_unreadable_file_degrades_to_the_literal_position(self, tmp_path):
        """Petanque may still resolve a file this helper cannot read."""
        missing = str(tmp_path / "nope.v")
        assert _position_for_where(missing, 4, 2, "before") == (4, 2)


# ---------------------------------------------------------------------------
# resolved
# ---------------------------------------------------------------------------


class TestResolvedReport:
    def test_names_the_line_it_read(self, proof_file):
        report = _resolved_report(
            proof_file, line=3, character=4, where="after", requested=(3, 4)
        )
        assert report["line"] == 3
        assert report["character"] == 4
        assert report["where"] == "after"
        assert report["line_text"] == "  - exact I."
        assert "requested" not in report

    def test_records_the_shift_when_where_moved_the_cursor(self, proof_file):
        report = _resolved_report(
            proof_file, line=2, character=0, where="before", requested=(2, 7)
        )
        assert report["requested"] == {"line": 2, "character": 7}

    def test_after_points_at_the_remedy(self, proof_file):
        report = _resolved_report(
            proof_file, line=5, character=0, where="after", requested=(5, 0)
        )
        assert "where='before'" in report["note"]

    def test_before_does_not_recommend_itself(self, proof_file):
        report = _resolved_report(
            proof_file, line=5, character=0, where="before", requested=(5, 0)
        )
        assert "where='after'" in report["note"]

    def test_line_text_is_capped(self, tmp_path):
        path = tmp_path / "Long.v"
        path.write_text("(* " + "x" * 5000 + " *)\n")
        report = _resolved_report(
            str(path), line=0, character=0, where="after", requested=(0, 0)
        )
        assert len(report["line_text"]) == 200

    def test_missing_line_text_is_omitted_not_faked(self, tmp_path):
        report = _resolved_report(
            str(tmp_path / "nope.v"),
            line=0,
            character=0,
            where="after",
            requested=(0, 0),
        )
        assert "line_text" not in report
        assert report["where"] == "after"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestWhereValidation:
    async def test_unknown_where_is_rejected_before_pet_is_touched(self, proof_file):
        state = {"recent_errors": collections.deque(maxlen=10)}
        result = await run_start(
            file=proof_file,
            theorem="",
            workspace="/tmp",
            lifespan_state=state,
            line=2,
            character=0,
            where="sideways",
        )
        assert result["success"] is False
        assert "where must be one of" in result["error"]


# ---------------------------------------------------------------------------
# Real pet
# ---------------------------------------------------------------------------


class TestWhereAgainstRealPet:
    """The whole point is which goal comes back, so pin that against pet."""

    @pytest.mark.skipif(
        not __import__("shutil").which("pet"), reason="pet not available"
    )
    async def test_before_and_after_straddle_the_tactic(self, tmp_path):
        from tests.conftest import make_lifespan_state

        (tmp_path / "_CoqProject").write_text("")
        (tmp_path / "P.v").write_text(
            "Theorem t : forall n : nat, n + 0 = n.\n"
            "Proof.\n"
            "  intros n.\n"
            "  induction n as [| k IH].\n"
            "  - reflexivity.\n"
            "  - simpl. rewrite IH. reflexivity.\n"
            "Qed.\n"
        )
        state = make_lifespan_state(pet_timeout=120.0, full=True)
        state["workspace"] = str(tmp_path)

        common = dict(
            file="P.v",
            theorem="",
            workspace=str(tmp_path),
            lifespan_state=state,
            line=2,  # "  intros n."
            character=4,
            timeout=120.0,
        )
        after = await run_start(**common, where="after")
        before = await run_start(**common, where="before")

        assert after["success"] and before["success"]
        # After the intro the binder is a hypothesis; before it, it is
        # still under the quantifier.
        assert "n : nat" in after["goals"]
        assert "forall n" in before["goals"]
        assert "forall n" not in after["goals"]
