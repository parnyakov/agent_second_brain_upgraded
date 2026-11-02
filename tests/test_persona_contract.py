"""The persona file is the agent's identity — verify its contract shape."""

from pathlib import Path

PERSONA = Path(__file__).resolve().parent.parent / "deploy" / "brain-system.md"


def test_persona_has_sentinel_header():
    text = PERSONA.read_text()
    assert text.startswith("# d-brain session contract")


LEGACY = PERSONA.with_name("brain-system-markers.md")


def test_reply_contract_is_the_final_message():
    """Reply ledger (2026-10-29): the final message of the turn is the reply;
    the machine tag is never echoed; [SILENT] silences a background turn."""
    text = PERSONA.read_text()
    assert "The final message of your turn IS the reply" in text
    assert "<dbrain-turn" in text and "never repeat it" in text
    assert "[SILENT]" in text
    assert "<<<R:ID>>>" not in text  # the marker contract lives in LEGACY only


def test_legacy_marker_instruction_is_conditional():
    """The marker-path persona (used while the ledger is off): markers are
    required ONLY when the request carries the marker instruction (wrap=True
    turns). Verbatim/steered input gets a normal reply — no unconditional
    'every request' wording allowed."""
    text = LEGACY.read_text()
    assert text.startswith("# d-brain session contract")
    assert "Every request ends with an instruction" not in text
    low = text.lower()
    assert "when" in low and "marker" in low
    assert "no marker instruction" in low or "without a marker instruction" in low
    assert "<<<R:ID>>>" in text and "<<<E:ID>>>" in text


def test_persona_names_autograph_memory():
    assert "autograph" in PERSONA.read_text()


def test_persona_has_long_cascade_dispatch_rule():
    """agent-infra: the root turn of a multi-level agent
    cascade must dispatch and close, not stay open while sub-agents run —
    otherwise the pane looks identical to a wedged session from outside."""
    text = PERSONA.read_text()
    assert "## Long cascades — dispatch, then close the turn" in text
    assert "dispatches and closes" in text
    assert "15 minutes" in text
