# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Camera Selection popup control regressions."""

from pathlib import Path


def test_distinct_and_cropbox_controls_use_editable_scrub_fields():
    root = Path(__file__).parents[2]
    panel = (root / "src/python/lfs_plugins/camera_selection_panel.py").read_text(encoding="utf-8")
    popup = (root / "src/visualizer/gui/rmlui/resources/camera_selection.rml").read_text(encoding="utf-8")

    assert "ScrubFieldController" in panel
    assert '"percentage": ScrubFieldSpec(0, 100, 1, "%d", data_type=int)' in panel
    assert '"overlap": ScrubFieldSpec(0, 100, 1, "%d", data_type=int)' in panel
    assert '"coverage": ScrubFieldSpec(0, 100, 1, "%d", data_type=int)' in panel
    assert "self._scrub_fields.mount(doc)" in panel
    assert "self._scrub_fields.sync_all()" in panel
    assert 'data-value="percentage" min="0" max="100" step="1"' in popup
    assert 'data-value="overlap" min="0" max="100" step="1"' in popup
    assert 'data-value="coverage" min="0" max="100" step="1"' in popup


def test_cropbox_popup_registers_its_record_list_and_keeps_coverage_retained():
    root = Path(__file__).parents[2]
    panel = (root / "src/python/lfs_plugins/camera_selection_panel.py").read_text(encoding="utf-8")
    popup = (root / "src/visualizer/gui/rmlui/resources/camera_selection.rml").read_text(encoding="utf-8")

    assert 'data-class-hidden="!is_distinct"' in popup
    assert 'model.bind_record_list("cropboxes")' in panel
    assert '<div data-if="is_distinct">' not in popup
    assert 'data-if="is_cropbox"' in popup
    assert 'data-class-hidden="!is_cropbox"' in popup
    assert "if len(records) == 1 and not self._cropbox_id:" in panel
    assert 'self._cropbox_id = records[0]["id"]' in panel


if __name__ == "__main__":
    test_distinct_and_cropbox_controls_use_editable_scrub_fields()
    test_cropbox_popup_registers_its_record_list_and_keeps_coverage_retained()
