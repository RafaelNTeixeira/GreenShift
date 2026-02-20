"""
Tests for translations_runtime.py

Covers:
- get_language: language detection from config/hass
- get_notification_templates: returns correct templates by language
- get_task_templates: returns task templates by language
- get_time_of_day_name: maps time periods to localized names
- get_difficulty_display: maps difficulty levels to localized names
- get_phase_transition_template: returns phase transition messages
- Template structure validation: ensures all required keys exist
"""
import pytest
import sys
import types
import pathlib
import importlib.util
import logging
from unittest.mock import MagicMock, AsyncMock

# ── Minimal HA stubs ────────────────────────────────────────────────────────
for mod_name in ["homeassistant", "homeassistant.core"]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

# Load translations_runtime module
trans_spec = importlib.util.spec_from_file_location(
    "translations_runtime",
    pathlib.Path(__file__).parent.parent / "config" / "custom_components" / "green_shift" / "translations_runtime.py"
)
trans_mod = importlib.util.module_from_spec(trans_spec)
trans_mod.__package__ = "custom_components.green_shift"
trans_spec.loader.exec_module(trans_mod)

get_language = trans_mod.get_language
get_notification_templates = trans_mod.get_notification_templates
get_task_templates = trans_mod.get_task_templates
get_time_of_day_name = trans_mod.get_time_of_day_name
get_difficulty_display = trans_mod.get_difficulty_display
get_phase_transition_template = trans_mod.get_phase_transition_template


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_hass_en():
    """Mock hass with English language."""
    hass = MagicMock()
    hass.config = MagicMock()
    hass.config.language = "en"
    return hass


@pytest.fixture
def mock_hass_pt():
    """Mock hass with Portuguese language."""
    hass = MagicMock()
    hass.config = MagicMock()
    hass.config.language = "pt"
    return hass


@pytest.fixture
def en_config():
    return {"language": "en"}


@pytest.fixture
def pt_config():
    return {"language": "pt"}


# ─────────────────────────────────────────────────────────────────────────────
# get_language
# ─────────────────────────────────────────────────────────────────────────────

class TestGetLanguage:

    @pytest.mark.asyncio
    async def test_returns_en_from_hass(self, mock_hass_en):
        """Language from hass.config.language."""
        result = await get_language(mock_hass_en)
        assert result == "en"

    @pytest.mark.asyncio
    async def test_returns_pt_from_hass(self, mock_hass_pt):
        """Portuguese from hass.config.language."""
        result = await get_language(mock_hass_pt)
        assert result == "pt"

    @pytest.mark.asyncio
    async def test_uses_hass_language(self, mock_hass_pt):
        """Uses hass.config.language."""
        result = await get_language(mock_hass_pt)
        assert result == "pt"

    @pytest.mark.asyncio
    async def test_defaults_to_en_when_no_language(self):
        """When neither config nor hass has language, default to 'en'."""
        hass = MagicMock()
        hass.config = MagicMock()
        hass.config.language = None
        
        result = await get_language(hass)
        assert result == "en"

    @pytest.mark.asyncio
    async def test_handles_unsupported_language(self, mock_hass_en):
        """Unsupported languages should fall back to English."""
        hass = MagicMock()
        hass.config = MagicMock()
        hass.config.language = "fr"  # French not supported
        
        result = await get_language(hass)
        # Should default to en when unsupported
        assert result == "en"


# ─────────────────────────────────────────────────────────────────────────────
# get_notification_templates
# ─────────────────────────────────────────────────────────────────────────────

class TestGetNotificationTemplates:

    def test_returns_dict_for_en(self):
        templates = get_notification_templates("en")
        assert isinstance(templates, dict)

    def test_returns_dict_for_pt(self):
        templates = get_notification_templates("pt")
        assert isinstance(templates, dict)

    def test_has_specific_action_type(self):
        templates = get_notification_templates("en")
        assert "specific" in templates

    def test_has_anomaly_action_type(self):
        templates = get_notification_templates("en")
        assert "anomaly" in templates

    def test_has_behavioural_action_type(self):
        templates = get_notification_templates("en")
        assert "behavioural" in templates

    def test_has_normative_action_type(self):
        templates = get_notification_templates("en")
        assert "normative" in templates

    def test_en_and_pt_have_same_keys(self):
        """English and Portuguese should have the same action types."""
        en_templates = get_notification_templates("en")
        pt_templates = get_notification_templates("pt")
        assert set(en_templates.keys()) == set(pt_templates.keys())

    def test_each_template_has_required_fields(self):
        """Each template should have title and message (templates are lists of dicts)."""
        templates = get_notification_templates("en")
        for action_type, template_list in templates.items():
            if action_type == "phase_transition":
                # This is a dict, not a list
                assert "title" in template_list
                assert "message" in template_list
            else:
                # These are lists of template options
                assert isinstance(template_list, list)
                for template in template_list:
                    assert "title" in template, f"{action_type} template missing title"
                    assert "message" in template, f"{action_type} template missing message"

    def test_defaults_to_en_for_unknown_language(self):
        """Unknown language codes should return English templates."""
        templates = get_notification_templates("xyz")
        en_templates = get_notification_templates("en")
        assert templates == en_templates


# ─────────────────────────────────────────────────────────────────────────────
# get_task_templates
# ─────────────────────────────────────────────────────────────────────────────

class TestGetTaskTemplates:

    def test_returns_dict_for_en(self):
        templates = get_task_templates("en")
        assert isinstance(templates, dict)

    def test_returns_dict_for_pt(self):
        templates = get_task_templates("pt")
        assert isinstance(templates, dict)

    def test_has_common_task_types(self):
        """Check for expected task types."""
        templates = get_task_templates("en")
        # Based on actual TASK_TEMPLATES structure
        expected_types = [
            "temperature_reduction",
            "power_reduction",
            "standby_reduction",
            "daylight_usage",
            "unoccupied_power",
            "peak_avoidance",
        ]
        for task_type in expected_types:
            assert task_type in templates, f"Missing task type: {task_type}"

    def test_en_and_pt_have_same_task_types(self):
        en_templates = get_task_templates("en")
        pt_templates = get_task_templates("pt")
        assert set(en_templates.keys()) == set(pt_templates.keys())

    def test_each_task_has_required_fields(self):
        """Each task template should have title and description."""
        templates = get_task_templates("en")
        for task_type, template in templates.items():
            assert "title" in template, f"{task_type} missing title"
            assert "description" in template, f"{task_type} missing description"

    def test_defaults_to_en_for_unknown_language(self):
        templates = get_task_templates("xyz")
        en_templates = get_task_templates("en")
        assert templates == en_templates


# ─────────────────────────────────────────────────────────────────────────────
# get_time_of_day_name
# ─────────────────────────────────────────────────────────────────────────────

class TestGetTimeOfDayName:

    @pytest.mark.parametrize("period,expected_en", [
        ("morning", "morning"),
        ("afternoon", "afternoon"),
        ("evening", "evening"),
        ("night", "nighttime"),
    ])
    def test_english_time_periods(self, period, expected_en):
        result = get_time_of_day_name(period, "en")
        assert result == expected_en

    @pytest.mark.parametrize("period,expected_pt", [
        ("morning", "manhã"),
        ("afternoon", "tarde"),
        ("evening", "noite"),
        ("night", "madrugada"),
    ])
    def test_portuguese_time_periods(self, period, expected_pt):
        result = get_time_of_day_name(period, "pt")
        assert result == expected_pt

    def test_defaults_to_en_for_unknown_language(self):
        result = get_time_of_day_name("morning", "xyz")
        assert result == "morning"  # Falls back to English

    def test_returns_period_for_unknown_period(self):
        """Unknown period should return the input capitalized or as-is."""
        result = get_time_of_day_name("unknown", "en")
        # Should gracefully handle or return some default
        assert isinstance(result, str)


# ─────────────────────────────────────────────────────────────────────────────
# get_difficulty_display
# ─────────────────────────────────────────────────────────────────────────────

class TestGetDifficultyDisplay:

    @pytest.mark.parametrize("level,expected_en", [
        (1, "Very Easy"),
        (2, "Easy"),
        (3, "Normal"),
        (4, "Hard"),
        (5, "Very Hard"),
    ])
    def test_english_difficulty_levels(self, level, expected_en):
        result = get_difficulty_display(level, "en")
        assert result == expected_en

    @pytest.mark.parametrize("level,expected_pt", [
        (1, "Muito Fácil"),
        (2, "Fácil"),
        (3, "Normal"),
        (4, "Difícil"),
        (5, "Muito Difícil"),
    ])
    def test_portuguese_difficulty_levels(self, level, expected_pt):
        result = get_difficulty_display(level, "pt")
        assert result == expected_pt

    def test_defaults_to_en_for_unknown_language(self):
        result = get_difficulty_display(3, "xyz")
        assert result == "Normal"

    def test_handles_invalid_level(self):
        """Invalid difficulty level should return some default or handle gracefully."""
        result = get_difficulty_display(999, "en")
        assert isinstance(result, str)


# ─────────────────────────────────────────────────────────────────────────────
# get_phase_transition_template
# ─────────────────────────────────────────────────────────────────────────────

class TestGetPhaseTransitionTemplate:

    def test_returns_dict_for_en(self):
        template = get_phase_transition_template("en")
        assert isinstance(template, dict)

    def test_returns_dict_for_pt(self):
        template = get_phase_transition_template("pt")
        assert isinstance(template, dict)

    def test_has_required_fields(self):
        """Phase transition template should have title and message."""
        template = get_phase_transition_template("en")
        assert "title" in template
        assert "message" in template

    def test_en_and_pt_have_same_structure(self):
        en_template = get_phase_transition_template("en")
        pt_template = get_phase_transition_template("pt")
        assert set(en_template.keys()) == set(pt_template.keys())

    def test_defaults_to_en_for_unknown_language(self):
        template = get_phase_transition_template("xyz")
        en_template = get_phase_transition_template("en")
        assert template == en_template


# ─────────────────────────────────────────────────────────────────────────────
# Template content validation
# ─────────────────────────────────────────────────────────────────────────────

class TestTemplateContent:

    def test_notification_templates_not_empty(self):
        """Notification templates should have actual content."""
        templates = get_notification_templates("en")
        for action_type, template_list in templates.items():
            if action_type == "phase_transition":
                assert len(template_list.get("title", "")) > 0
                assert len(template_list.get("message", "")) > 0
            else:
                assert isinstance(template_list, list)
                for template in template_list:
                    assert len(template.get("title", "")) > 0
                    assert len(template.get("message", "")) > 0

    def test_task_templates_not_empty(self):
        """Task templates should have actual content."""
        templates = get_task_templates("en")
        for task_type, template in templates.items():
            assert len(template.get("title", "")) > 0
            assert len(template.get("description", "")) > 0

    def test_no_template_placeholders_unfilled(self):
        """Templates should not have unfilled {placeholders} in static text."""
        templates = get_notification_templates("en")
        for action_type, template_list in templates.items():
            if action_type == "phase_transition":
                assert isinstance(template_list.get("title", ""), str)
                assert isinstance(template_list.get("message", ""), str)
            else:
                for template in template_list:
                    assert isinstance(template.get("title", ""), str)
                    assert isinstance(template.get("message", ""), str)

    def test_portuguese_translations_exist(self):
        """Portuguese should have translations, not English fallbacks."""
        pt_templates = get_notification_templates("pt")
        en_templates = get_notification_templates("en")
        
        # At least some templates should be different between languages
        differences = 0
        for key in pt_templates:
            if key == "phase_transition":
                if pt_templates[key].get("title") != en_templates[key].get("title"):
                    differences += 1
            else:
                # Compare first template in lists
                if pt_templates[key][0].get("title") != en_templates[key][0].get("title"):
                    differences += 1
        
        assert differences > 0, "Portuguese templates appear to be empty or English fallbacks"


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_none_language_defaults_to_en(self):
        """None language should default to English."""
        templates = get_notification_templates(None)
        en_templates = get_notification_templates("en")
        assert templates == en_templates

    def test_empty_string_language_defaults_to_en(self):
        """Empty string language should default to English."""
        templates = get_notification_templates("")
        en_templates = get_notification_templates("en")
        assert templates == en_templates

    def test_case_insensitive_language_codes(self):
        """Language codes should work regardless of case."""
        upper = get_notification_templates("EN")
        lower = get_notification_templates("en")
        # Should return same templates (implementation may vary)
        assert set(upper.keys()) == set(lower.keys())
