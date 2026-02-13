# Green Shift Translation System

## Overview

Green Shift provides **complete multilingual support** with two translation systems:

1. **Static Translations** (JSON files) - For UI elements defined at integration load time
2. **Runtime Translations** (Python module) - For dynamically generated content (notifications, tasks)

### What's Translated

| Component | Languages | Method | Auto-Switch |
|-----------|-----------|--------|-------------|
| Sensors (16) | EN, PT | JSON | ✅ Yes |
| Services (8) | EN, PT | JSON | ✅ Yes |
| Config Flow | EN, PT | JSON | ✅ Yes |
| **AI Notifications** | **EN, PT** | **Runtime** | **✅ Yes** |
| **Daily Tasks** | **EN, PT** | **Runtime** | **✅ Yes** |
| Dashboard UI | EN, PT | Manual YAML | ❌ No |
| Input Helpers | EN, PT | Customize YAML | ❌ No |

---

## How It Works

Home Assistant's translation system is **automatic** and requires **no language picker** in the config flow. The system works as follows:

1. **User's Language**: Home Assistant detects the user's preferred language from their profile settings (Settings → Profile → Language)
2. **Automatic Loading**: HA automatically loads the corresponding translation file (e.g., `en.json`, `pt.json`, `es.json`)
3. **Fallback**: If a translation doesn't exist for the user's language, it falls back to English (`en.json`)

## Supported Languages

Currently, Green Shift supports:

- 🇬🇧 **English** (`en.json`) - Default
- 🇵🇹 **Portuguese** (`pt.json`) - Portugal

## Translation Structure

Translation files are located in: `config/custom_components/green_shift/translations/`

Each translation file follows this structure:

```json
{
  "config": {
    "step": { /* Config flow translations */ },
    "error": { /* Error messages */ }
  },
  "selector": {
    "currency": { /* Dropdown options */ }
  },
  "entity": {
    "sensor": { /* Sensor names and attributes */ },
    "select": { /* Select entity names */ }
  },
  "services": { /* Service names and descriptions */ }
}
```

## Runtime Translations (Dynamic Content)

Dynamic content like **AI notifications** and **daily tasks** cannot use static JSON files because they are generated at runtime with variable data. For these, we use a Python module: `translations_runtime.py`

### How Runtime Translations Work

1. **Language Detection**: System reads user's language from `hass.config.language`
2. **Template Selection**: Chooses appropriate template dictionary (`en`, `pt`, or `es`)
3. **Dynamic Formatting**: Fills templates with real-time data (power values, device names, etc.)

### Example: AI Notification

**Template (translations_runtime.py):**
```python
NOTIFICATION_TEMPLATES = {
    "pt": {
        "specific": [{
            "title": "Alerta de Consumo Elevado",
            "message": "🔌 {device_name} está a usar {device_power}W..."
        }]
    }
}
```

**Runtime Execution:**
```python
language = get_language(hass)  # → "pt"
templates = get_notification_templates(language)
message = templates["specific"][0]["message"].format(
    device_name="Aquecedor da Sala",
    device_power=1500
)
# Result: "🔌 Aquecedor da Sala está a usar 1500W..."
```

### What Uses Runtime Translations

- **AI Notifications** (4 types):
  - Specific device alerts
  - Anomaly detection warnings
  - Behavioural suggestions
  - Normative comparisons
  
- **Daily Tasks** (6 types):
  - Temperature reduction
  - Power reduction
  - Standby reduction
  - Daylight usage
  - Unoccupied power
  - Peak avoidance

- **Difficulty Levels**: Very Easy, Easy, Normal, Hard, Very Hard

### Adding Runtime Translations for New Language

Edit `translations_runtime.py` and add your language code to:
1. `NOTIFICATION_TEMPLATES["xx"]` - All notification templates
2. `TASK_TEMPLATES["xx"]` - All task title/description templates
3. `DIFFICULTY_DISPLAY["xx"]` - Difficulty level names
4. `TIME_OF_DAY["xx"]` - Time of day phrases

---

## Adding a New Language

To add support for a new language (e.g., French):

### 1. Create the Translation File

Create a new file: `translations/fr.json`

Use the ISO 639-1 language code:
- French: `fr`
- German: `de`
- Italian: `it`
- Dutch: `nl`
- etc.

### 2. Copy the English Template

Start by copying `en.json` and translating all strings:

```bash
cp translations/en.json translations/fr.json
```

### 3. Translate All Strings

Open `fr.json` and translate all text values, keeping the keys unchanged:

```json
{
  "config": {
    "step": {
      "user": {
        "title": "Bienvenue sur Green Shift",
        "description": "Green Shift combine..."
      }
    }
  }
}
```

### 4. Test Your Translation

1. Restart Home Assistant
2. Go to Settings → Profile → Language
3. Select your new language
4. Reconfigure or reload the Green Shift integration
5. All UI elements should now appear in the selected language

## Translation Best Practices

### DO ✅
- Keep translation keys unchanged (only translate values)
- Maintain the same JSON structure
- Use natural, idiomatic language
- Test all config flow steps
- Include proper punctuation and accents
- Preserve markdown formatting in descriptions

### DON'T ❌
- Translate JSON keys (e.g., `"name"`, `"description"`)
- Change the file structure
- Remove any translation entries
- Use machine translation without review
- Mix languages within one file

## Translating Input Helpers

Input helpers (like `input_number.energy_saving_target` and `input_select.currency`) need to be customized separately. Use the appropriate customize file for your language:

### Configuration

Edit your `configuration.yaml` and choose **ONE** customize file:

**For English:**
```yaml
homeassistant:
  customize: !include customize_en.yaml
```

**For Portuguese:**
```yaml
homeassistant:
  customize: !include customize_pt.yaml
```

**For Spanish:**
```yaml
homeassistant:
  customize: !include customize_es.yaml
```

⚠️ **Important**: You can only have **ONE** active `customize:` line. Comment out the others with `#`.

### Customize File Structure

The customize files should contain **only** the entity customizations without the `homeassistant:` and `customize:` headers:

```yaml
# customize_pt.yaml - CORRECT structure
input_number.energy_saving_target:
  friendly_name: "Meta de Poupança (%)"

input_select.currency:
  friendly_name: "Moeda"
```

❌ **WRONG** (will cause errors):
```yaml
homeassistant:
  customize:
    input_number.energy_saving_target:
      friendly_name: "Meta de Poupança (%)"
```

The customize files translate:
- Energy Saving Target slider
- Electricity Price input
- Currency selector
- Task Difficulty selector

**Note**: After adding or changing the customize configuration, restart Home Assistant for changes to take effect.

## Translating Lovelace UI

The Lovelace dashboard (`ui-lovelace.yaml`) contains hardcoded text that needs manual translation. 

### ✅ Available Translated Dashboards

- **English:** `ui-lovelace.yaml` (original)
- **Portuguese:** `ui-lovelace-pt.yaml` ✅ **READY TO USE**

### Quick Setup - Use Portuguese Dashboard

**Option 1: Replace Main Dashboard (Simplest)**
```bash
# Backup English version
mv config/ui-lovelace.yaml config/ui-lovelace-en.yaml

# Use Portuguese version
mv config/ui-lovelace-pt.yaml config/ui-lovelace.yaml

# Restart Home Assistant
```

**Option 2: Multiple Dashboards in Sidebar**

Add to your `configuration.yaml`:
```yaml
lovelace:
  mode: yaml
  dashboards:
    lovelace-pt:
      mode: yaml
      title: Green Shift (Português)
      icon: mdi:leaf
      show_in_sidebar: true
      filename: ui-lovelace-pt.yaml
    lovelace-en:
      mode: yaml
      title: Green Shift (English)
      icon: mdi:leaf
      show_in_sidebar: true
      filename: ui-lovelace.yaml
```

This allows you to switch between languages from the sidebar.

### Create Additional Language Versions

1. **Create Language-Specific YAML Files**
   - Copy `ui-lovelace.yaml` to `ui-lovelace-es.yaml` (for Spanish)

2. **Translate All Text Content**
   - Translate markdown headers (e.g., "# 🔌 Monitored Devices" → "# 🔌 Dispositivos Monitorizados")
   - Translate card titles and descriptions
   - Translate labels and helper text
   - Keep entity IDs unchanged (e.g., `sensor.current_consumption`)

3. **Add to Configuration**
   - Add new dashboard to `configuration.yaml` as shown in Option 2
   
   In your `configuration.yaml`:
   ```yaml
   lovelace:
     mode: yaml
     resources: []
     dashboards:
       green-shift-en:
         mode: yaml
         filename: ui-lovelace.yaml
         title: Green Shift (English)
         icon: mdi:leaf
       green-shift-pt:
         mode: yaml
         filename: ui-lovelace-pt.yaml
         title: Green Shift (Português)
         icon: mdi:leaf
       green-shift-es:
         mode: yaml
         filename: ui-lovelace-es.yaml
         title: Green Shift (Español)
         icon: mdi:leaf
   ```

4. **Select Your Preferred Dashboard**
   - Users can switch between dashboards from the sidebar

### Example Translations

**English:**
```yaml
- type: markdown
  content: |
    # 📊 Dashboard 
    Analyze your energy consumption patterns
```

**Portuguese:**
```yaml
- type: markdown
  content: |
    # 📊 Dashboard 
    Analise os seus padrões de consumo de energia
```

**Spanish:**
```yaml
- type: markdown
  content: |
    # 📊 Dashboard 
    Analiza tus patrones de consumo de energía
```

## Architecture Details

### Entity Translation Keys

All entities use `_attr_translation_key` to link to translation files:

```python
class CurrentConsumptionSensor(GreenShiftBaseSensor):
    def __init__(self, collector):
        self._attr_translation_key = "current_consumption"  # Links to en.json
        self._attr_unique_id = f"{DOMAIN}_current"
```

This links to:

```json
{
  "entity": {
    "sensor": {
      "current_consumption": {
        "name": "Current Consumption"
      }
    }
  }
}
```

### Config Flow Translations

Config flow steps automatically use the `config.step.<step_id>` structure:

```python
async def async_step_settings(self, user_input=None):
    # Automatically uses translations from config.step.settings
    return self.async_show_form(step_id="settings", ...)
```

### Service Translations

Services use the `services.<service_name>` structure:

```yaml
# services.yaml
submit_task_feedback:
  name: Submit Task Feedback  # Overridden by translation
  description: Provide feedback...  # Overridden by translation
```

## Testing Translations

### Visual Testing
1. Change your HA language
2. Check all UI elements:
   - Config flow steps
   - Sensor names
   - Service descriptions
   - Error messages

### Validation
- Ensure all JSON files are valid
- Check for missing or extra keys compared to `en.json`
- Test special characters and unicode

## Contributing Translations

If you'd like to contribute a new language:

1. Create the translation file following this guide
2. Test it thoroughly
3. Submit a pull request with:
   - The new translation file
   - Updates to this README listing the new language

## Common Issues

### Translation Not Showing
- **Restart required**: Restart HA after adding new translation files
- **Cache**: Clear browser cache
- **File name**: Ensure you used the correct ISO 639-1 code

### Partial Translations
- If some text remains in English, check for:
  - Missing translation keys
  - Typos in key names
  - JSON syntax errors

### Entity Names
- Entity translation requires `_attr_translation_key` + `_attr_has_entity_name = True`
- Without these, entities will show the hardcoded `_attr_name` value

## Quick Start Guides

- **🇵🇹 Portuguese Users**: See [TRADUCAO_RAPIDA.md](./TRADUCAO_RAPIDA.md) for quick activation guide
- **🇵🇹 Documentação Completa**: See [TRADUCAO.md](./TRADUCAO.md) for complete Portuguese documentation

## Architecture Notes

The Green Shift translation system uses a **hybrid approach**:

1. **JSON translations** (`translations/*.json`) - Loaded once at integration startup, cached by Home Assistant
2. **Runtime translations** (`translations_runtime.py`) - Executed dynamically when notifications or tasks are generated
3. **YAML translations** (`customize_*.yaml`, `ui-lovelace-*.yaml`) - Manual file switching required

This architecture ensures:
- ✅ Zero performance impact for static content
- ✅ Dynamic content always uses current language setting
- ✅ Support for placeholder/variable substitution in messages
- ✅ Easy to add new languages without code changes

---

**Made with 💚 for multilingual energy conservation**
