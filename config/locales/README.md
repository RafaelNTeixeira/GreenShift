# Locales - Translation Files

This folder contains YAML translation files for the Green Shift integration.

## 📁 Contents

### Dashboard Translations
- **[`ui-lovelace-en.yaml`](./ui-lovelace-en.yaml)** - English dashboard
- **[`ui-lovelace-pt.yaml`](./ui-lovelace-pt.yaml)** - Portuguese dashboard

These files contain complete dashboard translations including:
- All tab titles and content
- Markdown descriptions and instructions
- Card titles and labels
- Button texts

### Helper Entity Translations
- **[`customize_en.yaml`](./customize_en.yaml)** - English entity names
- **[`customize_pt.yaml`](./customize_pt.yaml)** - Portuguese entity names

These files translate the friendly names of:
- `input_number.energy_saving_target` - Savings target slider
- `input_number.electricity_price` - Electricity price input
- `input_select.currency` - Currency selector
- `input_select.task_difficulty` - Task difficulty selector

## 🔧 Configuration

To activate a language, edit [`configuration.yaml`](../configuration.yaml):

```yaml
homeassistant:
  customize: !include locales/customize_pt.yaml  # or customize_en.yaml

lovelace:
  dashboards:
    lovelace-green-shift:
      mode: yaml
      filename: locales/ui-lovelace-pt.yaml  # or ui-lovelace-en.yaml
      title: Green Shift
      icon: mdi:leaf
```

## 🌍 Adding New Languages

To add a new language (e.g., Spanish):

1. **Copy English files as templates:**
   ```bash
   cp ui-lovelace-en.yaml ui-lovelace-es.yaml
   cp customize_en.yaml customize_es.yaml
   ```

2. **Translate all text content:**
   - Translate markdown headers and descriptions
   - Translate card titles and labels
   - Translate friendly names in customize file
   - Keep entity IDs unchanged (e.g., `sensor.current_consumption`)

3. **Update configuration.yaml:**
   ```yaml
   homeassistant:
     customize: !include locales/customize_es.yaml
   
   lovelace:
     dashboards:
       lovelace-green-shift:
         filename: locales/ui-lovelace-es.yaml
   ```

4. **Restart Home Assistant**

## 📚 Documentation

For complete translation guide, see:
- **[Translation Guide](../../docs/TRANSLATIONS.md)** - Complete translation system documentation
- **[Contributing Guide](../../docs/CONTRIBUTING.md)** - How to contribute translations

---

**Note:** These YAML files handle **dashboard and helper translations only**. The integration's core translations (sensors, services, config flow, AI notifications, tasks) are managed separately through JSON files and Python runtime translations.
