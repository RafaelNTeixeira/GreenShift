# Contributing to Green Shift

Thank you for your interest in contributing to Green Shift! This document provides guidelines and instructions for contributing to the project.

## 📋 Table of Contents

- [Ways to Contribute](#ways-to-contribute)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Guidelines](#code-guidelines)
- [Adding Translations](#adding-translations)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)
- [Feature Requests](#feature-requests)
- [Testing](#testing)
- [Contact](#contact)

---

## 🤝 Ways to Contribute

You can contribute to Green Shift in several ways:

### 1. **Translations** 🌍
Help make Green Shift accessible to more users by adding translations for new languages.

👉 **See [Translation Guide](./TRANSLATIONS.md)** for detailed instructions.

Currently supported:
- 🇬🇧 English
- 🇵🇹 Portuguese

### 2. **Bug Reports** 🐛
Report bugs you encounter with detailed information to help us fix them quickly.

### 3. **Feature Requests** ✨
Suggest new features or improvements to enhance Green Shift.

### 4. **Code Contributions** 💻
Submit code improvements, bug fixes or new features via pull requests.

### 5. **Documentation** 📚
Improve documentation, fix typos or add examples.

### 6. **Testing** 🧪
Test new features and provide feedback on pre-release versions.

---

## 🚀 Getting Started

### Prerequisites

- Home Assistant installation 
- Git for version control
- Text editor (VS Code recommended)

### Fork and Clone

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/GreenShift.git
   cd GreenShift
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/RafaelNTeixeira/GreenShift.git
   ```

---

## 🛠️ Development Setup

### 1. Install Prerequisites

```bash
# Install required Python packages
pip install numpy>=1.21.0
```

### 2. Link to Home Assistant

Create a symbolic link to test your changes:

```bash
# Navigate to HA custom_components folder
cd /config/custom_components/

# Create symbolic link to your development folder
ln -s /path/to/your/GreenShift/config/custom_components/green_shift green_shift
```

**Key Files Structure:**
- [`config/custom_components/green_shift/`](../config/custom_components/green_shift/) - Integration code
- [`config/custom_components/green_shift/translations/`](../config/custom_components/green_shift/translations/) - JSON translation files
- [`config/locales/`](../config/locales/) - YAML translation files (dashboards & helpers)
- [`config/configuration.yaml`](../config/configuration.yaml) - HA configuration

### 3. Enable Debug Logging

Add to your [`configuration.yaml`](../config/configuration.yaml):

```yaml
logger:
  default: info
  logs:
    custom_components.green_shift: debug
```

### 4. Restart Home Assistant

After making changes, restart HA to load your modifications:

```bash
ha core restart
```

---

## 📝 Code Guidelines

### Python Style

- **PEP 8**: Follow Python style guidelines
- **Type hints**: Use type annotations where applicable
- **Docstrings**: Document all functions and classes
- **Logging**: Use appropriate log levels (`_LOGGER.debug/info/warning/error`)

### Code Structure

```python
async def example_function(hass: HomeAssistant, param: str) -> dict:
    """
    Brief description of what the function does.
    
    Args:
        hass: Home Assistant instance
        param: Description of parameter
    
    Returns:
        Dictionary with results
    """
    _LOGGER.debug("Function called with param: %s", param)
    # Implementation
    return {"result": "value"}
```

### File Organization

**Integration Code** ([`config/custom_components/green_shift/`](../config/custom_components/green_shift/)):
- **[`__init__.py`](../config/custom_components/green_shift/__init__.py)**: Integration setup and entry point
- **[`config_flow.py`](../config/custom_components/green_shift/config_flow.py)**: Configuration flow UI
- **[`sensor.py`](../config/custom_components/green_shift/sensor.py)**: Sensor entity definitions
- **[`select.py`](../config/custom_components/green_shift/select.py)**: Select entity definitions
- **[`const.py`](../config/custom_components/green_shift/const.py)**: Constants and configuration
- **[`data_collector.py`](../config/custom_components/green_shift/data_collector.py)**: Real-time data collection
- **[`decision_agent.py`](../config/custom_components/green_shift/decision_agent.py)**: AI decision-making logic
- **[`task_manager.py`](../config/custom_components/green_shift/task_manager.py)**: Task generation and verification
- **[`storage.py`](../config/custom_components/green_shift/storage.py)**: Database management
- **[`helpers.py`](../config/custom_components/green_shift/helpers.py)**: Utility functions
- **[`translations_runtime.py`](../config/custom_components/green_shift/translations_runtime.py)**: Dynamic content translations

**Translation Files** ([`config/custom_components/green_shift/translations/`](../config/custom_components/green_shift/translations/)):
- **[`en.json`](../config/custom_components/green_shift/translations/en.json)**: English translations
- **[`pt.json`](../config/custom_components/green_shift/translations/pt.json)**: Portuguese translations

**Configuration Files** ([`config/`](../config/)):
- **[`configuration.yaml`](../config/configuration.yaml)**: Main HA configuration

**Locale Files** ([`config/locales/`](../config/locales/)):
- **[`ui-lovelace-en.yaml`](../config/locales/ui-lovelace-en.yaml)**: English dashboard
- **[`ui-lovelace-pt.yaml`](../config/locales/ui-lovelace-pt.yaml)**: Portuguese dashboard
- **[`customize_en.yaml`](../config/locales/customize_en.yaml)**: English helper names
- **[`customize_pt.yaml`](../config/locales/customize_pt.yaml)**: Portuguese helper names

### Best Practices

✅ **DO:**
- Write clear, self-documenting code
- Add comments for complex logic
- Handle exceptions gracefully
- Use async/await for I/O operations
- Test your changes thoroughly
- Keep functions focused and small

❌ **DON'T:**
- Commit commented-out code
- Use print statements (use logging instead)
- Make breaking changes without discussion
- Commit directly to main branch
- Include personal/sensitive data

---

## 🌍 Adding Translations

Green Shift uses a dual translation system:

### 1. Static UI Translations (JSON)

Translate sensors, services and config flow:

1. Create `translations/XX.json` (e.g., `es.json` for Spanish)
2. Copy structure from `translations/en.json`
3. Translate all text values (keep keys unchanged)

### 2. Dynamic Content Translations (Python)

Translate AI notifications and tasks in [`translations_runtime.py`](../config/custom_components/green_shift/translations_runtime.py):

1. Edit [`translations_runtime.py`](../config/custom_components/green_shift/translations_runtime.py)
2. Add your language code to:
   - `NOTIFICATION_TEMPLATES["XX"]`
   - `TASK_TEMPLATES["XX"]`
   - `DIFFICULTY_DISPLAY["XX"]`
   - `TIME_OF_DAY["XX"]`

### 3. Dashboard Translations (YAML)

1. Copy [`config/locales/ui-lovelace-en.yaml`](../config/locales/ui-lovelace-en.yaml) to `config/locales/ui-lovelace-XX.yaml`
2. Translate all markdown content and card titles
3. Keep entity IDs unchanged

### 4. Helper Translations (YAML)

1. Copy [`config/locales/customize_en.yaml`](../config/locales/customize_en.yaml) to `config/locales/customize_XX.yaml`
2. Translate all `friendly_name` values
3. Update [`configuration.yaml`](../config/configuration.yaml) to reference your new file:
   ```yaml
   homeassistant:
     customize: !include locales/customize_XX.yaml
   
   lovelace:
     dashboards:
       lovelace-green-shift:
         filename: locales/ui-lovelace-XX.yaml
   ```

👉 **Complete guide:** [TRANSLATIONS.md](./TRANSLATIONS.md)

---

## 📤 Submitting Changes

### Branch Naming

Use descriptive branch names:

```bash
feature/add-spanish-translation
bugfix/notification-cooldown
improvement/reduce-memory-usage
docs/update-readme
```

### Commit Messages

Write clear commit messages. Use conventional commit prefixes:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `imp:` - Improvements

### Pull Request Process

1. **Update your fork**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes** and commit:
   ```bash
   git add .
   git commit -m "feat: description of changes"
   ```

4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request** on GitHub:
   - Provide clear title and description
   - Reference related issues (e.g., "Fixes #123")
   - Include screenshots if UI changes
   - List testing steps

### Pull Request Checklist

- [ ] Code follows style guidelines
- [ ] Changes have been tested
- [ ] Documentation updated if needed
- [ ] Commit messages are clear
- [ ] No merge conflicts with main
- [ ] Translation files are valid JSON
- [ ] Debug logging added where appropriate

---

## 🐛 Reporting Issues

### Before Reporting

- **Search existing issues** to avoid duplicates
- **Test on latest version** of Green Shift
- **Check Home Assistant logs** for error messages

### Issue Template

When reporting a bug, include:

```markdown
**Description:**
Clear description of the issue

**Expected Behavior:**
What should happen

**Actual Behavior:**
What actually happens

**Steps to Reproduce:**
1. Step one
2. Step two
3. ...

**Environment:**
- Home Assistant version: 2024.1.0
- Green Shift version: 1.0.0
- Installation method: HACS / Manual

**Logs:**
```
Paste relevant logs from Home Assistant
```

**Screenshots:**
If applicable, add screenshots
```

### Log Collection

Enable debug logging and collect logs:

```yaml
logger:
  logs:
    custom_components.green_shift: debug
```

Then check: **Settings → System → Logs**

---

## ✨ Feature Requests

We welcome feature suggestions! Please:

1. **Check existing issues** to avoid duplicates
2. **Describe the feature** clearly
3. **Explain use case** - How would it help?
4. **Consider scope** - Is it aligned with project goals?

### Feature Request Template

```markdown
**Feature Description:**
Brief description of the feature

**Use Case:**
Why is this feature needed?

**Proposed Implementation:**
How could this work?

**Alternatives Considered:**
Other approaches you've thought about

**Additional Context:**
Any other relevant information
```

---

## 🧪 Testing

### Manual Testing

1. **Test all config flow steps**
2. **Verify sensor values** are correct
3. **Check notifications** appear properly
4. **Test task generation and verification**
5. **Validate translations** in all supported languages
6. **Test error conditions**

### Test Checklist

Integration Setup:
- [ ] Config flow completes successfully
- [ ] Sensors are created
- [ ] Dashboard loads without errors

Baseline Phase (First 14 days):
- [ ] No notifications sent
- [ ] Data collection working
- [ ] Baseline calculation accurate

Active Phase (Day 15+):
- [ ] Notifications appear
- [ ] Tasks generated daily
- [ ] Feedback system works
- [ ] Translations correct

Special Cases:
- [ ] Missing sensors handled gracefully
- [ ] Invalid input rejected
- [ ] Recovery after HA restart

---

## 📞 Contact

For questions, discussions or direct contact:

- **GitHub Issues**: [Create an issue](https://github.com/RafaelNTeixeira/GreenShift/issues)
- **Email**: rafael2003t.18@gmail.com
- **Pull Requests**: Reviewed regularly

---


## 🙏 Thank You!

Every contribution, no matter how small, helps make Green Shift better for everyone. We appreciate your time and effort!

**Happy Contributing! 🌱⚡**
