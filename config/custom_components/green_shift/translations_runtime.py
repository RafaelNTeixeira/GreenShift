"""
File: translations_runtime.py
Description: This module provides runtime translation support for the Green Shift Home Assistant component.
It includes functions to detect the user's preferred language from Home Assistant's configuration and retrieve the appropriate notification and task templates based on that language.
The module defines a set of notification templates for different categories (specific, anomaly, behavioural, normative, phase_transition) in both English and Portuguese, as well as task templates and difficulty display names.
This allows the component to dynamically generate notifications and tasks in the user's preferred language, enhancing the user experience and making the energy-saving suggestions more relatable and actionable.
"""

import logging

_LOGGER = logging.getLogger(__name__)

# Notification Templates by Language
NOTIFICATION_TEMPLATES = {
    "en": {
        "specific": [
            {
                "title": "High Consumption Alert",
                "message": "🔌 {device_name} is currently using {device_power}W, which is higher than usual. Consider turning it off when not in use."
            },
            {
                "title": "Appliance Usage Notice",
                "message": "⚡ Your {device_name} is drawing {device_power}W right now. If you're not actively using it, switching it off could save energy."
            }
        ],

        "anomaly": [
            {
                "title": "Unusual Consumption Pattern",
                "message": "📊 Your current power usage ({current_power}W) is {percent_above}% higher than your typical baseline ({baseline_power}W). Check if any devices were left on accidentally."
            },
            {
                "title": "Energy Anomaly Detected",
                "message": "🔍 We've detected unusual energy consumption in your {area_name}. The {metric} levels are outside the normal range. Worth investigating?"
            },
            {
                "title": "Consumption Spike Alert",
                "message": "⚠️ Your energy use just spiked to {current_power}W (normal: {baseline_power}W). This could indicate an appliance malfunction or unusual activity."
            },
            {
                "title": "Area Anomaly Notice",
                "message": "🏠 {area_name} is showing unusual patterns - {metric} readings are significantly different from normal. Everything okay there?"
            }
        ],

        "behavioural": [
            {
                "title": "Bedtime Energy Tip",
                "message": "🌙 It's {time_of_day} - remember to turn off devices in standby mode before bed. Small actions like this can save up to 10% on your energy bill.",
                "context_filter": "nighttime"
            },
            {
                "title": "Smart Habit Suggestion",
                "message": "💚 Try unplugging chargers when not in use. They consume power even when devices aren't connected - a simple habit that adds up over time."
            },
            {
                "title": "Comfort & Efficiency Tip",
                "message": "🌡️ Your {area_name} is at {area_temp}°C. Adjusting by just 1-2 degrees can save significant energy while maintaining comfort."
            },
            {
                "title": "Lighting Optimization",
                "message": "💡 Natural light is available during the day. Consider opening blinds instead of using artificial lighting when possible."
            },
            {
                "title": "Daylight Power Alert",
                "message": "☀️ It's a bright {time_of_day} with good natural light and you're consuming {current_power}W. Could some lights be turned off to save energy?",
                "context_filter": "daylight_waste"
            },
            {
                "title": "Bright Day Savings",
                "message": "🌞 Great lighting conditions detected during {time_of_day}! Your current power consumption is {current_power}W - turning off unnecessary lights could reduce this significantly.",
                "context_filter": "daylight_waste"
            },
            {
                "title": "Nobody Home Alert",
                "message": "🏠 No presence detected, but you're still consuming {current_power}W (baseline: {baseline_power}W). Are there devices that could be turned off?",
                "context_filter": "away_mode"
            },
            {
                "title": "Unoccupied Energy Use",
                "message": "🚶‍♂️ It looks like nobody's around, but power consumption is at {current_power}W. Consider turning off unused devices to save energy while away.",
                "context_filter": "away_mode"
            }
        ],

        "normative": [
            {
                "title": "Weekly Goal Update",
                "message": "🎯 Your consumption this week is {percent_above}% above target. You're close to achieving your {target_power}W goal - keep it up!"
            },
            {
                "title": "Progress Check-In",
                "message": "🏆 You've saved energy before - your best week showed {target_power}W average. Current week: {current_power}W. You can do it again!"
            },
            {
                "title": "Benchmark Update",
                "message": "📈 Your current energy use ({current_power}W) is {percent_above}% above your personal best. Let's work together to improve this week."
            },
            {
                "title": "Target Achievement",
                "message": "🌟 Great progress! You're {percent_above}% away from your weekly reduction target. A few small changes could close the gap."
            }
        ],

        "phase_transition": {
            "title": "Green Shift: Action Phase Started",
            "message": "### Baseline Phase Complete! 🎉\n\n**Daily Average:** {avg_daily_kwh} kWh\n**Peak Usage:** {peak_time}\n{top_area_section}**Target:** We've set a **{target}%** reduction goal for you (you can change this in the Settings tab)\n\n---\n### Your Potential Impact 🌍\nBy hitting your **{target}%** reduction goal, in one year you would save:\n* **{co2_kg} kg** of CO₂\n* The equivalent of planting **{trees}** mature trees\n* The carbon offset of **{flights}** short-haul flights\n"
        }
    },

    "pt": {
        "specific": [
            {
                "title": "Alerta de Consumo Elevado",
                "message": "🔌 {device_name} está atualmente a usar {device_power}W, o que é superior ao habitual. Considere desligá-lo quando não estiver em uso."
            },
            {
                "title": "Aviso de Uso de Aparelho",
                "message": "⚡ O seu {device_name} está a consumir {device_power}W neste momento. Se não o está a usar ativamente, desligá-lo pode poupar energia."
            }
        ],

        "anomaly": [
            {
                "title": "Padrão de Consumo Incomum",
                "message": "📊 O seu consumo de energia atual ({current_power}W) está {percent_above}% acima da sua baseline típica ({baseline_power}W). Verifique se deixou algum dispositivo ligado acidentalmente."
            },
            {
                "title": "Anomalia de Energia Detetada",
                "message": "🔍 Detetámos consumo de energia incomum na sua {area_name}. Os níveis de {metric} estão fora do intervalo normal. Vale a pena investigar?"
            },
            {
                "title": "Alerta de Pico de Consumo",
                "message": "⚠️ O seu uso de energia acabou de aumentar para {current_power}W (normal: {baseline_power}W). Isto pode indicar uma avaria num aparelho ou atividade incomum."
            },
            {
                "title": "Aviso de Anomalia na Área",
                "message": "🏠 {area_name} está a mostrar padrões incomuns - as leituras de {metric} são significativamente diferentes do normal. Está tudo bem aí?"
            }
        ],

        "behavioural": [
            {
                "title": "Dica de Energia para a Noite",
                "message": "🌙 É {time_of_day} - lembre-se de desligar dispositivos em modo standby antes de dormir. Pequenas ações como esta podem poupar até 10% na sua conta de energia.",
                "context_filter": "nighttime"
            },
            {
                "title": "Sugestão de Hábito Inteligente",
                "message": "💚 Experimente desligar carregadores quando não estiverem em uso. Eles consomem energia mesmo quando os dispositivos não estão conectados - um hábito simples que se acumula ao longo do tempo."
            },
            {
                "title": "Dica de Conforto & Eficiência",
                "message": "🌡️ A sua {area_name} está a {area_temp}°C. Ajustar apenas 1-2 graus pode poupar energia significativa mantendo o conforto."
            },
            {
                "title": "Otimização de Iluminação",
                "message": "💡 A luz natural está disponível durante o dia. Considere abrir as persianas em vez de usar iluminação artificial quando possível."
            },
            {
                "title": "Alerta de Consumo Diurno",
                "message": "☀️ É um {time_of_day} com boa luz natural e está a consumir {current_power}W. Será que algumas luzes podem ser desligadas para poupar energia?",
                "context_filter": "daylight_waste"
            },
            {
                "title": "Poupança em Dia Claro",
                "message": "🌞 Boas condições de iluminação natural detetadas durante {time_of_day}! O seu consumo atual é {current_power}W - desligar luzes desnecessárias poderia reduzir isto significativamente.",
                "context_filter": "daylight_waste"
            },
            {
                "title": "Alerta de Ausência",
                "message": "🏠 Nenhuma presença detetada, mas ainda está a consumir {current_power}W (baseline: {baseline_power}W). Há dispositivos que possam ser desligados?",
                "context_filter": "away_mode"
            },
            {
                "title": "Uso de Energia Desocupado",
                "message": "🚶‍♂️ Parece que não há ninguém, mas o consumo está em {current_power}W. Considere desligar dispositivos não utilizados para poupar energia durante a ausência.",
                "context_filter": "away_mode"
            }
        ],

        "normative": [
            {
                "title": "Atualização de Meta Semanal",
                "message": "🎯 O seu consumo esta semana está {percent_above}% acima da meta. Está perto de atingir o seu objetivo de {target_power}W - continue assim!"
            },
            {
                "title": "Verificação de Progresso",
                "message": "🏆 Já poupou energia antes - a sua melhor semana mostrou uma média de {target_power}W. Semana atual: {current_power}W. Consegue fazê-lo novamente!"
            },
            {
                "title": "Atualização de Benchmark",
                "message": "📈 O seu uso de energia atual ({current_power}W) está {percent_above}% acima do seu melhor pessoal. Vamos trabalhar juntos para melhorar esta semana."
            },
            {
                "title": "Conquista de Meta",
                "message": "🌟 Ótimo progresso! Está a {percent_above}% de distância da sua meta de redução semanal. Algumas pequenas mudanças podem fechar a diferença."
            }
        ],

        "phase_transition": {
            "title": "Green Shift: Fase de Ação Iniciada",
            "message": "### Fase de Baseline Completa! 🎉\n\n**Média Diária:** {avg_daily_kwh} kWh\n**Pico de Uso:** {peak_time}\n{top_area_section}**Meta:** Definimos uma meta de redução de **{target}%** para si (pode alterá-la no separador Definições)\n\n---\n### O Seu Impacto Potencial 🌍\nAo atingir a sua meta de redução de **{target}%**, num ano pouparia:\n* **{co2_kg} kg** de CO₂\n* O equivalente a plantar **{trees}** árvores maduras\n* A compensação de carbono de **{flights}** voos de curta distância\n"
        }
    }
}

# Task Templates by Language
TASK_TEMPLATES = {
    "en": {
        "temperature_reduction": {
            "title": "Reduce Temperature by {reduction:.1f}°C",
            "description": "Keep average temperature below {target_temp:.1f}°C today (current avg: {baseline_temp:.1f}°C)"
        },
        "temperature_increase": {
            "title": "Raise AC Setpoint by {increase:.1f}°C",
            "description": "Allow average temperature to stay above {target_temp:.1f}°C today (current avg: {baseline_temp:.1f}°C) - reduces over-cooling energy use"
        },
        "power_reduction": {
            "title": "Reduce Power by {reduction_pct:.1f}%",
            "description": "Keep average power below {target_power:.0f}W today (7-day avg: {baseline_power:.0f}W)"
        },
        "daylight_usage": {
            "title": "Use Natural Light ({reduction_pct:.1f}% less power)",
            "description": "Keep daytime power (08:00-17:00) below {target_power:.0f}W by using natural light"
        },
        "unoccupied_power": {
            "title": "Turn Off Devices in {target_area}",
            "description": "Reduce power in {target_area} to below {target_power:.0f}W when unoccupied"
        },
        "peak_avoidance": {
            "title": "Reduce Peak Hour Usage ({peak_hour:02d}:00)",
            "description": "Keep power below {target_power:.0f}W during {peak_hour:02d}:00-{next_hour:02d}:00"
        }
    },

    "pt": {
        "temperature_reduction": {
            "title": "Reduzir Temperatura em {reduction:.1f}°C",
            "description": "Mantenha a temperatura média abaixo de {target_temp:.1f}°C hoje (média atual: {baseline_temp:.1f}°C)"
        },
        "temperature_increase": {
            "title": "Aumentar Setpoint do AC em {increase:.1f}°C",
            "description": "Permita que a temperatura média fique acima de {target_temp:.1f}°C hoje (média atual: {baseline_temp:.1f}°C) - reduz o consumo de arrefecimento excessivo"
        },
        "power_reduction": {
            "title": "Reduzir Potência em {reduction_pct:.1f}%",
            "description": "Mantenha a potência média abaixo de {target_power:.0f}W hoje (média 7 dias: {baseline_power:.0f}W)"
        },
        "daylight_usage": {
            "title": "Usar Luz Natural ({reduction_pct:.1f}% menos potência)",
            "description": "Mantenha a potência diurna (08:00-17:00) abaixo de {target_power:.0f}W usando luz natural"
        },
        "unoccupied_power": {
            "title": "Desligar Dispositivos em {target_area}",
            "description": "Reduza a potência em {target_area} para abaixo de {target_power:.0f}W quando desocupado"
        },
        "peak_avoidance": {
            "title": "Reduzir Uso na Hora de Pico ({peak_hour:02d}:00)",
            "description": "Mantenha a potência abaixo de {target_power:.0f}W durante {peak_hour:02d}:00-{next_hour:02d}:00"
        }
    }
}

# Difficulty Display Names
DIFFICULTY_DISPLAY = {
    "en": {
        1: "Very Easy",
        2: "Easy",
        3: "Normal",
        4: "Hard",
        5: "Very Hard"
    },
    "pt": {
        1: "Muito Fácil",
        2: "Fácil",
        3: "Normal",
        4: "Difícil",
        5: "Muito Difícil"
    }
}

# Time of Day Translations
TIME_OF_DAY = {
    "en": {
        "morning": "morning",
        "day": "daytime",
        "afternoon": "afternoon",
        "evening": "evening",
        "night": "nighttime"
    },
    "pt": {
        "morning": "manhã",
        "day": "durante o dia",
        "afternoon": "tarde",
        "evening": "noite",
        "night": "madrugada"
    }
}


async def get_language(hass) -> str:
    """
    Get user's preferred language from Home Assistant.
    Falls back to English if not found.

    Returns:
        str: The 2-letter ISO code of the user's preferred language (e.g., 'en', 'pt')
    """
    try:
        # Try to get language from HA configuration
        if hasattr(hass.config, 'language'):
            lang = hass.config.language
            # Normalize to 2-letter ISO code
            if lang and len(lang) >= 2:
                lang_code = lang[:2].lower()
                # Check if we have translations for this language
                if lang_code in NOTIFICATION_TEMPLATES:
                    _LOGGER.info("✓ Detected language from HA config: %s", lang_code)
                    return lang_code
                else:
                    _LOGGER.debug("Language %s not in available translations: %s", lang_code, list(NOTIFICATION_TEMPLATES.keys()))

        _LOGGER.info("No specific language detected, defaulting to English")

        # Default fallback
        return "en"
    except Exception as e:
        _LOGGER.warning("Error detecting language, defaulting to English: %s", e)
        return "en"


def get_notification_templates(language: str) -> dict:
    """
    Get notification templates for specified language.
    
    Args:
        language (str): The 2-letter ISO code of the language to retrieve templates for

    Returns:
        dict: A dictionary containing notification templates for the specified language, or English templates if the language is not available.
    """
    return NOTIFICATION_TEMPLATES.get(language, NOTIFICATION_TEMPLATES["en"])


def get_phase_transition_template(language: str) -> dict:
    """
    Get phase transition template for specified language.
    
    Args:
        language (str): The 2-letter ISO code of the language to retrieve the template for

    Returns:
        dict: A dictionary containing the phase transition template for the specified language, or the English template if the language is not available.
    """
    templates = NOTIFICATION_TEMPLATES.get(language, NOTIFICATION_TEMPLATES["en"])
    return templates.get("phase_transition", NOTIFICATION_TEMPLATES["en"]["phase_transition"])


def get_task_templates(language: str) -> dict:
    """
    Get task templates for specified language.
    
    Args:
        language (str): The 2-letter ISO code of the language to retrieve templates for

    Returns:
        dict: A dictionary containing task templates for the specified language, or English templates if the language is not available.
    """
    return TASK_TEMPLATES.get(language, TASK_TEMPLATES["en"])


def get_difficulty_display(difficulty: int, language: str) -> str:
    """
    Get difficulty display name in specified language.
    
    Args:
        difficulty (int): The difficulty level (1-5)
        language (str): The 2-letter ISO code of the language to retrieve the display name for

    Returns:
        str: The difficulty display name in the specified language, or "Normal" if not available.
    """
    return DIFFICULTY_DISPLAY.get(language, DIFFICULTY_DISPLAY["en"]).get(difficulty, "Normal")


def get_time_of_day_name(time_key: str, language: str) -> str:
    """
    Get time of day name in specified language.
    
    Args:
        time_key (str): The key for the time of day (e.g., "morning", "day", etc.)
        language (str): The 2-letter ISO code of the language to retrieve the name for

    Returns:
        str: The time of day name in the specified language, or the original key if not available.
    """
    return TIME_OF_DAY.get(language, TIME_OF_DAY["en"]).get(time_key, time_key)


# Verification reason template strings
# Keys used by task_manager.verify_tasks() to produce the UI reason field
VERIFICATION_REASON_TEMPLATES = {
    "en": {
        "target_achieved":         "Target achieved",
        "waiting_for_peak_hour":   "Waiting for peak hour ({peak_hour:02d}:00)",
        "evaluation_deferred":     "Evaluation deferred",
        "avg_above_target":        "Avg: {actual}{unit}, target was {target}{unit}",
        "insufficient_data":       "Insufficient data",
        # Task-type specific failure reasons
        "temp_reduction_failed":   "Avg temp: {actual}°C - target was ≤{target}°C (reduce heating)",
        "temp_increase_failed":    "Avg temp: {actual}°C - target was ≥{target}°C (raise AC setpoint to save cooling energy)",
        "power_above_target":      "Avg power: {actual}W - target was ≤{target}W",
        "daylight_above_target":   "Daytime avg: {actual}W - target was ≤{target}W (use more natural light)",
        "unoccupied_above_target": "Unoccupied avg: {actual}W - target was ≤{target}W (turn off idle devices)",
        "peak_above_target":       "Peak-hour avg: {actual}W - target was ≤{target}W (shift usage away from peak)",
    },
    "pt": {
        "target_achieved":         "Objetivo atingido",
        "waiting_for_peak_hour":   "À espera da hora de pico ({peak_hour:02d}:00)",
        "evaluation_deferred":     "Avaliação adiada",
        "avg_above_target":        "Média: {actual}{unit}, objetivo era {target}{unit}",
        "insufficient_data":       "Dados insuficientes",
        # Task-type specific failure reasons
        "temp_reduction_failed":   "Temp. média: {actual}°C - objetivo era ≤{target}°C (reduzir aquecimento)",
        "temp_increase_failed":    "Temp. média: {actual}°C - objetivo era ≥{target}°C (aumentar setpoint do AC para poupar energia de arrefecimento)",
        "power_above_target":      "Potência média: {actual}W - objetivo era ≤{target}W",
        "daylight_above_target":   "Média diurna: {actual}W - objetivo era ≤{target}W (aproveitar mais luz natural)",
        "unoccupied_above_target": "Média desocupado: {actual}W - objetivo era ≤{target}W (desligar dispositivos inativos)",
        "peak_above_target":       "Média hora pico: {actual}W - objetivo era ≤{target}W (evitar consumo na ponta)",
    },
}


def get_verification_reason_templates(language: str) -> dict:
    """
    Get task-verification reason string templates for the specified language.

    Args:
        language (str): 2-letter ISO language code (e.g. 'en', 'pt')

    Returns:
        dict: Mapping of reason keys to format strings.
    """
    return VERIFICATION_REASON_TEMPLATES.get(language, VERIFICATION_REASON_TEMPLATES["en"])
