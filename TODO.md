## ✅ GreenShift – TODO List

Checklist to track remaining features, fixes and tests before completion.

### 🧠 UX / Frontend
- [x] Show a pop-up when the intervention phase ends
- [x] Explain metrics in a clear way
- [x] Display charts with energy consumption data per area
- [x] Top 5 power consumers chart
- [x] Finish Dashboard interface
- [x] Add a UI input for "Weekly Reduction Target" to replace the hardcoded 0.85 (15%) value currently in the agent
- [x] Improve Profile interface
- [x] Improve Settings interface
- [x] Improve Gamification interface
 - [x] Display only number of generated tasks
- [x] Improve Monitored Devices tab
- [x] Improve Dashboard tab
- [x] Add emojis to places where they are missing to make screen more appealing
- [x] Add extra attributes to money saved in Profile tab
- [x] Update guide from Settings tab
- [x] Add metrics to Devices tab
- [x] Add presence state in text also, not just an emoji in Devices tab
- [x] Translations to Portuguese in the interface
    - [x] UI translation
    - [x] Daily tasks translations
    - [x] Notification translations
    - [x] Check during baseline
- [x] Check notification report when transitioning from baseline to active intervention
- [x] Test interface on mobile
- [x] Mention that user level is based in completed tasks from the last 30 days
- [x] Update hardcoded watt values from Live Focus Area table: use instead classifications based on mean consumption for each area
- [ ] Top 5 consumers section not appearing after setup
- [ ] "Entity Not Found" errors:
    - [ ] Current consumption cost not appearing
    - [ ] Gauge Week challenge not appearing
    - [ ] Section below gauge not appearing also
    - [ ] Engagement index not appearing in profile page
    - [ ] Accummulated savings not appearing in profile page
    - [ ] Weekly challenge not appearing in profile page
    - [ ] Energy goals slider not appearing in settings page
    - [ ] Currency dropdown not appearing in settings page
    - [ ] Eletricity dropdown not appearing in settings page
 
### 🎮 Gamification
- [x] Improve gamification daily tasks
- [x] Create a service to check if daily tasks have been completed
- [x] Implement feedback buttons for adaptive difficulty
- [x] Store completed/failed tasks in SQLite to track tasks with high/low completion
- [x] Improve logic of Cooperative Goal
- [x] Create a service to check if weekly challenge has been completed
- [x] Implement adaptive difficulty
- [x] Need to round sensor values from tasks
- [x] Store data needed for analysis/results extraction
- [x] Test adaptive difficulty
- [x] Test gamification logic and UI

### 🤖 AI
- [x] Correct state vector gathered data
- [x] Fix state vector with 0º of temperature, etc.
- [x] Dismiss/Accept buttons for notifications
- [x] Improve anomaly index
- [x] Develop behaviour index
- [x] Develop fatigue index
- [x] Check why it is still sending notifications when notification count exceeds MAX_NOTIFICATIONS_PER_DAY
- [x] Add mechanism that does not use notification timeouts. Must submit notification when most needed and when fatigue is not high.
- [x] Check if indices are updating during the intervention
- [x] Check adaptive difficulty updates
- [x] Fix Q-table in JSON with duplicate key entries
- [x] Test anomaly index with anomaly simulated anomaly scenario
- [x] Test nudge selection based on current environment and priority
- [x] Consider if 50 bins of power consumption is enough for intervention (decrease, keep or increase) - increased to 100
- [x] AI learning during baseline intervention
- [x] Store data needed for analysis/results extraction
- [x] Test AI-driven features (notifications, recommendations)
- [x] Analyze solution for office weekends
- [x] Add illuminance and presence logic. Control when using artificial lights when not needed.
- [x] Improve q-table. Current approach is storing rewards immediately after sending a notification. This way, the AI won't learn what are the best nudges to send to the user: need to make q-table update asynchronous, updating the q-table only when feedback button as been clicked.
- [x] When rejecting nudges, AI is getting positive rewards

### 📊 Data
- [x] Build tables required for analysis extraction
- [x] Store data required for analysis extraction
- [x] Build backup daily to prevent data loss
- [x] Check query 13
- [x] Improve research data queries
- [x] Manage memory usage
- [x] Analyze quantity of rl episodes being stored into the database (~60K entries per 21 days of runtime)
- [x] Consider if AI state history and notifications should be kept in persistent memory (JSON)
- [x] Store HA's weather info to analyze consumption based on heating degree days. Update storage accordingly to store temperature.
- [x] Store weather condition

### Validation
- [x] Add Unit tests
- [x] Check .gitignore
- [x] Add pre-commit hook to run tests before executing a commit
- [x] Improve code documentation where needed
- [ ] Update README

---
