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
- [x] Add extra attributes to money saved in Profile tab?
- [x] Update guide from Settings tab

### 🎮 Gamification
- [x] Improve gamification daily tasks
- [x] Create a service to check if daily tasks have been completed
- [x] Implement feedback buttons for adaptive difficulty
- [x] Store completed/failed tasks in SQLite to track tasks with high/low completion
- [x] Improve logic of Cooperative Goal
- [x] Create a service to check if weekly challenge has been completed
- [x] Implement adaptive difficulty
- [x] Need to round sensor values from tasks
- [ ] Store data needed for analysis/results extraction
- [ ] EXTRA: Add level system based on XP obtained from task/challenge completion? Or just level based on number of tasks completion
- [ ] Test gamification logic and UI


### 🤖 AI
- [x] Correct state vector gathered data
- [x] Fix state vector with 0º of temperature, etc.
- [x] Dismiss/Accept buttons for notifications
- [x] Improve anomaly index
- [x] Develop behaviour index
- [x] Develop fatigue index
- [x] Set default MAX_NOTIFICATIONS_PER_DAY value to 5
- [x] Check why it is still sending notifications when notification count exceeds MAX_NOTIFICATIONS_PER_DAY
- [ ] Check if indices are updating during the intervention
- [ ] Check adaptive difficulty updates
- [x] Q-table in JSON with duplicate key entries
- [ ] Check notification templates that make more sense
- [ ] Store data needed for analysis/results extraction
- [ ] Test AI-driven features (notifications, recommendations)

---
