PYTHON ?= python3

# -----------------------------------------------------------------------------
# Environment entries
# -----------------------------------------------------------------------------

# FHP (active)
FHP_DB := ./fhp_data/fhp_db/research_data.db
FHP_EXPORT_DIR := ./fhp_data/fhp_research_data
FHP_CLEAN_DIR := ./fhp_data/fhp_cleaned_data

# FEUP (active)
FEUP_DB := ./feup_data/feup_db/research_data.db
FEUP_EXPORT_DIR := ./feup_data/feup_research_data
FEUP_CLEAN_DIR := ./feup_data/feup_cleaned_data

# HOME1 (planned)
# HOME1_DB := ./home1_data/home1_db/research_data.db
# HOME1_EXPORT_DIR := ./home1_data/home1_research_data
# HOME1_CLEAN_DIR := ./home1_data/home1_cleaned_data

# HOME2 (planned)
# HOME2_DB := ./home2_data/home2_db/research_data.db
# HOME2_EXPORT_DIR := ./home2_data/home2_research_data
# HOME2_CLEAN_DIR := ./home2_data/home2_cleaned_data

FORMS_FILE := ./global_data/research_participant_survey.csv
FORMS_FILE_FOLDER_OUTPUT := ./global_data

.PHONY: all export_all clean_all merge

export_all:
	$(PYTHON) export_research.py $(FHP_DB) $(FHP_EXPORT_DIR)
	$(PYTHON) export_research.py $(FEUP_DB) $(FEUP_EXPORT_DIR)
	# $(PYTHON) export_research.py $(HOME1_DB) $(HOME1_EXPORT_DIR)
	# $(PYTHON) export_research.py $(HOME2_DB) $(HOME2_EXPORT_DIR)

clean_all:
	$(PYTHON) clean_research_data.py $(FHP_EXPORT_DIR) $(FHP_CLEAN_DIR) --forms $(FORMS_FILE) --forms-output $(FORMS_FILE_FOLDER_OUTPUT)
	$(PYTHON) clean_research_data.py $(FEUP_EXPORT_DIR) $(FEUP_CLEAN_DIR) 
	# $(PYTHON) clean_research_data.py $(HOME1_EXPORT_DIR) $(HOME1_CLEAN_DIR) 
	# $(PYTHON) clean_research_data.py $(HOME2_EXPORT_DIR) $(HOME2_CLEAN_DIR)

merge:
	$(PYTHON) merge_research_data.py $(FHP_CLEAN_DIR) $(FEUP_CLEAN_DIR)
	# $(PYTHON) merge_research_data.py $(FHP_CLEAN_DIR) $(FEUP_CLEAN_DIR) $(HOME1_CLEAN_DIR) $(HOME2_CLEAN_DIR)

all: export_all clean_all merge
