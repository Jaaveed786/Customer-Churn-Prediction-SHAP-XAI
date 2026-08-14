.PHONY: install train app test clean

install:
	pip install -r requirements.txt

train:
	python -m src.train
	python -m src.generate_shap_plots

app:
	streamlit run app/streamlit_app.py

test:
	pytest tests/ -v

clean:
	rmdir /s /q models 2>nul || true
	rmdir /s /q reports\figures 2>nul || true
