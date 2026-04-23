@echo off
call "C:\Users\SRI CHARAN\miniconda3\Scripts\activate.bat"
conda activate tf-gpu
pip install "tensorflow<2.11" keras pandas numpy matplotlib scikit-learn seaborn jupyter flask requests beautifulsoup4 pytest openpyxl shap scipy joblib
cd project/main_files
python PIML_training.py
