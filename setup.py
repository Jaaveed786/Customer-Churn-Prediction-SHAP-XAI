from setuptools import setup, find_packages

setup(
    name="customer-churn-prediction",
    version="1.0.0",
    description="Customer Churn Prediction with Explainable AI (SHAP)",
    packages=find_packages(where="."),
    python_requires=">=3.10",
    install_requires=open("requirements.txt").read().splitlines(),
)
