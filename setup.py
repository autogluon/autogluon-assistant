from setuptools import find_packages, setup

setup(
    name="AutoMLAgent",
    version="0.1.0",
    package_dir={"": "automlagent/src"},  # Updated path
    packages=find_packages(where="automlagent/src"),  # Updated path
    install_requires=[
        "langgraph",
        "autogluon",
        "autogluon.tabular[skex]",
        # "speechbrain",
        "FlagEmbedding",
        "faiss-cpu",
        "importlib-resources>=6.4.5",
        "langchain>=0.3.3",
        "langchain_openai>=0.2.2",
        "langchain_aws>=0.2.2",
        "pydantic>=2.9.2",
        "hydra-core>=1.3",
        "matplotlib>=3.9.2",
        "typer>=0.12.5",
        "rich>=13.8.1",
        "s3fs>=2025.3.2",
        "fsspec>=2025.3.2",
        "joblib>=1.4.2",
        "python-calamine",
        "tenacity>=8.2.2",
        "pandas>=2.2",
        "streamlit>=1.37",
        "streamlit-aggrid>=1.0.2",
        "streamlit-extras>=0.4",
        "psutil>=5.9.8",
    ],
    author="FANGAreNotGnu",
    description="AutoMLAgent beta",
)
