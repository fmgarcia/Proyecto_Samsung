try:
    import datasets
    import transformers
    print("Imports successful!")
    print(f"Datasets version: {datasets.__version__}")
    print(f"Transformers version: {transformers.__version__}")
except Exception as e:
    print(f"Import failed: {e}")
