
try:
    import google.generativeai
    import dotenv
    print("Dependencies installed.")
except ImportError as e:
    print(f"Missing dependency: {e}")
