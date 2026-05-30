import sys
import traceback
from src.config import Config
from src.extractor import Extractor

if __name__ == "__main__":
    try:
        config = Config()
        provider = getattr(config.model, "provider", "")
        model_name = getattr(config.model, "name", "")
        api_key = getattr(config.model, "api_key", "")
        masked_key = api_key[:10] + "..." if api_key else "<empty>"

        print(f"provider={provider}")
        print(f"model_name={model_name}")
        print(f"api_key_masked={masked_key}")

        extractor = Extractor(config)
        prompt = 'Return ONLY valid JSON: {"name":"John Doe","email":"john@example.com","phone":"","skills":[],"experience":[],"education":[],"certifications":[]}'
        result = extractor._call_gemini(prompt)

        print("\nparsed_result=")
        print(result)
    except Exception as exc:
        print("Exception occurred:")
        traceback.print_exc()
