from app.core.config import get_settings

settings = get_settings()

print(f"Open ai api key: {settings.deepseek_api_key}")
