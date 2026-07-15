from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    log_level: str = "INFO"

    llm_base_url: str
    llm_api_key: str
    llm_model_primary: str
    llm_model_cheap: str
    embed_model: str
    embed_dimension: str 

    llm_temperature: float = 0.1
    llm_max_tokens: int = 500

    request_timeout_s: int = 30
    max_retries: int = 3

settings = Settings()