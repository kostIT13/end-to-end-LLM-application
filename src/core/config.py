from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    LOG_LEVEL: str = "INFO"
    DEBUG: bool = False

    LLM_BASE_URL: str
    LLM_API_KEY: str
    LLM_MODEL_PRIMARY: str
    LLM_MODEL_CHEAP: str
    EMBED_MODEL: str
    EMBED_DIMENSION: str 

    PGVECTOR_URL: str
    PGVECTOR_DB: str
    PGVECTOR_HOST: str
    PGVECTOR_PASSWORD: str
    PGVECTOR_USER: str
    PGVECTOR_PORT: int = 5432

    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 500

    REQUES_TIMEOUT_S: int = 30
    MAX_RETRIES: int = 3

settings = Settings()