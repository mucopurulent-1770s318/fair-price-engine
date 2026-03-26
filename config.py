from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Ollama (free, local, private — used for both identifier and decomposer)
    OLLAMA_HOST: str = "http://localhost:11434"
    IDENTIFIER_MODEL: str = "llava"      # needs vision
    DECOMPOSER_MODEL: str = "llava"      # text-only works; swap to llama3.2 for speed

    # Confidence gate threshold — below this, UI shows confirmation step
    CONFIDENCE_THRESHOLD: float = Field(default=0.65, ge=0.0, le=1.0)

    # Pricing grounding mode
    GROUNDING_MODE: str = "llm"          # "llm" (MVP) | "live" (v1.1 with FRED/BLS)

    # Server
    PORT: int = 8000
    TLS_ENABLED: bool = False
    TLS_CERT_PATH: str = "certs/cert.pem"
    TLS_KEY_PATH: str  = "certs/key.pem"


settings = Settings()
