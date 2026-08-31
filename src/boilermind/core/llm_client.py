from typing import Any
import os


class LLMClient:


    def __init__(self):

        self.api_key = os.getenv(
            "DASHSCOPE_API_KEY"
        )

        self.model = os.getenv(
            "BOILERMIND_QWEN_MODEL",
            "qwen3.7-plus",
        )

        self.base_url = os.getenv(
            "BOILERMIND_QWEN_BASE_URL"
        )

        self.timeout = float(
            os.getenv("BOILERMIND_QWEN_TIMEOUT", "240")
        )

        self.max_tokens = int(
            os.getenv("BOILERMIND_QWEN_MAX_TOKENS", "4096")
        )

        self.max_retries = int(
            os.getenv("BOILERMIND_QWEN_MAX_RETRIES", "2")
        )


    def generate(
        self,
        prompt: str
    ) -> Any:


        if not self.api_key:

            raise RuntimeError(
                "DASHSCOPE_API_KEY_NOT_FOUND"
            )


        if not self.base_url:
            raise RuntimeError(
                "BOILERMIND_QWEN_BASE_URL_NOT_FOUND"
            )

        import httpx
        from openai import OpenAI

        with httpx.Client(
            trust_env=False,
            timeout=self.timeout,
        ) as http_client:
            with OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=http_client,
                timeout=self.timeout,
                max_retries=self.max_retries,
            ) as client:
                response = client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                )

        return response.choices[0].message.content or ""
