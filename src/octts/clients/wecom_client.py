from __future__ import annotations

from octts.config import Settings


class WeComClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.wecom_webhook_url:
            raise ValueError("WECOM_WEBHOOK_URL is required.")
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("requests is not installed.") from exc
        self._requests = requests
        self._webhook_url = settings.wecom_webhook_url
        self._timeout = settings.request_timeout_seconds

    def send_markdown(self, content: str) -> None:
        response = self._requests.post(
            self._webhook_url,
            json={
                "msgtype": "markdown",
                "markdown": {
                    "content": content,
                },
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("errcode") not in (0, "0", None):
            raise ValueError(f"WeCom webhook rejected message: {payload}")
