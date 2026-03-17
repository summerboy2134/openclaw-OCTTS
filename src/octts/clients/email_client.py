from __future__ import annotations

import smtplib
from email.message import EmailMessage
from typing import Optional

from octts.config import Settings


class EmailClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.smtp_host:
            raise ValueError("OCTTS_SMTP_HOST is required when email is enabled.")
        if not settings.smtp_username:
            raise ValueError("OCTTS_SMTP_USERNAME is required when email is enabled.")
        if not settings.smtp_password:
            raise ValueError("OCTTS_SMTP_PASSWORD is required when email is enabled.")

        self._host = settings.smtp_host
        self._port = settings.smtp_port
        self._username = settings.smtp_username
        self._password = settings.smtp_password
        self._use_tls = settings.smtp_use_tls
        self._timeout = settings.request_timeout_seconds

    def send_message(
        self,
        *,
        subject: str,
        body: str,
        recipients: list[str],
        attachments: Optional[list[tuple[str, bytes, str]]] = None,
    ) -> None:
        if not recipients:
            raise ValueError("At least one email recipient is required.")

        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self._username
        message["To"] = ", ".join(recipients)
        message.set_content(body)

        for filename, content, mime_type in attachments or []:
            maintype, subtype = mime_type.split("/", maxsplit=1)
            message.add_attachment(content, maintype=maintype, subtype=subtype, filename=filename)

        with smtplib.SMTP(self._host, self._port, timeout=self._timeout) as smtp:
            smtp.ehlo()
            if self._use_tls:
                smtp.starttls()
                smtp.ehlo()
            smtp.login(self._username, self._password)
            smtp.send_message(message)
