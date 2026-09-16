"""Capital.com authenticated session creation."""

from __future__ import annotations

import requests

import config


def create_session() -> tuple[str, str]:
    config.validate_live_config()
    url = f"{config.get_base_url()}/session"
    headers = {
        "X-CAP-API-KEY": config.API_KEY,
        "Content-Type": "application/json",
    }
    data = {
        "identifier": config.EMAIL,
        "password": config.PASSWORD,
        "encryptedPassword": False,
    }

    try:
        response = requests.post(
            url, headers=headers, json=data, timeout=config.REQUEST_TIMEOUT
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Capital.com session creation failed: {exc}") from exc

    cst = response.headers.get("CST")
    xst = response.headers.get("X-SECURITY-TOKEN")
    if not cst or not xst:
        raise RuntimeError("Capital.com did not return session tokens")
    return cst, xst
