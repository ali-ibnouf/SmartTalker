"""SSRF protection and input sanitization for Custom API Tools.

Blocks requests to internal/private networks and cloud metadata endpoints.
Applied both at tool creation time (URL validation) and execution time.
"""

from __future__ import annotations

import ipaddress
import json
import socket
from urllib.parse import urlparse

from src.utils.logger import setup_logger

logger = setup_logger("agent.security")

# Blocked hostnames (metadata endpoints, loopback, etc.)
BLOCKED_HOSTS = {
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
    "metadata.google.internal",
    "169.254.169.254",       # AWS/GCP/Azure metadata
    "100.100.100.200",       # Alibaba Cloud metadata
}

# Blocked private/reserved IP networks
BLOCKED_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),    # IPv6 private
    ipaddress.ip_network("fe80::/10"),   # IPv6 link-local
]


def validate_tool_url(url: str) -> bool:
    """Validate that a tool URL does not point to internal/private addresses.

    Args:
        url: The URL to validate.

    Returns:
        True if the URL is safe to call, False if blocked.
    """
    if not url:
        return False

    parsed = urlparse(url)
    hostname = parsed.hostname

    if not hostname:
        return False

    # Block non-HTTP protocols
    if parsed.scheme not in ("http", "https"):
        logger.warning("SSRF: blocked non-HTTP scheme", extra={"url": url, "scheme": parsed.scheme})
        return False

    # Block known dangerous hostnames
    if hostname.lower() in BLOCKED_HOSTS:
        logger.warning("SSRF: blocked hostname", extra={"url": url, "hostname": hostname})
        return False

    # Resolve DNS and check IP against blocked networks
    try:
        resolved_ip = socket.gethostbyname(hostname)
        ip = ipaddress.ip_address(resolved_ip)
        for network in BLOCKED_NETWORKS:
            if ip in network:
                logger.warning(
                    "SSRF: blocked private IP",
                    extra={"url": url, "hostname": hostname, "resolved_ip": resolved_ip},
                )
                return False
    except (socket.gaierror, ValueError):
        # DNS resolution failed — block by default
        logger.warning("SSRF: DNS resolution failed", extra={"url": url, "hostname": hostname})
        return False

    return True


def sanitize_tool_input(input_data: dict) -> dict:
    """Sanitize tool input data by round-tripping through JSON.

    Removes non-serializable objects and normalizes types.

    Args:
        input_data: Raw input data from the LLM.

    Returns:
        Sanitized dict safe for API calls.
    """
    try:
        return json.loads(json.dumps(input_data, default=str))
    except (TypeError, ValueError):
        return {}
