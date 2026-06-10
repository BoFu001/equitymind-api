"""
api/auth.py

API key authentication for equitymind-core.

Current:  returns True for all keys (placeholder)
Future:   sends api_key to equitymind-portal-backend for validation
          replace the body of verify_api_key with portal HTTP call
          query.py remains unchanged
"""

import logging

logger = logging.getLogger(__name__)


async def verify_api_key(api_key: str) -> bool:
    """
    Validates API key against equitymind-portal-backend.

    Returns True if valid, False otherwise.

    TODO: Replace body when equitymind-portal-backend is ready:
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{PORTAL_URL}/api/v1/auth/verify",
            json={"api_key": api_key},
            headers={"Authorization": f"Bearer {PORTAL_SECRET}"},
            timeout=5.0,
        )
        data = response.json()
        return data.get("valid", False)
    """
    # PLACEHOLDER — accepts all keys until portal is ready
    return True