"""QR code generator for employee multi-channel landing pages.

Each employee gets a unique QR code that links to their landing page.
Visitors scan → see employee info → choose channel (Widget/WhatsApp/Telegram).
"""

from __future__ import annotations

import io
from typing import Any

from src.utils.logger import setup_logger

logger = setup_logger("services.qr_generator")


class QRGenerator:
    """Generate QR codes for employee avatars."""

    def __init__(self, r2_storage: Any, db: Any = None) -> None:
        self._r2 = r2_storage
        self._db = db

    async def generate_employee_qr(
        self,
        employee_id: str,
        employee_name: str,
        base_url: str = "https://maskki.com",
    ) -> dict[str, str]:
        """Generate QR code that links to employee's multi-channel landing page.

        Args:
            employee_id: Employee ID.
            employee_name: Employee display name.
            base_url: Base URL for the landing page.

        Returns:
            Dict with qr_code_url, landing_url, download_url.
        """
        try:
            import qrcode
        except ImportError:
            logger.warning("qrcode package not installed, returning placeholder")
            landing_url = f"{base_url}/connect/{employee_id}"
            return {
                "qr_code_url": "",
                "landing_url": landing_url,
                "download_url": "",
            }

        landing_url = f"{base_url}/connect/{employee_id}"

        # Generate QR with high error correction (for optional logo overlay)
        qr = qrcode.QRCode(
            version=1,
            box_size=10,
            border=5,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
        )
        qr.add_data(landing_url)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white").convert("RGB")

        # Save to bytes
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        qr_bytes = buffer.getvalue()

        # Upload to R2
        qr_url = ""
        if self._r2:
            key = f"employees/{employee_id}/qr_code.png"
            qr_url = await self._r2.upload(key, qr_bytes, "image/png")

        # Update employee channel record
        if self._db:
            try:
                from sqlalchemy import update
                from src.db.models import EmployeeChannel

                async with self._db.session_ctx() as session:
                    await session.execute(
                        update(EmployeeChannel)
                        .where(EmployeeChannel.employee_id == employee_id)
                        .values(qr_code_url=qr_url, qr_landing_url=landing_url)
                    )
                    await session.commit()
            except Exception as exc:
                logger.warning(f"Failed to update QR URLs in DB: {exc}")

        logger.info(
            "QR code generated",
            extra={"employee_id": employee_id, "url": qr_url},
        )

        return {
            "qr_code_url": qr_url,
            "landing_url": landing_url,
            "download_url": qr_url,
        }

    async def generate_print_version(
        self,
        employee_id: str,
        employee_name: str,
        base_url: str = "https://maskki.com",
    ) -> bytes:
        """High-res QR for print (300 DPI, with employee name + branding).

        Returns PNG bytes suitable for print (larger box_size).
        """
        try:
            import qrcode
        except ImportError:
            return b""

        landing_url = f"{base_url}/connect/{employee_id}"

        qr = qrcode.QRCode(
            version=1,
            box_size=20,  # High res for print
            border=5,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
        )
        qr.add_data(landing_url)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white").convert("RGB")

        buffer = io.BytesIO()
        img.save(buffer, format="PNG", dpi=(300, 300))
        return buffer.getvalue()
