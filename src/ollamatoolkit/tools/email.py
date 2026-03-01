# ./ollamatoolkit/tools/email.py
"""
OllamaToolkit - Email Tools
===========================
Wrapper for emailtoolkit providing email validation, extraction, and analysis.

Enables Ollama models to:
- Validate email addresses (syntax and DNS)
- Extract emails from text
- Canonicalize emails for comparison
- Check domain health
- Filter disposable/temporary emails

Usage:
    from ollamatoolkit.tools.email import EmailTools

    tools = EmailTools()

    # Validate
    is_valid = tools.is_valid("test@example.com")

    # Extract from text
    emails = tools.extract("Contact: alice@example.com or bob@gmail.com")

    # Check domain
    health = tools.check_domain("example.com")
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EmailInfo:
    """Parsed email information."""

    original: str
    normalized: str
    canonical: str
    local: str
    domain: str
    valid_syntax: bool
    valid_dns: Optional[bool] = None
    disposable: bool = False
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original": self.original,
            "normalized": self.normalized,
            "canonical": self.canonical,
            "local": self.local,
            "domain": self.domain,
            "valid_syntax": self.valid_syntax,
            "valid_dns": self.valid_dns,
            "disposable": self.disposable,
            "reason": self.reason,
        }


@dataclass
class DomainHealth:
    """Domain health check result."""

    domain: str
    has_mx: bool
    has_a: bool
    mx_hosts: List[str]
    a_hosts: List[str]
    disposable: bool

    @property
    def deliverable(self) -> bool:
        """Check if domain can receive email."""
        return self.has_mx or self.has_a

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "domain": self.domain,
            "has_mx": self.has_mx,
            "has_a": self.has_a,
            "mx_hosts": self.mx_hosts,
            "a_hosts": self.a_hosts,
            "disposable": self.disposable,
            "deliverable": self.deliverable,
        }


class EmailTools:
    """
    Email validation, extraction, and analysis tools.

    Wraps the emailtoolkit package for use with Ollama agents.
    """

    def __init__(self):
        """Initialize email tools."""
        self._et = None
        self._init_emailtoolkit()

    def _init_emailtoolkit(self):
        """Lazy-load emailtoolkit."""
        try:
            import emailtoolkit as et

            self._et = et
            logger.debug("emailtoolkit loaded successfully")
        except ImportError:
            logger.warning(
                "emailtoolkit not installed. Run: pip install emailtoolkit[dns]"
            )
            self._et = None

    @property
    def available(self) -> bool:
        """Check if emailtoolkit is available."""
        return self._et is not None

    def is_valid(self, email: str) -> bool:
        """
        Check if an email address is valid.

        Args:
            email: Email address to validate

        Returns:
            True if valid, False otherwise
        """
        if not self._et:
            logger.error("emailtoolkit not available")
            return False

        return self._et.is_valid(email)

    def parse(self, email: str) -> Optional[EmailInfo]:
        """
        Parse and validate an email address.

        Args:
            email: Email address to parse

        Returns:
            EmailInfo with parsed details, or None if invalid
        """
        if not self._et:
            return None

        try:
            parsed = self._et.parse(email)
            return EmailInfo(
                original=parsed.original,
                normalized=parsed.normalized,
                canonical=parsed.canonical,
                local=parsed.local,
                domain=parsed.domain,
                valid_syntax=parsed.valid_syntax,
                valid_dns=parsed.deliverable_dns,
                disposable=parsed.domain_info.disposable
                if parsed.domain_info
                else False,
                reason=parsed.reason,
            )
        except Exception as e:
            logger.debug(f"Failed to parse email: {e}")
            return None

    def normalize(self, email: str) -> Optional[str]:
        """
        Normalize an email address (lowercase domain).

        Args:
            email: Email address

        Returns:
            Normalized email or None if invalid
        """
        if not self._et:
            return None

        try:
            return self._et.normalize(email)
        except Exception:
            return None

    def canonical(self, email: str) -> Optional[str]:
        """
        Get canonical form (handles Gmail dots, plus-addressing).

        Args:
            email: Email address

        Returns:
            Canonical email or None if invalid
        """
        if not self._et:
            return None

        try:
            return self._et.canonical(email)
        except Exception:
            return None

    def compare(self, email1: str, email2: str) -> bool:
        """
        Compare two emails by canonical identity.

        Correctly determines that test.user@gmail.com and
        testuser+sales@googlemail.com are the same.

        Args:
            email1: First email
            email2: Second email

        Returns:
            True if same identity, False otherwise
        """
        if not self._et:
            return False

        try:
            return self._et.compare(email1, email2)
        except Exception:
            return False

    def extract(
        self,
        text: str,
        unique: bool = True,
        max_results: Optional[int] = None,
    ) -> List[EmailInfo]:
        """
        Extract email addresses from text.

        Handles:
        - Free text with emails
        - mailto: links
        - Cloudflare-protected emails

        Args:
            text: Text to extract emails from
            unique: If True, deduplicate by canonical form
            max_results: Maximum number of results

        Returns:
            List of EmailInfo objects
        """
        if not self._et:
            return []

        try:
            found = self._et.extract(text)

            results = []
            for email in found:
                info = EmailInfo(
                    original=email.original,
                    normalized=email.normalized,
                    canonical=email.canonical,
                    local=email.local,
                    domain=email.domain,
                    valid_syntax=email.valid_syntax,
                    valid_dns=email.deliverable_dns
                    if hasattr(email, "deliverable_dns")
                    else None,
                    disposable=email.domain_info.disposable
                    if email.domain_info
                    else False,
                )
                results.append(info)

                if max_results and len(results) >= max_results:
                    break

            return results
        except Exception as e:
            logger.error(f"Email extraction failed: {e}")
            return []

    def check_domain(self, domain: str) -> Optional[DomainHealth]:
        """
        Check domain health for email deliverability.

        Args:
            domain: Domain to check

        Returns:
            DomainHealth with MX/A records info
        """
        if not self._et:
            return None

        try:
            info = self._et.domain_health(domain)
            return DomainHealth(
                domain=info.ascii_domain,
                has_mx=info.has_mx,
                has_a=info.has_a,
                mx_hosts=list(info.mx_hosts) if info.mx_hosts else [],
                a_hosts=list(info.a_hosts) if info.a_hosts else [],
                disposable=info.disposable,
            )
        except Exception as e:
            logger.error(f"Domain health check failed: {e}")
            return None

    def is_disposable(self, email: str) -> bool:
        """
        Check if email is from a disposable/temporary email provider.

        Args:
            email: Email address

        Returns:
            True if disposable, False otherwise
        """
        parsed = self.parse(email)
        return parsed.disposable if parsed else False

    def validate_list(
        self,
        emails: List[str],
        check_dns: bool = False,
    ) -> Dict[str, bool]:
        """
        Validate a list of email addresses.

        Args:
            emails: List of emails to validate
            check_dns: If True, also check DNS deliverability

        Returns:
            Dict mapping email to validity
        """
        results = {}
        for email in emails:
            if check_dns:
                parsed = self.parse(email)
                results[email] = (
                    parsed.valid_dns
                    if parsed and parsed.valid_dns is not None
                    else False
                )
            else:
                results[email] = self.is_valid(email)
        return results

    def summary(self, text: str) -> Dict[str, Any]:
        """
        Generate summary of emails found in text.

        Args:
            text: Text to analyze

        Returns:
            Summary with counts and lists
        """
        emails = self.extract(text)

        valid = [e for e in emails if e.valid_syntax]
        disposable = [e for e in emails if e.disposable]
        domains = list({e.domain for e in emails})

        return {
            "total_found": len(emails),
            "valid_count": len(valid),
            "disposable_count": len(disposable),
            "unique_domains": len(domains),
            "domains": domains,
            "emails": [e.normalized for e in emails],
        }
