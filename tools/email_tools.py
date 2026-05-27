"""
tools/email_tools.py
─────────────────────
IONOS email client tools for the AgeMem agent.

Provides tools to read, send, and reply to emails via IONOS IMAP/SMTP.
Uses only Python standard library (imaplib, smtplib, email).

Configuration (environment variables):
    IONOS_EMAIL      — IONOS email address (e.g. you@yourdomain.com)
    IONOS_PASSWORD   — IONOS email password
    IONOS_IMAP_HOST  — IMAP server host (default: imap.ionos.com)
    IONOS_IMAP_PORT  — IMAP server port (default: 993)
    IONOS_SMTP_HOST  — SMTP server host (default: smtp.ionos.com)
    IONOS_SMTP_PORT  — SMTP server port (default: 587)

Security:
    - TLS/SSL enforced for both IMAP and SMTP
    - Credentials loaded from environment only, never hardcoded
    - Results capped to prevent context explosion
"""

from __future__ import annotations

import imaplib
import smtplib
import email
from email.header import decode_header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ── Configuration from environment ─────────────────────────────────────────

EMAIL_ADDRESS = os.environ.get("IONOS_EMAIL", "info@kennydb.store")
EMAIL_PASSWORD = os.environ.get("IONOS_PASSWORD", "")
IMAP_HOST = os.environ.get("IONOS_IMAP_HOST", "imap.ionos.com")
IMAP_PORT = int(os.environ.get("IONOS_IMAP_PORT", "993"))
SMTP_HOST = os.environ.get("IONOS_SMTP_HOST", "smtp.ionos.com")
SMTP_PORT = int(os.environ.get("IONOS_SMTP_PORT", "587"))

# ── Limits ─────────────────────────────────────────────────────────────────

READ_EMAILS_DEFAULT_LIMIT = 20
READ_EMAILS_MAX_LIMIT = 50
EMAIL_BODY_MAX_CHARS = 5000       # Truncate per-email body for context safety
REPLY_BODY_MAX_CHARS = 5000       # Reasonable cap on generated reply
RESULTS_TOTAL_MAX_CHARS = 12000   # Total result cap for the tool output


# ── Helpers ─────────────────────────────────────────────────────────────────


def _decode_str(value) -> str:
    """Decode encoded email header strings to plain text."""
    if value is None:
        return ""
    parts = decode_header(value)
    decoded = []
    for part, charset in parts:
        if isinstance(part, bytes):
            decoded.append(part.decode(charset or "utf-8", errors="replace"))
        else:
            decoded.append(part)
    return "".join(decoded)


def _get_body(msg) -> str:
    """Extract the plain-text body from an email message."""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition", ""))
            if content_type == "text/plain" and "attachment" not in disposition:
                charset = part.get_content_charset() or "utf-8"
                return part.get_payload(decode=True).decode(charset, errors="replace")
    else:
        charset = msg.get_content_charset() or "utf-8"
        return msg.get_payload(decode=True).decode(charset, errors="replace")
    return ""


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, adding ellipsis if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [truncated]"


def _check_credentials() -> Optional[str]:
    """Verify credentials are configured. Returns error message or None."""
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        return (
            "Email credentials not configured. Set IONOS_EMAIL and "
            "IONOS_PASSWORD environment variables."
        )
    return None


# ── Tool implementations ───────────────────────────────────────────────────


def fetch_emails(folder: str = "INBOX", limit: int = 20) -> str:
    """
    Connect to IMAP and download recent emails.

    Args:
        folder: IMAP folder name (default "INBOX")
        limit: Maximum number of most recent emails to return (1-50)

    Returns:
        Formatted string with email list and bodies, capped to prevent
        context explosion. Each email includes: uid, subject, from, date, body.
    """
    creds_error = _check_credentials()
    if creds_error:
        return f"[EMAIL ERROR] {creds_error}"

    limit = max(1, min(limit, READ_EMAILS_MAX_LIMIT))
    emails = []

    try:
        with imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT) as imap:
            imap.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            imap.select(folder)

            status, message_ids = imap.search(None, "ALL")
            if status != "OK":
                return "[EMAIL ERROR] No emails found or search failed."

            ids = message_ids[0].split()
            recent_ids = ids[-limit:] if len(ids) >= limit else ids

            for uid in reversed(recent_ids):  # newest first
                status, data = imap.fetch(uid, "(RFC822)")
                if status != "OK":
                    continue

                raw_email = data[0][1]
                msg = email.message_from_bytes(raw_email)

                emails.append({
                    "uid": uid.decode(),
                    "subject": _decode_str(msg.get("Subject")),
                    "from": _decode_str(msg.get("From")),
                    "date": msg.get("Date") or "",
                    "message_id": msg.get("Message-ID") or "",
                    "body": _get_body(msg),
                })

            imap.logout()

    except imaplib.IMAP4.error as e:
        logger.error(f"IMAP error: {e}")
        return f"[EMAIL ERROR] IMAP authentication or connection failed. Check credentials and that IMAP is enabled in IONOS settings. Error: {e}"
    except Exception as e:
        logger.error(f"Email fetch error: {e}")
        return f"[EMAIL ERROR] {e}"

    if not emails:
        return f"[EMAIL] No emails found in '{folder}'."

    # Build result with cap
    lines = [f"[EMAILS] {len(emails)} recent email(s) in '{folder}':", "=" * 50]
    total_chars = len(lines[0]) + len(lines[1])

    for i, em in enumerate(emails, 1):
        header = (
            f"\n--- Email {i} | UID: {em['uid']} ---\n"
            f"From: {em['from']}\n"
            f"Date: {em['date']}\n"
            f"Subject: {em['subject']}\n"
        )
        body = _truncate(em["body"], EMAIL_BODY_MAX_CHARS)

        block = header + body
        if total_chars + len(block) > RESULTS_TOTAL_MAX_CHARS:
            remaining = len(emails) - i + 1
            lines.append(
                f"\n... [{remaining} more email(s) omitted. "
                f"Use fetch_email_by_uid with the UID to view a specific email, "
                f"or use a smaller limit to browse fewer emails in full.]"
            )
            break

        lines.append(block)
        total_chars += len(block)

    lines.append("\n" + "=" * 50)
    return "\n".join(lines)


def fetch_email_by_uid(folder: str = "INBOX", uid: str = "") -> str:
    """
    Fetch a single email by its UID (obtained from fetch_emails output).

    Unlike fetch_emails which returns batches, this targets exactly one
    email using its IMAP UID. Useful when the batch listing truncates
    some emails and you need to read a specific one.

    Args:
        folder: IMAP folder containing the email (default "INBOX")
        uid: UID of the email to fetch (from fetch_emails output)

    Returns:
        Formatted string with the full email body (capped at per-email limit).
    """
    creds_error = _check_credentials()
    if creds_error:
        return f"[EMAIL ERROR] {creds_error}"

    if not uid:
        return "[EMAIL ERROR] 'uid' is required. Use fetch_emails to list emails and get their UIDs."

    try:
        with imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT) as imap:
            imap.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            imap.select(folder)

            status, data = imap.fetch(uid.encode(), "(RFC822)")
            if status != "OK":
                return (
                    f"[EMAIL ERROR] Could not find email with UID '{uid}' "
                    f"in folder '{folder}'. Use fetch_emails to list available UIDs."
                )

            raw_email = data[0][1]
            msg = email.message_from_bytes(raw_email)

            imap.logout()

    except imaplib.IMAP4.error as e:
        logger.error(f"IMAP error fetching UID {uid}: {e}")
        return f"[EMAIL ERROR] IMAP fetch failed for UID '{uid}': {e}"
    except Exception as e:
        logger.error(f"Fetch UID error: {e}")
        return f"[EMAIL ERROR] {e}"

    header = (
        f"[EMAIL] UID: {uid}\n"
        f"From: {_decode_str(msg.get('From'))}\n"
        f"Date: {msg.get('Date') or ''}\n"
        f"Subject: {_decode_str(msg.get('Subject'))}\n"
        f"Message-ID: {msg.get('Message-ID') or ''}\n"
        + "=" * 50 + "\n"
    )
    body = _truncate(_get_body(msg), EMAIL_BODY_MAX_CHARS)
    return header + body


def list_folders() -> str:
    """
    List all available IMAP folders (mailboxes) on the server.

    Returns a list of folder names like INBOX, Sent, Drafts, Archive, etc.
    Use this to discover which folders are available before reading emails.

    Returns:
        Formatted string listing available folders.
    """
    creds_error = _check_credentials()
    if creds_error:
        return f"[EMAIL ERROR] {creds_error}"

    try:
        with imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT) as imap:
            imap.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            status, folder_data = imap.list()

            if status != "OK":
                return "[EMAIL ERROR] Could not list folders."

            imap.logout()

    except Exception as e:
        logger.error(f"List folders error: {e}")
        return f"[EMAIL ERROR] {e}"

    folders = []
    for line in folder_data:
        decoded = line.decode("utf-8", errors="replace")
        # IMAP LIST response: * LIST (flags) "/" folder_name
        # Folder name is the last quoted or space-separated token
        parts = decoded.split('"')
        if len(parts) >= 2:
            name = parts[-1].strip()
            if name:
                folders.append(name)

    if not folders:
        return "[EMAIL] No folders found."

    lines = [f"[EMAIL FOLDERS] {len(folders)} folder(s) available:"]
    for f in sorted(folders):
        lines.append(f"  - {f}")
    return "\n".join(lines)


def send_email(
    to: str,
    subject: str,
    body: str,
    custom_sender: str = None,
    from_name: str = None,
    from_email: str = None,
) -> str:
    """
    Send a new plain-text email via IONOS SMTP.

    Args:
        to: Recipient email address
        subject: Email subject line
        body: Plain-text email body
        custom_sender: Raw From header value (backward-compatible, e.g. '"Name" <email>')
        from_name: Display name for the sender (used with from_email)
        from_email: Email address for the sender (used with from_name)

    Returns:
        Success or error message.
    """
    creds_error = _check_credentials()
    if creds_error:
        return f"[EMAIL ERROR] {creds_error}"

    if not to or not subject:
        return "[EMAIL ERROR] 'to' and 'subject' are required."

    body = _truncate(body, REPLY_BODY_MAX_CHARS)

    # Build From header: custom_sender > from_name+from_email > default
    if custom_sender:
        from_header = custom_sender
    elif from_name and from_email:
        from_header = f'"{from_name}" <{from_email}>'
    elif from_email:
        from_header = from_email
    else:
        from_header = EMAIL_ADDRESS

    msg = MIMEMultipart()
    msg["From"] = from_header
    msg["To"] = to
    msg["Subject"] = subject
    if from_email:
        msg["Reply-To"] = from_email
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.sendmail(EMAIL_ADDRESS, to, msg.as_string())

        logger.info(f"Email sent to {to}")
        return f"[EMAIL SENT] Email sent successfully to {to} with subject '{subject}'."

    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"SMTP auth error: {e}")
        return f"[EMAIL ERROR] SMTP authentication failed. Check credentials. Error: {e}"
    except Exception as e:
        logger.error(f"SMTP send error: {e}")
        return f"[EMAIL ERROR] Failed to send: {e}"


def reply_to_email(
    folder: str,
    uid: str,
    body: str,
    from_name: str = None,
    from_email: str = None,
) -> str:
    """
    Reply to a specific email by its UID.

    Fetches the original email by UID, then sends a reply with proper
    In-Reply-To and References headers for email threading.

    Args:
        folder: IMAP folder containing the email (default "INBOX")
        uid: UID of the email to reply to (from fetch_emails output)
        body: Reply body text
        from_name: Display name for the sender (used with from_email)
        from_email: Email address for the sender (used with from_name)

    Returns:
        Success or error message.
    """
    creds_error = _check_credentials()
    if creds_error:
        return f"[EMAIL ERROR] {creds_error}"

    if not uid:
        return "[EMAIL ERROR] 'uid' is required. Use the UID from fetch_emails results."

    if not body:
        return "[EMAIL ERROR] 'body' is required for the reply."

    body = _truncate(body, REPLY_BODY_MAX_CHARS)

    # Fetch the original email to get threading headers and recipient
    try:
        with imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT) as imap:
            imap.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            imap.select(folder)

            status, data = imap.fetch(uid.encode(), "(RFC822)")
            if status != "OK":
                return f"[EMAIL ERROR] Could not find email with UID '{uid}'. Use fetch_emails to get valid UIDs."

            raw_email = data[0][1]
            original = email.message_from_bytes(raw_email)

            imap.logout()

    except Exception as e:
        logger.error(f"IMAP fetch for reply error: {e}")
        return f"[EMAIL ERROR] Failed to fetch original email: {e}"

    # Extract original details
    original_from = _decode_str(original.get("From"))
    original_subject = _decode_str(original.get("Subject"))
    original_message_id = original.get("Message-ID", "")
    original_body = _get_body(original)

    # Build reply subject
    subject = original_subject
    if not subject.lower().startswith("re:"):
        subject = "Re: " + subject

    # Build reply with quoting
    quoted = "\n".join(f"> {line}" for line in original_body.splitlines())
    full_body = f"{body}\n\n--- Original Message ---\n{quoted}"

    # Build From header: from_name+from_email > default
    if from_name and from_email:
        from_header = f'"{from_name}" <{from_email}>'
    elif from_email:
        from_header = from_email
    else:
        from_header = EMAIL_ADDRESS

    # Build MIME message
    msg = MIMEMultipart()
    msg["From"] = from_header
    msg["To"] = original_from
    msg["Subject"] = subject
    if from_email:
        msg["Reply-To"] = from_email

    if original_message_id:
        msg["In-Reply-To"] = original_message_id
        msg["References"] = original_message_id

    msg.attach(MIMEText(full_body, "plain"))

    # Send
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.sendmail(EMAIL_ADDRESS, original_from, msg.as_string())

        logger.info(f"Reply sent to {original_from}")
        return f"[EMAIL SENT] Reply sent to {original_from} (Re: {original_subject[:50]}{'...' if len(original_subject) > 50 else ''})"

    except Exception as e:
        logger.error(f"SMTP reply error: {e}")
        return f"[EMAIL ERROR] Failed to send reply: {e}"


# ── OpenAI Tool Definitions ────────────────────────────────────────────────


tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "read_emails",
            "description": (
                "Read recent emails from an IMAP folder. "
                "Returns a list of emails with their UID, sender, subject, date, and body. "
                "Use this when the user asks to check or read their email. "
                "Set folder='INBOX' for received mail, folder='Sent' for sent mail. "
                "Results are capped to prevent context overflow — if some emails are truncated, "
                "use fetch_email_by_uid to view a specific email in full. "
                "Each email's UID can be used to reply via reply_to_email."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "folder": {
                        "type": "string",
                        "description": "IMAP folder to read from (default 'INBOX'). Use 'Sent' for sent emails, or try list_email_folders to discover all available folders."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of recent emails to return (default 20, max 50). Use a smaller limit (e.g. 5) to avoid truncation and see more body text."
                    },
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_email_by_uid",
            "description": (
                "Fetch a single email by its IMAP UID. "
                "Use this when read_emails truncated some emails and the user wants to see "
                "a specific one in full, or when they ask to 'fetch the Nth email' from a previous listing. "
                "The UID comes from read_emails output (e.g. 'UID: 1011'). "
                "This fetches exactly one email with its full body (up to the per-email body cap)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "folder": {
                        "type": "string",
                        "description": "IMAP folder containing the email (default 'INBOX')."
                    },
                    "uid": {
                        "type": "string",
                        "description": "UID of the email to fetch, from read_emails output."
                    },
                },
                "required": ["uid"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_sent_emails",
            "description": (
                "Read recent emails from the Sent folder. "
                "Convenience wrapper — identical to read_emails(folder='Sent'). "
                "Use this when the user asks to see emails they sent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of recent sent emails to return (default 20, max 50)."
                    },
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_email_folders",
            "description": (
                "List all available IMAP folders (mailboxes) on the server. "
                "Returns folder names like INBOX, Sent, Drafts, Archive, etc. "
                "Use this when the user asks what folders are available, "
                "or before reading from a folder other than INBOX."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": (
                "Send a new plain-text email via IONOS SMTP. "
                "Use this when the user asks you to compose and send an email on their behalf. "
                "All three parameters (to, subject, body) are required. "
                "To customize the sender, provide from_name and from_email (e.g. from_name='John Doe', from_email='john@example.com')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email address."
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line."
                    },
                    "body": {
                        "type": "string",
                        "description": "Plain-text email body content."
                    },
                    "custom_sender": {
                        "type": "string",
                        "description": "A raw From header value for backward compatibility (e.g. '\"Name\" <email>')."
                    },
                    "from_name": {
                        "type": "string",
                        "description": "Display name for the sender (used together with from_email to build '\"Name\" <email>')."
                    },
                    "from_email": {
                        "type": "string",
                        "description": "Email address for the sender (used together with from_name to build '\"Name\" <email>')."
                    },
                },
                "required": ["to", "subject", "body"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "reply_to_email",
            "description": (
                "Reply to a specific email by its UID (obtained from read_emails). "
                "Fetches the original email, sets proper threading headers (In-Reply-To, References), "
                "and quotes the original message. Use this when the user wants to respond to a specific email. "
                "To customize the sender, provide from_name and from_email (e.g. from_name='John Doe', from_email='john@example.com')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "folder": {
                        "type": "string",
                        "description": "IMAP folder containing the email (default 'INBOX')."
                    },
                    "uid": {
                        "type": "string",
                        "description": "UID of the email to reply to. Get this from read_emails output."
                    },
                    "body": {
                        "type": "string",
                        "description": "Reply body text."
                    },
                    "from_name": {
                        "type": "string",
                        "description": "Display name for the sender (used together with from_email to build '\"Name\" <email>')."
                    },
                    "from_email": {
                        "type": "string",
                        "description": "Email address for the sender (used together with from_name to build '\"Name\" <email>')."
                    },
                },
                "required": ["uid", "body"]
            }
        }
    },
]
