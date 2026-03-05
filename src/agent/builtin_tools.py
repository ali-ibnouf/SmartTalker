"""Built-in tool definitions for the SmartTalker Agent Engine.

Each tool is a dict matching the ToolRegistry schema with:
tool_id, name, description, category, owner, input_schema,
requires_confirmation.

These tools are registered automatically when the agent starts
and are available to all employees.
"""

from __future__ import annotations

import json


# =============================================================================
# Built-in Tool Definitions
# =============================================================================

BUILTIN_TOOLS: list[dict] = [
    {
        "tool_id": "send_email",
        "name": "Send Email",
        "description": (
            "Send an email to a specified recipient. Use this when a visitor "
            "requests information to be sent via email, or when follow-up "
            "communication is needed after the conversation."
        ),
        "category": "built_in",
        "owner": "maskki",
        "input_schema": json.dumps({
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient email address",
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject line",
                },
                "body": {
                    "type": "string",
                    "description": "Email body content (plain text or HTML)",
                },
                "cc": {
                    "type": "string",
                    "description": "Optional CC email address",
                },
            },
            "required": ["to", "subject", "body"],
        }),
        "requires_confirmation": True,
    },
    {
        "tool_id": "create_ticket",
        "name": "Create Support Ticket",
        "description": (
            "Create a support ticket for issues that need human follow-up. "
            "Use this when a visitor reports a problem, files a complaint, "
            "or requests something that requires manual processing."
        ),
        "category": "built_in",
        "owner": "maskki",
        "input_schema": json.dumps({
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Brief ticket title summarizing the issue",
                },
                "description": {
                    "type": "string",
                    "description": "Detailed description of the issue or request",
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "urgent"],
                    "description": "Ticket priority level",
                },
                "category": {
                    "type": "string",
                    "description": "Ticket category (e.g. billing, technical, general)",
                },
            },
            "required": ["title", "description", "priority"],
        }),
        "requires_confirmation": False,
    },
    {
        "tool_id": "schedule_callback",
        "name": "Schedule Callback",
        "description": (
            "Schedule a callback for a visitor at a specified date and time. "
            "Use this when a visitor wants to be contacted later, or when "
            "an issue requires follow-up at a specific time."
        ),
        "category": "built_in",
        "owner": "maskki",
        "input_schema": json.dumps({
            "type": "object",
            "properties": {
                "phone": {
                    "type": "string",
                    "description": "Phone number to call back",
                },
                "preferred_time": {
                    "type": "string",
                    "description": "Preferred callback time in ISO 8601 format",
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for the callback",
                },
                "notes": {
                    "type": "string",
                    "description": "Additional notes for the callback agent",
                },
            },
            "required": ["phone", "preferred_time", "reason"],
        }),
        "requires_confirmation": True,
    },
    {
        "tool_id": "search_knowledge",
        "name": "Search Knowledge Base",
        "description": (
            "Search the employee's knowledge base for relevant Q&A entries. "
            "Use this to find specific information, product details, policies, "
            "or procedures that can help answer the visitor's question."
        ),
        "category": "built_in",
        "owner": "maskki",
        "input_schema": json.dumps({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to find relevant knowledge entries",
                },
                "category": {
                    "type": "string",
                    "description": "Optional category to filter results",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 5)",
                },
            },
            "required": ["query"],
        }),
        "requires_confirmation": False,
    },
    {
        "tool_id": "transfer_to_human",
        "name": "Transfer to Human Operator",
        "description": (
            "Transfer the conversation to a human operator. Use this when "
            "the visitor explicitly requests to speak with a human, when the "
            "issue is too complex to handle, or when company policy requires "
            "human intervention."
        ),
        "category": "built_in",
        "owner": "maskki",
        "input_schema": json.dumps({
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Reason for transferring to a human operator",
                },
                "department": {
                    "type": "string",
                    "description": "Target department (e.g. sales, support, billing)",
                },
                "summary": {
                    "type": "string",
                    "description": "Brief conversation summary for the operator",
                },
            },
            "required": ["reason"],
        }),
        "requires_confirmation": False,
    },
    {
        "tool_id": "collect_visitor_info",
        "name": "Collect Visitor Information",
        "description": (
            "Store or update visitor contact information and preferences. "
            "Use this when a visitor provides their name, email, phone number, "
            "or other personal details during the conversation."
        ),
        "category": "built_in",
        "owner": "maskki",
        "input_schema": json.dumps({
            "type": "object",
            "properties": {
                "display_name": {
                    "type": "string",
                    "description": "Visitor's display name",
                },
                "email": {
                    "type": "string",
                    "description": "Visitor's email address",
                },
                "phone": {
                    "type": "string",
                    "description": "Visitor's phone number",
                },
                "language": {
                    "type": "string",
                    "description": "Visitor's preferred language code",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags to add to the visitor profile",
                },
            },
        }),
        "requires_confirmation": False,
    },
]


def get_openai_tool_definitions(tools: list[dict]) -> list[dict]:
    """Convert built-in tool definitions to OpenAI function-calling format.

    Args:
        tools: List of tool definition dicts from BUILTIN_TOOLS.

    Returns:
        List of dicts in OpenAI tools format for the chat completions API.
    """
    openai_tools = []
    for tool in tools:
        schema = tool["input_schema"]
        if isinstance(schema, str):
            schema = json.loads(schema)

        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool["tool_id"],
                "description": tool["description"],
                "parameters": schema,
            },
        })
    return openai_tools
