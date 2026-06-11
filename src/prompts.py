"""Prompt templates for embedding extraction."""

PROMPTEOL_TEMPLATE = 'This sentence : "{text}" means in one word:"'


def build_prompteol(text: str) -> str:
    """Wrap text with the PromptEOL template."""
    return PROMPTEOL_TEMPLATE.format(text=text.replace('"', "'"))
