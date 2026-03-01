# ./ollamatoolkit/agent.py
"""
Ollama Toolkit - Agent Shim
===========================
Backward compatibility module.
Redirects 'from ollamatoolkit.agent import SimpleAgent' to 'ollamatoolkit.agents.simple'.
"""


# Trigger warning on import
# warnings.warn("ollamatoolkit.agent is deprecated. Use ollamatoolkit.agents.simple instead.", DeprecationWarning, stacklevel=2)
