# ./scripts/install-cursor-skills.ps1
& (Join-Path $PSScriptRoot 'install-agent-skills.ps1') -Agent cursor @args
