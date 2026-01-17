#!/bin/bash
echo "Core: $(cloc --csv --quiet henri/agent.py henri/cli.py henri/messages.py henri/permissions.py henri/config.py | tail -1 | cut -d, -f5) | Providers: $(cloc --csv --quiet henri/providers | tail -1 | cut -d, -f5) | Tools: $(cloc --csv --quiet henri/tools | tail -1 | cut -d, -f5)"
