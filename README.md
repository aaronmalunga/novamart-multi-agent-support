# NovaMart Multi-Agent Customer Support

Multi-agent customer support system on AWS Bedrock AgentCore.

A five-agent Orchestrator → Workers system for the NovaMart e-commerce support desk, built with the Strands Agents SDK and deployed to Amazon Bedrock AgentCore Runtime. An orchestrator routes each request to Inventory, Refund, Policy and Communication agents; the Policy agent runs parallel RAG over three Bedrock Knowledge Bases (returns, shipping, warranty). Bedrock Guardrails, AgentCore Memory and CloudWatch / X-Ray observability complete the setup.

Udacity project: Enterprise Multi-Agent Architecture with Amazon Bedrock AgentCore.

## Key files

- `src/agent_orchestrator.py` — completed implementation (Tasks 2, 3, 4 and 6)
- `.env` — populated Knowledge Base IDs, AgentCore Runtime ARN and guardrail ID/version
- `screenshots/test-score-120-of-120.png` — `python tests/test_agent.py all` result
- `screenshots/xray-trace-map.png` — X-Ray trace map (NovaMart-Orchestrator → worker agents → Knowledge Bases)
- `screenshots/xray-trace-map-list-view.png` — the same trace map as a node list

All other files are the Udacity starter, unchanged, apart from one helper, `scripts/sync_env.py`, which fills `.env` from the AWS account.
