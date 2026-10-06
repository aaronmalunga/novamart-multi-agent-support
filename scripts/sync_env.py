"""
scripts/sync_env.py
===================
Fill .env from what actually exists in the AWS account, so no IDs or ARNs
have to be copied by hand:

  RETURNS_KB_ID / SHIPPING_KB_ID / WARRANTY_KB_ID  <- Knowledge Bases found by name
  GUARDRAIL_ID / GUARDRAIL_VERSION                 <- guardrail found by name, latest numbered version
  AGENTCORE_RUNTIME_ARN                            <- runtime recorded by the AgentCore CLI

Run from the project root:   python scripts/sync_env.py
Safe to re-run: it only rewrites these keys and leaves the rest of .env alone.
"""
import os
import sys

import boto3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'src')]
os.chdir(ROOT)

import config  # noqa: E402

ENV_PATH = os.path.join(ROOT, '.env')
REGION = config.AWS_REGION


def knowledge_base_ids() -> dict:
    client = boto3.client('bedrock-agent', region_name=REGION)
    by_name = {}
    for page in client.get_paginator('list_knowledge_bases').paginate():
        for kb in page['knowledgeBaseSummaries']:
            by_name[kb['name']] = kb['knowledgeBaseId']
    return {f"{d.upper()}_KB_ID": by_name.get(f"novamart-{d}-policy-kb", '')
            for d in ('returns', 'shipping', 'warranty')}


def guardrail() -> dict:
    client = boto3.client('bedrock', region_name=REGION)
    gid = next((g['id'] for g in client.list_guardrails().get('guardrails', [])
                if g['name'] == config.GUARDRAIL_NAME), '')
    if not gid:
        return {}
    versions = [g['version'] for g in client.list_guardrails(guardrailIdentifier=gid).get('guardrails', [])
                if g.get('version', 'DRAFT').isdigit()]
    return {'GUARDRAIL_ID': gid,
            'GUARDRAIL_VERSION': str(max(map(int, versions))) if versions else ''}


def runtime() -> dict:
    import agentcore_cli
    arn = agentcore_cli.deployed_runtime_arn()
    return {'AGENTCORE_RUNTIME_ARN': arn} if arn else {}


def write_env(values: dict) -> None:
    if not os.path.exists(ENV_PATH):
        with open(os.path.join(ROOT, '.env.example'), encoding='utf-8') as src:
            open(ENV_PATH, 'w', encoding='utf-8').write(src.read())
    lines = open(ENV_PATH, encoding='utf-8').read().splitlines()
    seen = set()
    for i, line in enumerate(lines):
        key = line.split('=', 1)[0].strip()
        if key in values and not line.lstrip().startswith('#'):
            lines[i] = f"{key}={values[key]}"
            seen.add(key)
    lines += [f"{k}={v}" for k, v in values.items() if k not in seen]
    open(ENV_PATH, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    found = {}
    for label, fn in (('Knowledge Bases', knowledge_base_ids), ('Guardrail', guardrail),
                      ('Runtime', runtime)):
        try:
            found.update({k: v for k, v in fn().items() if v})
        except Exception as exc:                                  # noqa: BLE001
            print(f"  [skip] {label}: {exc}")
    write_env(found)
    print("Updated .env:")
    for key in ('RETURNS_KB_ID', 'SHIPPING_KB_ID', 'WARRANTY_KB_ID', 'GUARDRAIL_ID',
                'GUARDRAIL_VERSION', 'AGENTCORE_RUNTIME_ARN'):
        print(f"  {key:<22} = {found.get(key, '(not found yet)')}")
