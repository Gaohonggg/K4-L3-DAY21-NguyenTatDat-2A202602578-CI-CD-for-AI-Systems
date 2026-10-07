"""Open runner-only SSH ingress and remove the exact rule created by this run."""

import argparse
import ipaddress
import os
import re
from urllib.request import urlopen


def _validate_group(group_id: str) -> None:
    if not re.fullmatch(r"sg-(?:[0-9a-f]{8}|[0-9a-f]{17})", group_id):
        raise ValueError("SERVER_SECURITY_GROUP_ID must be a valid security group ID")


def open_ssh(group_id: str, cidr: str, client) -> str:
    _validate_group(group_id)
    network = ipaddress.ip_network(cidr, strict=True)
    if network.version != 4 or network.prefixlen != 32 or not network.network_address.is_global:
        raise ValueError("Only a single public IPv4 /32 is allowed")
    response = client.authorize_security_group_ingress(
        GroupId=group_id,
        IpPermissions=[{
            "IpProtocol": "tcp", "FromPort": 22, "ToPort": 22,
            "IpRanges": [{"CidrIp": str(network), "Description": "Income lab GitHub Actions runner"}],
        }],
    )
    rules = response.get("SecurityGroupRules", [])
    if len(rules) != 1 or not re.fullmatch(r"sgr-(?:[0-9a-f]{8}|[0-9a-f]{17})", rules[0].get("SecurityGroupRuleId", "")):
        raise RuntimeError("AWS did not return the created SSH rule ID; inspect the security group")
    return rules[0]["SecurityGroupRuleId"]


def close_ssh(group_id: str, rule_id: str, client) -> None:
    _validate_group(group_id)
    if not re.fullmatch(r"sgr-(?:[0-9a-f]{8}|[0-9a-f]{17})", rule_id):
        raise ValueError("Invalid security group rule ID")
    client.revoke_security_group_ingress(GroupId=group_id, SecurityGroupRuleIds=[rule_id])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["open", "close"])
    parser.add_argument("--group-id", default=os.environ.get("SERVER_SECURITY_GROUP_ID", ""))
    parser.add_argument("--rule-id", default=os.environ.get("SSH_RULE_ID", ""))
    args = parser.parse_args()
    _validate_group(args.group_id)
    import boto3
    client = boto3.client("ec2")
    if args.command == "open":
        output_path = os.environ["GITHUB_OUTPUT"]
        with urlopen("https://checkip.amazonaws.com", timeout=10) as response:
            address = response.read(128).decode("ascii").strip()
        # Check the output file before granting access, then persist immediately.
        with open(output_path, "a", encoding="utf-8") as stream:
            rule_id = open_ssh(args.group_id, f"{address}/32", client)
            try:
                stream.write(f"rule_id={rule_id}\n")
                stream.flush()
            except OSError:
                close_ssh(args.group_id, rule_id, client)
                raise
        print(f"Opened SSH for runner {address}/32: {rule_id}")
    else:
        close_ssh(args.group_id, args.rule_id, client)
        print(f"Removed runner SSH rule: {args.rule_id}")


if __name__ == "__main__":
    main()
