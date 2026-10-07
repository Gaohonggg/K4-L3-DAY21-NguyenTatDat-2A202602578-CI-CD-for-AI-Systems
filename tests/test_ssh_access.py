from unittest.mock import Mock

import pytest

from src.ssh_access import close_ssh, open_ssh

GROUP = "sg-0123456789abcdef0"
RULE = "sgr-0123456789abcdef0"


def test_runner_access_opens_only_port_22_for_one_address():
    client = Mock()
    client.authorize_security_group_ingress.return_value = {"SecurityGroupRules": [{"SecurityGroupRuleId": RULE}]}
    assert open_ssh(GROUP, "8.8.8.8/32", client) == RULE
    permissions = client.authorize_security_group_ingress.call_args.kwargs["IpPermissions"]
    assert permissions[0]["FromPort"] == permissions[0]["ToPort"] == 22
    assert permissions[0]["IpProtocol"] == "tcp"
    assert permissions[0]["IpRanges"][0]["CidrIp"] == "8.8.8.8/32"


@pytest.mark.parametrize("cidr", ["0.0.0.0/0", "8.8.8.0/24", "127.0.0.1/32", "10.0.0.1/32", "::1/128"])
def test_broad_private_or_non_ipv4_access_is_rejected(cidr):
    client = Mock()
    with pytest.raises(ValueError):
        open_ssh(GROUP, cidr, client)
    client.authorize_security_group_ingress.assert_not_called()


def test_cleanup_revokes_only_the_created_rule():
    client = Mock()
    close_ssh(GROUP, RULE, client)
    client.revoke_security_group_ingress.assert_called_once_with(GroupId=GROUP, SecurityGroupRuleIds=[RULE])
