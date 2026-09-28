import os
import time
from datetime import datetime, timezone

import boto3

COUNTED = {"PreSignUp_SignUp", "PreSignUp_ExternalProvider"}
_table = {}


def table():
    if "client" not in _table:
        _table["client"] = boto3.client("dynamodb")
    return _table["client"]


def admit(client, name: str, day: str, daily: int, total: int, expires: int) -> str | None:
    for sk, limit, message in (
        (f"day#{day}", daily, "Sign ups are full for today. Try again tomorrow."),
        ("total", total, "Sign ups are closed for now."),
    ):
        try:
            client.update_item(
                TableName=name,
                Key={"pk": {"S": "signups"}, "sk": {"S": sk}},
                UpdateExpression="ADD used :one SET expires = :expires",
                ConditionExpression="attribute_not_exists(used) OR used < :limit",
                ExpressionAttributeValues={":one": {"N": "1"}, ":limit": {"N": str(limit)}, ":expires": {"N": str(expires if sk != "total" else 4102444800)}},
            )
        except client.exceptions.ConditionalCheckFailedException:
            return message
    return None


def handler(event, context):
    if event.get("triggerSource") not in COUNTED:
        return event
    now = time.time()
    refused = admit(
        table(),
        os.environ["ACCOUNTS_TABLE"],
        datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d"),
        int(os.environ["DAILY_SIGNUPS"]),
        int(os.environ["TOTAL_SIGNUPS"]),
        int(now) + 3 * 86400,
    )
    if refused:
        raise Exception(refused)
    return event
