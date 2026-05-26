from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.redis_client import redis_client


print(redis_client.ping())

redis_client.set("myagent:test", "hello redis")
value = redis_client.get("myagent:test")

print(value)
