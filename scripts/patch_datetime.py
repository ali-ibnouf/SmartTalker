import os
import glob

directory = r"c:\Users\User\Documents\smart talker\SmartTalker\src\services\cost_guardian\*.py"

for filepath in glob.glob(directory):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace timezone-aware datetime with naive datetime for PostgreSQL
    # Except in places where it's being serialized to ISO format where it doesn't matter,
    # but replacing it everywhere is safest for consistency in asyncpg.
    new_content = content.replace(
        "datetime.now(timezone.utc)",
        "datetime.now(timezone.utc).replace(tzinfo=None)"
    )
    
    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Patched {filepath}")

print("Done patching cost_guardian files locally.")
