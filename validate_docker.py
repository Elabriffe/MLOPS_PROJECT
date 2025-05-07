import subprocess
import json
import os
import sys

exit_code = 0
valid_dockerfiles = []
invalid_dockerfiles = []
valid_compose_files = []
invalid_compose_files = []

# ---------------- Dockerfile Lint ----------------
print("🔎 Linting Dockerfiles...")
for root, _, files in os.walk("."):
    for filename in files:
        if filename.startswith("Dockerfile"):
            filepath = os.path.join(root, filename)
            print(f"📄 Checking {filepath}")
            try:
                result = subprocess.run(
                    ["hadolint", "--format", "json", filepath],
                    capture_output=True,
                    text=True,
                    check=False
                )
                if result.stdout.strip():
                    violations = json.loads(result.stdout)
                    errors = [v for v in violations if v.get("level") == "error"]
                    if errors:
                        print(f"❌ Errors in {filepath}:")
                        for err in errors:
                            print(f"  - Line {err['line']}: {err['message']}")
                        invalid_dockerfiles.append(filepath)
                        exit_code = 1
                    else:
                        print(f"✅ {filepath} passed (only warnings or info)")
                        valid_dockerfiles.append(filepath)
                else:
                    print(f"✅ {filepath} passed (no issues)")
                    valid_dockerfiles.append(filepath)
            except Exception as e:
                print(f"❌ Failed to lint {filepath}: {e}")
                invalid_dockerfiles.append(filepath)
                exit_code = 1

# ---------------- docker-compose Validation ----------------
print("\n🔎 Validating docker-compose files...")
for root, _, files in os.walk("."):
    for filename in files:
        if filename.startswith("docker-compose") and filename.endswith((".yml", ".yaml")):
            filepath = os.path.join(root, filename)
            print(f"📄 Validating {filepath}")
            result = subprocess.run(
                ["docker-compose", "-f", filepath, "config"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"❌ Error in {filepath}:")
                print(result.stderr or result.stdout)
                invalid_compose_files.append(filepath)
                exit_code = 1
            else:
                print(f"✅ {filepath} is valid")
                valid_compose_files.append(filepath)

# ---------------- Résumé ----------------
print("\n📦 Summary")
print("----------")

if valid_dockerfiles:
    print("✅ Valid Dockerfiles:")
    for f in valid_dockerfiles:
        print(f"  - {f}")
if invalid_dockerfiles:
    print("❌ Invalid Dockerfiles:")
    for f in invalid_dockerfiles:
        print(f"  - {f}")

if valid_compose_files:
    print("✅ Valid docker-compose files:")
    for f in valid_compose_files:
        print(f"  - {f}")
if invalid_compose_files:
    print("❌ Invalid docker-compose files:")
    for f in invalid_compose_files:
        print(f"  - {f}")

sys.exit(exit_code)
