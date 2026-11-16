from fastapi import APIRouter

router = APIRouter(prefix="/api")


def memory_mb():
    """How much memory this server process is using, read from Linux itself
    (/proc/self/status). VmRSS = memory used right now, VmHWM = the highest
    it has ever reached since the server started (the "peak").
    On Windows this file doesn't exist, so we return None there."""
    try:
        values = {}
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(("VmRSS:", "VmHWM:")):
                    key, amount_kb = line.split()[0].rstrip(":"), int(line.split()[1])
                    values[key] = round(amount_kb / 1024, 1)
        return {"current_mb": values.get("VmRSS"), "peak_mb": values.get("VmHWM"), "limit_mb": 512}
    except OSError:
        return None


@router.get("/health")
def health():
    # Render calls this to check the app is alive; the memory numbers are for us.
    return {"status": "ok", "memory": memory_mb()}
