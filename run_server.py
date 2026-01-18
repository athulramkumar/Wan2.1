#!/usr/bin/env python3
"""
Wan2.1 Video Generation API Server

Entry point for running the production API server.

Usage:
    python run_server.py
    python run_server.py --host 0.0.0.0 --port 8888
    python run_server.py --port 8888 --kill-existing
    
Environment variables:
    WAN_HOST: Server host (default: 0.0.0.0)
    WAN_PORT: Server port (default: 8888)
    RUNPOD_POD_ID: RunPod pod ID for generating public URL
"""

import argparse
import os
import signal
import subprocess
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def get_process_on_port(port: int) -> dict | None:
    """
    Check if a process is running on the specified port.
    
    Returns:
        dict with pid, name, cmdline if process found, None otherwise
    """
    try:
        # Use ss to find process on port
        result = subprocess.run(
            ["ss", "-tlnp", f"sport = :{port}"],
            capture_output=True,
            text=True
        )
        
        output = result.stdout
        
        # Parse the output to find PID
        # Example: LISTEN 0 128 0.0.0.0:8888 0.0.0.0:* users:(("python",pid=12345,fd=6))
        for line in output.split('\n'):
            if f":{port}" in line and "users:" in line:
                # Extract PID from users:(("name",pid=XXXXX,fd=Y))
                import re
                match = re.search(r'users:\(\("([^"]+)",pid=(\d+)', line)
                if match:
                    name = match.group(1)
                    pid = int(match.group(2))
                    
                    # Get command line
                    try:
                        with open(f"/proc/{pid}/cmdline", "r") as f:
                            cmdline = f.read().replace('\x00', ' ').strip()
                    except:
                        cmdline = "unknown"
                    
                    return {
                        "pid": pid,
                        "name": name,
                        "cmdline": cmdline[:100]  # Truncate long commands
                    }
        
        return None
        
    except Exception as e:
        print(f"Warning: Could not check port {port}: {e}")
        return None


def kill_process(pid: int) -> bool:
    """Kill a process by PID."""
    try:
        os.kill(pid, signal.SIGTERM)
        
        # Wait a moment and check if it's dead
        import time
        time.sleep(2)
        
        try:
            os.kill(pid, 0)  # Check if still alive
            # Still alive, force kill
            os.kill(pid, signal.SIGKILL)
            time.sleep(1)
        except ProcessLookupError:
            pass  # Process is dead
        
        return True
    except Exception as e:
        print(f"Error killing process {pid}: {e}")
        return False


def get_public_url(port: int) -> str | None:
    """Generate RunPod public URL if running on RunPod."""
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if pod_id:
        return f"https://{pod_id}-{port}.proxy.runpod.net"
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Wan2.1 Video Generation API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_server.py                          # Start on default port 8888
  python run_server.py --port 8000              # Start on port 8000
  python run_server.py --port 8888 --kill-existing   # Kill existing process on port
  python run_server.py --port 8888 -y           # Auto-confirm killing existing process
        """
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("WAN_HOST", "0.0.0.0"),
        help="Server host address (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("WAN_PORT", "8888")),
        help="Server port (default: 8888)"
    )
    
    parser.add_argument(
        "--kill-existing", "-k",
        action="store_true",
        help="Kill existing process on port (will ask for confirmation)"
    )
    
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Auto-confirm killing existing process (use with --kill-existing)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development only)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (keep at 1 for GPU)"
    )
    
    args = parser.parse_args()
    
    # Check if something is running on the port
    existing = get_process_on_port(args.port)
    
    if existing:
        print(f"\n⚠️  Port {args.port} is already in use!")
        print(f"   Process: {existing['name']} (PID: {existing['pid']})")
        print(f"   Command: {existing['cmdline']}")
        print()
        
        if args.kill_existing:
            if args.yes:
                confirm = "y"
            else:
                confirm = input(f"Kill this process and start server? [y/N]: ").strip().lower()
            
            if confirm == "y":
                print(f"Killing process {existing['pid']}...")
                if kill_process(existing['pid']):
                    print("✓ Process killed")
                else:
                    print("✗ Failed to kill process")
                    sys.exit(1)
            else:
                print("Aborted.")
                sys.exit(0)
        else:
            print("Use --kill-existing (-k) to kill the existing process")
            print(f"Example: python run_server.py --port {args.port} --kill-existing")
            sys.exit(1)
    
    # Generate public URL
    public_url = get_public_url(args.port)
    
    # Print banner
    print()
    print("=" * 60)
    print("  WAN2.1 VIDEO GENERATION API SERVER")
    print("=" * 60)
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  Workers: {args.workers}")
    print(f"  Reload: {args.reload}")
    print("=" * 60)
    print()
    print("  Local Endpoints:")
    print(f"    POST http://{args.host}:{args.port}/generate")
    print(f"    GET  http://{args.host}:{args.port}/status/{{job_id}}")
    print(f"    GET  http://{args.host}:{args.port}/video/{{job_id}}")
    print(f"    GET  http://{args.host}:{args.port}/health")
    print(f"    GET  http://{args.host}:{args.port}/config")
    print(f"    GET  http://{args.host}:{args.port}/docs  (API docs)")
    
    if public_url:
        print()
        print("  🌐 Public URL (RunPod):")
        print(f"    {public_url}/health")
        print(f"    {public_url}/docs")
        print(f"    {public_url}/generate")
    
    print()
    print("=" * 60)
    print()
    
    # Import and run uvicorn
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
