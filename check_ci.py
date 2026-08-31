import urllib.request
import json
import sys

# 1. Fetch latest run
req = urllib.request.Request("https://api.github.com/repos/Gaurav-Ojha65/Mortgage_AI/actions/runs?per_page=1")
req.add_header("Accept", "application/vnd.github.v3+json")
try:
    with urllib.request.urlopen(req) as response:
        runs_data = json.loads(response.read().decode())
    
    if not runs_data.get('workflow_runs'):
        print("No workflow runs found.")
        sys.exit(0)
    
    run = runs_data['workflow_runs'][0]
    print(f"GitHub Actions URL: {run['html_url']}")
    
    # 2. Fetch jobs for the run
    jobs_req = urllib.request.Request(run['jobs_url'])
    jobs_req.add_header("Accept", "application/vnd.github.v3+json")
    with urllib.request.urlopen(jobs_req) as response:
        jobs_data = json.loads(response.read().decode())
    
    for job in jobs_data['jobs']:
        print(f"Job: {job['name']} | Status: {job['status']} | Conclusion: {job['conclusion']}")

except Exception as e:
    print(f"Error: {e}")
