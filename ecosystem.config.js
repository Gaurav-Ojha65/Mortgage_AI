module.exports = {
  apps: [
    {
      name: "mortgage-backend",
      script: "uvicorn",
      // Pointing to your actual api file instead of main since we established it's api.py
      args: "api:app --host 0.0.0.0 --port 8001",
      cwd: "./backend",
      autorestart: true,
      max_restarts: 10,
      restart_delay: 3000, // 3 seconds
      watch: false
    }
  ]
};
