// PM2 process manager config for the LIVE execution process only.
// Runs out of venv-live (neo_api_client + ML libs, no yfinance -- see
// requirements-live.txt for why it's a separate venv from the screener).
// The pre-market screener (src/premarket.py, main venv) must have already
// run and produced data/watchlist.json before this starts -- see
// scripts/deploy/crontab.txt, which chains the two together.
// Deploy with: pm2 start scripts/deploy/ecosystem.config.js
module.exports = {
  apps: [
    {
      name: "apex-trading-bot",
      cwd: "/home/ubuntu/apex-trading-bot",
      script: "venv-live/bin/python",
      args: "-m src.live_session",
      interpreter: "none",
      autorestart: true,
      max_restarts: 10,
      restart_delay: 5000,
      kill_timeout: 10000, // give SIGTERM time to flush open positions before SIGKILL
      env: {
        PYTHONUNBUFFERED: "1",
      },
      out_file: "logs/out.log",
      error_file: "logs/error.log",
      time: true,
    },
  ],
};
