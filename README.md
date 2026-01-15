# Game Backlog Concierge

A lightweight Streamlit app that recommends games using Steam metadata (tags/genres), plus platform and multiplayer constraints, with an optional “time-to-beat” length preference (playtime proxy).

**Live app:** https://gamebacklogconcierge-rvkaxrcssczuj3jsudu3ug.streamlit.app/  
**Repo:** https://github.com/brunohawkins/game_backlog_concierge

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
