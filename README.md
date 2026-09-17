# Agent Second Brain

A personal AI agent in Telegram with Obsidian-based memory. You send
thoughts, tasks and notes by text or voice; the agent files them, keeps
context between conversations, reminds you of what matters and sends a daily
summary.

Primary documentation is in Russian: [README.ru.md](README.ru.md).

## What it does

- Accepts text, voice messages, photos, documents and forwards.
- Works as a full agent: searches notes, edits files, runs commands on the server.
- Keeps tasks, notes, projects and a goal cascade (3 years, year, month, week).
- Creates reminders and recurring jobs ("every weekday at 9:00 send my day plan").
- Processes the day's entries nightly at 21:00 or on `/process` and sends a report.
- Asks for confirmation before publishing, payments, irreversible deletion,
  granting access or messaging third parties.

User guide (Russian): [templates/vault/GUIDE.md](templates/vault/GUIDE.md).

## Architecture

```
Telegram bot
    │  text, voice (transcribed by Deepgram), photos, files
    ▼
Persistent agent session (Codex or Claude Code)
    │  reads and writes plain Markdown
    ▼
Obsidian vault (notes, tasks, goals, memory)
    │  automatic sync
    ▼
Private GitHub repository (memory backup)
```

Daily processing and health checks run on systemd timers; reminders and
recurring jobs are created by the agent through the `cron` skill.

## Two repositories

- **This code repository:** bot, agent instructions, skills, templates. No
  personal data; history starts from a clean commit.
- **Your private memory repository** (the vault, default name `dbrain-vault`):
  your notes, tasks, goals and agent memory. Created during setup, always
  private. In this code repository `vault/` is ignored except the shared
  agent instructions in `vault/.claude/`.

## Install

The only supported install guide is [docs/install.ru.md](docs/install.ru.md).

In short: a fresh Ubuntu 24.04 server, a dedicated user, then one start
command from the guide that installs `gh`, signs in to GitHub, clones this
private repository to `~/projects/agent-second-brain` and runs
`bash ~/projects/agent-second-brain/bootstrap.sh`. Setup asks for the
Telegram bot token and your Telegram ID, a Deepgram key, your timezone, the
engine (1 Codex, default; 2 Claude Code) and the permission profile (1 full
access on a dedicated server, default; 2 standard), and creates a private
GitHub repository for the vault. Then send `/onboarding` to the bot.

Setup is not unattended: it needs you three times on another device (GitHub
device code, pressing Start in your bot, the ChatGPT or Claude device login).
It reports success only after the agent answers a health-check question;
otherwise it prints what to do and is safe to re-run. There are no built-in
integrations with Google, Notion and similar services and no connection
wizard.

## Privacy

- Voice audio goes to Deepgram for transcription.
- Text goes to the AI provider you chose (OpenAI for Codex, Anthropic for Claude Code).
- Everything else stays on your server and in your private memory repository.
- Tokens and keys live only in `.env` on the server.

## Commands

Telegram: `/start`, `/help`, `/status`, `/process`, `/onboarding`, `/new`,
`/compact`, `/relogin`.

Server: `dbrain status | doctor | logs | restart | repair | login | onboarding | permissions`.

## Updating

```bash
git -C ~/projects/agent-second-brain pull --ff-only && bash ~/projects/agent-second-brain/upgrade.sh
```

Updates never touch your memory: the vault is a separate repository.

## License and credits

[MIT](LICENSE).

Based on the open-source project
[agent-second-brain](https://github.com/smixs/agent-second-brain) by
Serge Shima ([smixs](https://github.com/smixs)), including the
[autograph](https://github.com/smixs/autograph) memory engine.
