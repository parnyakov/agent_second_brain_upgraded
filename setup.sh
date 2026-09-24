#!/bin/bash
# =============================================================================
# Agent Second Brain: установка на сервер (v3)
# =============================================================================
# Запускается из уже склонированного чистого дистрибутива (bootstrap.sh
# делает это за вас). Скрипт можно запускать повторно: каждый шаг проверяет,
# сделан ли он, и пропускает готовое. Уже введённые токены берутся из .env.
#
# Тяжёлую часть (зависимости Python, systemd --user сервисы, первая проверка
# здоровья) выполняет upgrade.sh: один источник правды для установки и
# обновления.
#
#   bash ~/projects/agent-second-brain/bootstrap.sh
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$PROJECT_DIR/.env"
VAULT_DIR="$PROJECT_DIR/vault"
RUNTIME_DIR="${DBRAIN_RUNTIME_DIR:-$HOME/.dbrain}"
# The services run Claude Code with CLAUDE_CONFIG_DIR=$HOME/.claude. The login
# and first-run screens done here must land in that same directory: without
# it they are recorded in ~/.claude.json and the bot's session stops on the
# first-run screens again (found on a clean server).
# Fixed, not "keep yours": the units hardcode %h/.claude.
export CLAUDE_CONFIG_DIR="$HOME/.claude"
TOTAL_STEPS=14
# Test harness only: skips live HTTPS checks of Telegram/Deepgram.
OFFLINE_TEST="${DBRAIN_SETUP_OFFLINE_TEST:-0}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()    { echo -e "  $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn()    { echo -e "${YELLOW}[!]${NC} $1"; }
error()   { echo -e "${RED}[X]${NC} $1"; }
ask()     { echo -e "${YELLOW}?${NC} $1"; }
STEP=0
step() {
    STEP=$((STEP + 1))
    echo ""
    echo -e "${CYAN}${BOLD}Шаг $STEP/$TOTAL_STEPS. $1${NC}"
}
fail() {
    error "$1"
    [ -n "${2:-}" ] && echo "    Что сделать: $2"
    echo "    После исправления запустите установку ещё раз: bash $PROJECT_DIR/bootstrap.sh"
    echo "    Готовые шаги повторно выполняться не будут."
    exit 1
}
check_command() { command -v "$1" >/dev/null 2>&1; }

# ── .env helpers: values never go to argv, logs or the terminal ───────────
env_get() {
    [ -f "$ENV_FILE" ] || return 0
    grep -E "^$1=" "$ENV_FILE" | tail -1 | cut -d= -f2- || true
}
env_set() {
    local key="$1" value="$2" tmp
    umask 077
    [ -f "$ENV_FILE" ] || cp "$PROJECT_DIR/.env.example" "$ENV_FILE"
    tmp="$(mktemp "$ENV_FILE.XXXXXX")"
    KEY="$key" VALUE="$value" python3 - "$ENV_FILE" "$tmp" <<'PY'
import os, sys
key, value = os.environ["KEY"], os.environ["VALUE"]
src, dst = sys.argv[1], sys.argv[2]
lines, done = [], False
for line in open(src, encoding="utf-8").read().splitlines():
    if line.split("=", 1)[0] == key and not line.startswith("#"):
        if not done:
            lines.append(f"{key}={value}")
            done = True
        continue
    lines.append(line)
if not done:
    lines.append(f"{key}={value}")
open(dst, "w", encoding="utf-8").write("\n".join(lines) + "\n")
PY
    chmod 600 "$tmp"
    mv "$tmp" "$ENV_FILE"
}

validate_telegram_token() { [[ $1 =~ ^[0-9]{6,12}:[A-Za-z0-9_-]{30,}$ ]]; }
validate_telegram_id()    { [[ $1 =~ ^[0-9]{3,15}$ ]]; }
validate_deepgram_key()   { [[ $1 =~ ^[A-Za-z0-9]{20,}$ ]]; }

# Read a secret without echoing it to the screen. The prompt goes to stderr:
# stdout of this function is the value itself (captured by the caller).
read_secret() {
    local prompt="$1" value
    ask "$prompt" >&2
    if ! IFS= read -r -s value; then
        echo "" >&2
        fail "Ввод прерван." "Запустите установку снова."
    fi
    echo "" >&2
    printf '%s' "$value"
}

# Plain answer; end of input aborts instead of looping forever.
read_answer() {
    local __var="$1"
    if ! IFS= read -r "$__var"; then
        fail "Ввод прерван." "Запустите установку снова."
    fi
}

# =============================================================================
# Checks
# =============================================================================

check_user_and_os() {
    step "Проверка пользователя и системы"
    if [ "$EUID" -eq 0 ]; then
        fail "Установку нельзя запускать от root." \
             "Создайте пользователя и войдите им заново по SSH: docs/install.ru.md, раздел «5. Рабочий пользователь»."
    fi
    if [ ! -f "$PROJECT_DIR/pyproject.toml" ] || [ ! -d "$PROJECT_DIR/src/d_brain" ] || [ ! -d "$PROJECT_DIR/templates/vault" ]; then
        fail "setup.sh запущен не из папки дистрибутива." \
             "Склонируйте чистый дистрибутив и запустите bootstrap.sh из него (раздел «Одна команда запуска»)."
    fi
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        if [[ "${ID:-}" != "ubuntu" && "${ID:-}" != "debian" ]]; then
            warn "Установка проверена на Ubuntu 24.04 и Debian 12. У вас: ${ID:-unknown}. Продолжаем на ваш риск."
        fi
    fi
    if ! sudo -v; then
        fail "Не удалось получить права sudo." "Проверьте, что пользователь в группе sudo: groups"
    fi
    success "Пользователь $USER, папка $PROJECT_DIR"
}

install_system_deps() {
    step "Системные пакеты (git, curl, tmux, gh, python3)"
    local missing=()
    for cmd in git curl tmux gh python3; do check_command "$cmd" || missing+=("$cmd"); done
    if [ ${#missing[@]} -eq 0 ]; then
        success "Все пакеты уже установлены"
        return
    fi
    info "Устанавливаю: ${missing[*]}"
    sudo apt-get update -qq || fail "apt-get update завершился ошибкой." "Проверьте интернет на сервере: ping -c 3 github.com"
    sudo apt-get install -y -qq git curl ca-certificates tmux gh python3 \
        || fail "Не удалось установить системные пакеты." "Повторите через минуту; если не помогло: sudo apt-get install -y git curl tmux gh python3"
    success "Системные пакеты установлены"
}

install_uv() {
    step "Менеджер Python-пакетов uv"
    export PATH="$HOME/.local/bin:$PATH"
    if check_command uv; then
        success "uv уже установлен"
        return
    fi
    curl -LsSf https://astral.sh/uv/install.sh | sh || fail "Не удалось установить uv." "Проверьте доступ к astral.sh и повторите."
    grep -q 'HOME/.local/bin' "$HOME/.bashrc" 2>/dev/null || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    check_command uv || fail "uv установлен, но не найден в PATH." "Выполните: source ~/.bashrc"
    success "uv установлен"
}

install_nodejs() {
    step "Node.js 20 (нужен для Codex и Claude Code)"
    if check_command node && [ "$(node --version | cut -d'v' -f2 | cut -d'.' -f1)" -ge 18 ]; then
        success "Node.js $(node --version) уже установлен"
        return
    fi
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - || fail "Не удалось подключить репозиторий Node.js."
    sudo apt-get install -y -qq nodejs || fail "Не удалось установить Node.js."
    success "Node.js $(node --version) установлен"
}

# =============================================================================
# Configuration
# =============================================================================

configure_github() {
    step "Вход в GitHub (для приватной копии памяти)"
    if gh auth status -h github.com >/dev/null 2>&1; then
        success "GitHub уже подключён"
    else
        echo "  Вход нужен только для вашей приватной памяти (репозиторий dbrain-vault)."
        echo "  На первый вопрос GitHub (Y/n) нажмите Enter, это значит «да»."
        echo "  Затем GitHub покажет одноразовый код и ссылку github.com/login/device."
        echo "  Откройте ссылку на компьютере, введите код и подтвердите доступ."
        gh auth login -h github.com -p https -w || fail "Вход в GitHub не завершён." "Повторите установку и завершите вход по ссылке."
    fi
    # git push/pull через HTTPS берут учётные данные у gh; токен не попадает
    # в URL репозитория и в .git/config.
    gh auth setup-git -h github.com >/dev/null 2>&1 || true
    success "GitHub готов"
}

# Copy SRC into DST without overwriting anything that already exists.
copy_missing() {
    local src="$1" dst="$2" path
    while IFS= read -r -d '' path; do
        mkdir -p "$dst/$path" || return 1
    done < <(cd "$src" && find . -mindepth 1 -type d -print0)
    while IFS= read -r -d '' path; do
        [ -e "$dst/$path" ] || [ -L "$dst/$path" ] || cp -a "$src/$path" "$dst/$path" || return 1
    done < <(cd "$src" && find . -mindepth 1 ! -type d -print0)
}

initialize_vault() {
    step "Личная папка памяти (vault) из чистого шаблона"
    if [ ! -d "$PROJECT_DIR/templates/vault" ]; then
        fail "Шаблон templates/vault отсутствует; чужие данные не используем." "Склонируйте дистрибутив заново."
    fi
    mkdir -p "$VAULT_DIR"
    # Повторный запуск не перезаписывает ваши ответы и заметки: копируются
    # только отсутствующие файлы. Не `cp -n`: coreutils 9.2+ (Ubuntu 24.04)
    # печатает на нём предупреждение, похожее на ошибку.
    copy_missing "$PROJECT_DIR/templates/vault" "$VAULT_DIR" \
        || fail "Не удалось скопировать шаблон памяти в $VAULT_DIR." \
                "Проверьте место на диске (df -h) и права папки: ls -la $VAULT_DIR"
    chmod 700 "$VAULT_DIR"
    if [ ! -d "$VAULT_DIR/.git" ]; then
        git -C "$VAULT_DIR" init -q -b main
    fi
    git -C "$VAULT_DIR" config user.name "d-brain vault"
    git -C "$VAULT_DIR" config user.email "vault@localhost"
    success "Vault: $VAULT_DIR"
}

collect_tokens() {
    step "Telegram и Deepgram"
    local token user_id deepgram
    token="$(env_get TELEGRAM_BOT_TOKEN)"
    user_id="$(env_get ALLOWED_USER_IDS | tr -d '[] ' | cut -d, -f1)"
    deepgram="$(env_get DEEPGRAM_API_KEY)"

    if validate_telegram_token "$token"; then
        success "Токен Telegram-бота уже сохранён"
    else
        echo "  Токен выдаёт @BotFather после команды /newbot (инструкция, пункт 2.2 «Telegram-бот»)."
        while true; do
            token="$(read_secret "Вставьте токен бота (символы не отображаются) и нажмите Enter:")"
            validate_telegram_token "$token" && break
            error "Не похоже на токен. Он выглядит как 1234567890:AA... Скопируйте его целиком."
        done
        env_set TELEGRAM_BOT_TOKEN "$token"
    fi

    if validate_telegram_id "$user_id"; then
        success "Ваш Telegram ID уже сохранён"
    else
        echo "  Свой числовой ID узнайте у @userinfobot: откройте его и нажмите Start."
        while true; do
            ask "Ваш Telegram ID (только цифры):"
            read_answer user_id
            validate_telegram_id "$user_id" && break
            error "ID состоит только из цифр, например 123456789."
        done
        env_set ALLOWED_USER_IDS "[$user_id]"
    fi

    if validate_deepgram_key "$deepgram"; then
        success "Ключ Deepgram уже сохранён"
    else
        echo "  Ключ создаётся на console.deepgram.com → API Keys → Create a New API Key."
        while true; do
            deepgram="$(read_secret "Вставьте ключ Deepgram (символы не отображаются):")"
            validate_deepgram_key "$deepgram" && break
            error "Ключ Deepgram состоит из латинских букв и цифр, не короче 20 символов."
        done
        env_set DEEPGRAM_API_KEY "$deepgram"
    fi

    probe_services
}

probe_services() {
    if [ "$OFFLINE_TEST" = "1" ]; then
        warn "Тестовый режим: проверка Telegram/Deepgram по сети пропущена"
        return
    fi
    local result
    while true; do
        # Секреты передаются через окружение процесса, не через аргументы.
        set +e
        result="$(TELEGRAM_BOT_TOKEN="$(env_get TELEGRAM_BOT_TOKEN)" \
            DEEPGRAM_API_KEY="$(env_get DEEPGRAM_API_KEY)" \
            CHAT_ID="$(env_get ALLOWED_USER_IDS | tr -d '[] ' | cut -d, -f1)" \
            python3 "$PROJECT_DIR/scripts/setup_probe.py")"
        local code=$?
        set -e
        case "$code" in
            0) success "$result"; return ;;
            3) error "Telegram не принял токен."
               ask "Ввести токен заново? (Y/n)"; read_answer reply
               if [[ ! $reply =~ ^[Nn]$ ]]; then env_set TELEGRAM_BOT_TOKEN ""; collect_tokens; return; fi
               fail "Токен бота не работает." "Получите новый токен у @BotFather командой /token и запустите установку снова." ;;
            4) warn "$result"
               echo "  Откройте своего бота в Telegram, нажмите Start (или отправьте /start), затем вернитесь сюда."
               ask "Нажмите Enter, чтобы проверить ещё раз, или введите 'id', чтобы исправить Telegram ID:"
               read_answer reply
               if [ "$reply" = "id" ]; then env_set ALLOWED_USER_IDS ""; collect_tokens; return; fi ;;
            5) error "Deepgram не принял ключ."
               ask "Ввести ключ заново? (Y/n)"; read_answer reply
               if [[ ! $reply =~ ^[Nn]$ ]]; then env_set DEEPGRAM_API_KEY ""; collect_tokens; return; fi
               fail "Ключ Deepgram не работает." "Создайте новый ключ на console.deepgram.com и запустите установку снова." ;;
            *) fail "Не удалось связаться с Telegram или Deepgram ($result)." "Проверьте интернет на сервере: curl -I https://api.telegram.org" ;;
        esac
    done
}

choose_settings() {
    step "Часовой пояс и движок агента"
    local tz engine reply detected
    tz="$(env_get TZ)"
    detected="$(timedatectl show -p Timezone --value 2>/dev/null || true)"
    [ -z "$tz" ] || [ "$tz" = "UTC" ] && tz="${detected:-Europe/Moscow}"
    [ "$tz" = "Etc/UTC" ] && tz="Europe/Moscow"
    ask "Часовой пояс для напоминаний и отчётов [Enter = $tz]:"
    read_answer reply
    tz="${reply:-$tz}"
    if [ ! -e "/usr/share/zoneinfo/$tz" ]; then
        warn "Часовой пояс $tz не найден, использую Europe/Moscow. Позже можно поменять TZ в .env."
        tz="Europe/Moscow"
    fi
    env_set TZ "$tz"
    # Ежедневная обработка и проверка здоровья идут по времени сервера:
    # выставляем серверу тот же пояс, чтобы «21:00» значило 21:00 у вас.
    if [ "$(timedatectl show -p Timezone --value 2>/dev/null || true)" != "$tz" ]; then
        sudo timedatectl set-timezone "$tz" 2>/dev/null || warn "Не удалось сменить часовой пояс сервера; расписания будут по UTC."
    fi

    engine="$(env_get DBRAIN_CHAT_ENGINE)"
    engine="${engine:-codex}"
    echo "  Какой ИИ будет «мозгом» агента:"
    echo "    1 = Codex (подписка ChatGPT Plus/Pro)"
    echo "    2 = Claude Code (подписка Claude Pro/Max)"
    local default_choice=1
    [ "$engine" = "claude" ] && default_choice=2
    ask "Выбор [Enter = $default_choice]:"
    read_answer reply
    reply="${reply:-$default_choice}"
    if [ "$reply" = "2" ]; then engine="claude"; else engine="codex"; fi
    env_set DBRAIN_CHAT_ENGINE "$engine"
    env_set DBRAIN_CRON_ENGINE "$engine"
    env_set VAULT_PATH "./vault"
    AGENT_ENGINE="$engine"
    success "Часовой пояс $tz, движок $engine"
}

install_and_authorize_engine() {
    step "Установка и вход: $AGENT_ENGINE"
    if [ "$AGENT_ENGINE" = "claude" ]; then
        install_claude_cli
        authorize_claude
    else
        install_codex_cli
        authorize_codex
    fi
}

install_claude_cli() {
    if check_command claude; then
        success "Claude Code уже установлен"
        return
    fi
    sudo npm install -g @anthropic-ai/claude-code || fail "Не удалось установить Claude Code." "Повторите установку через минуту."
    success "Claude Code установлен"
}

install_codex_cli() {
    if check_command codex; then
        success "Codex уже установлен"
        return
    fi
    sudo npm install -g @openai/codex || fail "Не удалось установить Codex." "Повторите установку через минуту."
    success "Codex установлен"
}

claude_logged_in() {
    # JSON-поле loggedIn стабильнее, чем текст статуса. Вывод берём целиком:
    # `grep -q` в конвейере под pipefail может дать ложное «не вошли».
    local out
    out="$(claude auth status --json 2>/dev/null || true)"
    [[ "$out" =~ \"loggedIn\":\ *true ]]
}

# Экраны первого запуска пройдены в том же каталоге, что у сервисов. Иначе
# сессия бота остановится на них и агент не ответит.
claude_first_run_done() {
    python3 - "$CLAUDE_CONFIG_DIR/.claude.json" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1]))
    sys.exit(0 if isinstance(data, dict) and data.get("hasCompletedOnboarding") else 1)
except Exception:
    sys.exit(1)
PY
}

authorize_claude() {
    if claude_logged_in && claude_first_run_done; then
        success "Claude Code: вход уже выполнен"
        return
    fi
    if claude_logged_in; then
        echo "  Сейчас откроется Claude Code. Нажимайте Enter на приветственных экранах."
        echo "  Когда увидите поле ввода Claude, введите /exit."
    else
        echo "  Сейчас откроется Claude Code. Выберите вход через подписку, откройте ссылку на компьютере,"
        echo "  подтвердите вход и вставьте код обратно. Когда увидите приглашение Claude, введите /exit."
    fi
    ask "Нажмите Enter, чтобы начать вход:"
    read_answer _
    (cd "$PROJECT_DIR" && claude) || true
    if claude_logged_in && claude_first_run_done; then
        success "Claude Code: вход выполнен"
    else
        fail "Вход в Claude Code не выполнен." "Запустите установку снова, пройдите экраны Claude до поля ввода и введите /exit."
    fi
}

# `codex login status` reports on stderr; capture both streams in full
# (a pipe into `grep -q` could end early and trip pipefail).
codex_logged_in() {
    local out code=0
    out="$(codex login status 2>&1)" || code=$?
    out="${out,,}"
    [ "$code" -eq 0 ] && [[ "$out" == *"logged in"* ]] && [[ "$out" != *"not logged in"* ]]
}

authorize_codex() {
    if codex_logged_in; then
        success "Codex: вход уже выполнен"
        return
    fi
    echo "  Codex покажет ссылку и код. Откройте ссылку на компьютере, войдите в аккаунт ChatGPT и введите код."
    codex login --device-auth || true
    if codex_logged_in; then
        success "Codex: вход выполнен"
    else
        fail "Вход в Codex не выполнен (ссылка и код действуют 15 минут)." "Запустите установку снова и завершите вход на компьютере."
    fi
}

choose_permissions() {
    step "Права агента"
    local profile reply
    profile="$(env_get DBRAIN_PERMISSION_PROFILE)"
    profile="${profile:-max}"
    echo "  Агент работает от вашего пользователя $USER на этом сервере."
    echo "    1 = Полный доступ на выделенном сервере (рекомендуется): агент сам ставит пакеты и"
    echo "        настраивает сервер через sudo без вопросов. Используйте только если на сервере"
    echo "        нет ничего, кроме агента."
    echo "    2 = Стандартный: агент свободно работает в своей папке и расписаниях, но системные"
    echo "        команды (sudo) выполняете вы сами по его подсказке."
    local default_choice=1
    [ "$profile" = "standard" ] && default_choice=2
    ask "Выбор [Enter = $default_choice]:"
    read_answer reply
    reply="${reply:-$default_choice}"
    if [ "$reply" = "2" ]; then profile="standard"; else profile="max"; fi
    bash "$PROJECT_DIR/scripts/configure-permissions.sh" "$profile" \
        || fail "Не удалось применить права." "Запустите вручную: bash $PROJECT_DIR/scripts/configure-permissions.sh $profile"
}

# Nightly housekeeping is opt-in on purpose: it closes terminal windows, and
# a user who keeps long-running work in tmux must get to say no. "on" only
# arms the timer; every deletion rule inside the script stays conservative.
choose_cleanup() {
    step "Ночная уборка сервера"
    local current reply default_choice=1
    current="$(env_get DBRAIN_CLEANUP)"
    [ "$current" = "off" ] && default_choice=2
    echo "  Раз в сутки в 20:30 агент может закрывать заброшенные окна терминала"
    echo "  (7 суток без активности, ничего не запущено, никто не подключён) и чистить"
    echo "  кэши, которые восстанавливаются сами. Ваша память, история диалогов и"
    echo "  рабочие файлы не трогаются, отчёт приходит вам в Telegram."
    echo "    1 = включить (рекомендуется)"
    echo "    2 = не включать"
    ask "Выбор [Enter = $default_choice]:"
    read_answer reply
    reply="${reply:-$default_choice}"
    if [ "$reply" = "2" ]; then env_set DBRAIN_CLEANUP off; else env_set DBRAIN_CLEANUP on; fi
}

# Existing memory repository (new server, reinstall): a vault that only holds
# the fresh template commit is replaced by the saved memory; a vault with
# its own history is rebased onto it instead, never overwritten.
restore_vault_from_remote() {
    local name="$1"
    git fetch -q origin main 2>/dev/null || return 0
    git rev-parse -q --verify origin/main >/dev/null || return 0
    if [ "$(git rev-list --count HEAD 2>/dev/null || echo 0)" -le 1 ]; then
        git reset -q --hard origin/main
        git branch -q --set-upstream-to=origin/main 2>/dev/null || true
        success "Память восстановлена из $name"
    else
        git pull -q --rebase origin main \
            || fail "Не удалось объединить локальную память с $name." "Выполните: git -C $VAULT_DIR status и перешлите вывод агенту."
    fi
}

configure_vault_backup() {
    step "Приватная резервная копия памяти в GitHub"
    local name reply visibility
    cd "$VAULT_DIR"
    git add -A
    if git rev-parse -q --verify HEAD >/dev/null; then
        git commit -q -m "Update private vault (setup)" >/dev/null 2>&1 || true
    else
        git commit -q -m "Initial private vault" >/dev/null 2>&1 || true
    fi
    if git remote get-url origin >/dev/null 2>&1; then
        info "Репозиторий памяти уже подключён"
    else
        name="$(env_get DBRAIN_VAULT_REPO)"
        name="${name:-dbrain-vault}"
        ask "Имя приватного репозитория для памяти [Enter = $name]:"
        read_answer reply
        name="${reply:-$name}"
        if gh repo view "$name" >/dev/null 2>&1; then
            visibility="$(gh repo view "$name" --json visibility -q .visibility)"
            [ "$visibility" = "PRIVATE" ] || fail "Репозиторий $name существует и он не приватный; sync refused." \
                "Выберите другое имя или сделайте репозиторий приватным в настройках GitHub."
            git remote add origin "$(gh repo view "$name" --json url -q .url).git"
            restore_vault_from_remote "$name"
        else
            gh repo create "$name" --private --source=. --remote=origin \
                || fail "GitHub не создал репозиторий $name." "Проверьте вход: gh auth status"
        fi
        env_set DBRAIN_VAULT_REPO "$name"
    fi
    # Ночная копия и /process тоже проверяют PRIVATE перед каждой отправкой.
    env_set DBRAIN_REQUIRE_PRIVATE_VAULT 1
    # Fail closed: память уходит только в PRIVATE репозиторий без секретов в URL.
    VISIBILITY="$(gh repo view "$(git remote get-url origin)" --json visibility -q .visibility 2>/dev/null || echo UNKNOWN)"
    if [ "$VISIBILITY" != "PRIVATE" ]; then
        fail "Репозиторий памяти не PRIVATE ($VISIBILITY); sync refused." "Сделайте его приватным на github.com и повторите."
    fi
    if git remote get-url origin | grep -Eq '(github_pat_|ghp_|gho_|@github\.com)'; then
        fail "В адресе репозитория памяти найден секрет; sync refused." "Выполните: git -C $VAULT_DIR remote set-url origin https://github.com/ВАШ_ЛОГИН/$name.git"
    fi
    git push -q -u origin HEAD:main || fail "Не удалось отправить память в GitHub." "Проверьте: gh auth status; затем повторите установку."
    cd "$PROJECT_DIR"
    success "Память сохраняется в приватный репозиторий (ежедневно и после /process)"
}

run_upgrade() {
    step "Сервисы агента (systemd), зависимости и проверка здоровья"
    # Новая установка: служебное сообщение старых версий о смене кнопок
    # новому пользователю не нужно.
    if [ ! -f "$RUNTIME_DIR/setup.completed" ]; then
        mkdir -p "$RUNTIME_DIR"
        [ -e "$RUNTIME_DIR/keyboard_removed" ] || echo "fresh install" > "$RUNTIME_DIR/keyboard_removed"
    fi
    rm -f "$RUNTIME_DIR/install-doctor.rc"
    local rc=0
    bash "$PROJECT_DIR/upgrade.sh" || rc=$?
    # 3 = всё установлено, но осмотр красный: разбирается в итоговой проверке.
    if [ "$rc" = "3" ] && [ -f "$RUNTIME_DIR/install-doctor.rc" ]; then
        return 0
    fi
    [ "$rc" = "0" ] || fail "upgrade.sh завершился ошибкой." "Посмотрите сообщение выше; затем dbrain doctor"
}

final_check() {
    step "Итоговая проверка"
    local units_ok=1 doctor_rc
    for unit in dbrain-bot.service dbrain-watchdog.service; do
        if systemctl --user is-active --quiet "$unit" 2>/dev/null; then
            success "$unit работает"
        else
            error "$unit не работает"; units_ok=0
        fi
    done
    for timer in dbrain-process.timer dbrain-doctor.timer; do
        if systemctl --user is-enabled --quiet "$timer" 2>/dev/null; then
            success "расписание $timer включено"
        else
            error "расписание $timer не включено"; units_ok=0
        fi
    done
    if [ "$units_ok" != "1" ]; then
        fail "Сервисы агента не запустились." \
             "Выполните dbrain repair, затем dbrain logs 100 и перешлите вывод тому, кто помогает вам с установкой."
    fi
    doctor_rc="$(cat "$RUNTIME_DIR/install-doctor.rc" 2>/dev/null || echo none)"
    if [ "$doctor_rc" != "0" ]; then
        # Не «Установка завершена»: сервисы работают, но агент не отвечает.
        error "Сервисы запущены, но агент пока не отвечает: осмотр нашёл проблемы (строки ❌ выше)."
        echo "    Что сделать:"
        echo "      1. Если в строке canary «нет входа в подписку»: dbrain login"
        echo "         (откройте ссылку на компьютере и введите код)."
        echo "      2. Проверьте: dbrain doctor  → должно быть «Осмотр пройден»."
        echo "      3. Завершите установку: bash $PROJECT_DIR/bootstrap.sh"
        echo "    Введённые токены и память сохранены, готовые шаги не повторятся."
        exit 1
    fi
    success "Агент ответил на проверочный вопрос"
    mkdir -p "$RUNTIME_DIR"
    date -Iseconds > "$RUNTIME_DIR/setup.completed"
    success "Установка завершена"
    if [ "$OFFLINE_TEST" != "1" ]; then
        TELEGRAM_BOT_TOKEN="$(env_get TELEGRAM_BOT_TOKEN)" CHAT_ID="$(env_get ALLOWED_USER_IDS | tr -d '[] ' | cut -d, -f1)" \
            python3 "$PROJECT_DIR/scripts/setup_probe.py" --announce >/dev/null 2>&1 || true
    fi
}

print_outro() {
    echo ""
    echo -e "${GREEN}${BOLD}Готово. Что дальше:${NC}"
    echo "  1. Откройте Telegram, найдите своего бота и отправьте /onboarding"
    echo "     Агент проведёт знакомство: профиль, проекты, цели, заметки. Можно отвечать голосом,"
    echo "     прерываться и продолжать той же командой."
    echo "  2. Памятка о возможностях: файл vault/GUIDE.md и команда /help в боте."
    echo "  3. Альтернатива без Telegram: dbrain onboarding resume"
    echo ""
    echo "  Команды на сервере:"
    echo "    dbrain status       состояние"
    echo "    dbrain doctor       проверка и подсказки"
    echo "    dbrain logs         последние логи"
    echo "    dbrain repair       перезапуск при проблемах"
    echo "    dbrain permissions  права агента (status | max | standard)"
    echo ""
}

main() {
    echo -e "${BOLD}Agent Second Brain: установка${NC}"
    echo "  Займёт 10–20 минут. Можно прервать Ctrl+C и запустить снова: готовое не повторится."
    check_user_and_os
    install_system_deps
    install_uv
    install_nodejs
    configure_github
    initialize_vault
    collect_tokens
    choose_settings
    install_and_authorize_engine
    choose_permissions
    choose_cleanup
    configure_vault_backup
    run_upgrade
    final_check
    print_outro
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
