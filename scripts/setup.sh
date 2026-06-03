#!/usr/bin/env bash
# Konfiguruje srodowisko produkcyjne projektu RemoteDroids na Raspberry Pi / Linux.
#
# Uzycie:
#   bash scripts/setup.sh [opcje]
#
# Opcje:
#   --skip-install      Pomija instalacje zaleznosci pip
#   --recreate-venv     Usuwa i odtwarza virtualenv od zera
#   --install-service   Kopiuje i wlacza usluge systemd (wymaga sudo)
#   -h, --help          Wyswietla pomoc

set -euo pipefail

# ---------- kolory ----------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}$*${NC}"; }
success() { echo -e "${GREEN}$*${NC}"; }
warn()    { echo -e "${YELLOW}$*${NC}"; }
error()   { echo -e "${RED}BLAD: $*${NC}" >&2; exit 1; }

# ---------- parsowanie argumentow ----------
SKIP_INSTALL=false
RECREATE_VENV=false
INSTALL_SERVICE=false

for arg in "$@"; do
    case "$arg" in
        --skip-install)    SKIP_INSTALL=true ;;
        --recreate-venv)   RECREATE_VENV=true ;;
        --install-service) INSTALL_SERVICE=true ;;
        -h|--help)
            sed -n '2,15p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) error "Nieznana opcja: $arg" ;;
    esac
done

# ---------- katalog projektu ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
info "Katalog projektu: $PROJECT_ROOT"

# ---------- sprawdz Python ----------
PYTHON_CMD=""
for candidate in python3.13 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" --version 2>&1)
        if [[ "$ver" =~ Python\ 3\.([0-9]+) ]]; then
            minor="${BASH_REMATCH[1]}"
            if (( minor >= 13 )); then
                PYTHON_CMD="$candidate"
                success "Znaleziono $ver"
                break
            fi
        fi
    fi
done
[[ -n "$PYTHON_CMD" ]] || error "Nie znaleziono Pythona 3.13+. Zainstaluj Python i dodaj go do PATH."

# ---------- virtualenv ----------
VENV_DIR="$PROJECT_ROOT/.venv"

if $RECREATE_VENV && [[ -d "$VENV_DIR" ]]; then
    warn "Usuwam istniejacy .venv..."
    rm -rf "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
    info "Tworzenie virtualenv w .venv..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
else
    success "Virtualenv juz istnieje (.venv)."
fi

PIP="$VENV_DIR/bin/pip"
PYTHON="$VENV_DIR/bin/python"

# ---------- instalacja zaleznosci ----------
if ! $SKIP_INSTALL; then
    info "Aktualizacja pip..."
    "$PYTHON" -m pip install --upgrade pip

    info "Instalacja zaleznosci produkcyjnych (requirements-prod.txt)..."
    "$PIP" install -r "$PROJECT_ROOT/requirements-prod.txt"

    success "Zaleznosci zainstalowane pomyslnie."
else
    warn "Pomijam instalacje zaleznosci (--skip-install)."
fi

# ---------- usluga systemd (opcjonalnie) ----------
if $INSTALL_SERVICE; then
    SERVICE_SRC="$PROJECT_ROOT/scripts/systemd/remote-droids-server.service"
    SERVICE_DST="/etc/systemd/system/remote-droids-server.service"

    [[ -f "$SERVICE_SRC" ]] || error "Nie znaleziono pliku uslugi: $SERVICE_SRC"

    info "Kopiowanie pliku uslugi do $SERVICE_DST..."
    sudo cp "$SERVICE_SRC" "$SERVICE_DST"

    # Podmien WorkingDirectory i ExecStart na aktualna sciezke projektu
    sudo sed -i \
        -e "s|WorkingDirectory=.*|WorkingDirectory=$PROJECT_ROOT|" \
        -e "s|ExecStart=.*|ExecStart=$VENV_DIR/bin/python $PROJECT_ROOT/src/server/init.py|" \
        -e "s|User=.*|User=$(whoami)|" \
        "$SERVICE_DST"

    sudo systemctl daemon-reload
    sudo systemctl enable remote-droids-server.service
    sudo systemctl start  remote-droids-server.service

    success "Usluga systemd zainstalowana i uruchomiona."
    info "Status: sudo systemctl status remote-droids-server.service"
fi

# ---------- uruchomienie serwera ----------
echo ""
success "=== Konfiguracja zakonczona - uruchamiam serwer ==="
info "Uruchamianie: $PYTHON src/server/init.py"
exec "$PYTHON" "$PROJECT_ROOT/src/server/init.py"
