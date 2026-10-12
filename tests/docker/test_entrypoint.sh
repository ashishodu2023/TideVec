#!/bin/sh
# shellcheck disable=SC2016
# Checks docker/cpu/entrypoint.sh without Docker: the server binary is
# replaced by a stub that prints its arguments (one per line) and whether
# TIDEVEC_API_KEY reached its environment.
set -u
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

cat > "$tmp/server" <<'EOF'
#!/bin/sh
for a in "$@"; do printf 'ARG:%s\n' "$a"; done
printf 'KEY:%s\n' "${TIDEVEC_API_KEY:-<unset>}"
EOF
chmod +x "$tmp/server"
sed "s#/usr/local/bin/tidevec-server#$tmp/server#" "$root/docker/cpu/entrypoint.sh" > "$tmp/entrypoint.sh"

pass=0 fail=0
# run NAME EXPECT_RC [VAR=VALUE ...] -- [ARGS ...]
run() {
    name=$1 want_rc=$2; shift 2
    envs=""
    while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do envs="$envs $1"; shift; done
    [ "$#" -gt 0 ] && shift
    # shellcheck disable=SC2086
    out=$(env -i PATH="$PATH" $envs sh "$tmp/entrypoint.sh" "$@" 2>&1)
    rc=$?
    if [ "$rc" -ne "$want_rc" ]; then
        echo "FAIL $name: rc=$rc (want $want_rc)"; echo "$out" | sed 's/^/    /'
        fail=$((fail + 1)); return 1
    fi
    return 0
}
has()    { printf '%s\n' "$out" | grep -qxF -- "$1"; }
check()  { if "$@"; then :; else echo "FAIL $name: $*"; echo "$out" | sed 's/^/    /'; fail=$((fail + 1)); return 1; fi; }
ok()     { pass=$((pass + 1)); echo "ok   $name"; }

run defaults 0 -- &&
    check has "ARG:--require-auth" && check has "ARG:--ultra-durable" &&
    check eval 'printf "%s\n" "$out" | grep -q "^KEY:tv_[0-9a-f]\{32\}$"' &&
    check eval '! printf "%s\n" "$out" | grep -q "^ARG:--quiet$"' && ok

run auth_off 0 TIDEVEC_REQUIRE_AUTH=0 -- &&
    check has "ARG:--no-require-auth" && check has "KEY:<unset>" &&
    check eval '! printf "%s\n" "$out" | grep -q "Generated"' && ok

run auth_off_keeps_no_key 0 TIDEVEC_REQUIRE_AUTH=0 TIDEVEC_API_KEY=k -- &&
    check has "ARG:--no-require-auth" && ok

run ultra_and_segments_off 0 TIDEVEC_ULTRA_DURABLE=0 TIDEVEC_SEGMENT_STORE=0 -- &&
    check has "ARG:--no-ultra-durable" && check has "ARG:--no-segment-store" && ok

run backup_interval 0 TIDEVEC_BACKUP=1 TIDEVEC_BACKUP_INTERVAL_HOURS=12 -- &&
    check has "ARG:--backup" && check has "ARG:--backup-interval-hours" && check has "ARG:12" && ok
run backup_interval_zero 1 TIDEVEC_BACKUP_INTERVAL_HOURS=0 -- && ok
run backup_interval_text 1 TIDEVEC_BACKUP_INTERVAL_HOURS=6h -- && ok

run log_warn 0 TIDEVEC_LOG_LEVEL=WARN -- && check has "ARG:--quiet" && ok
run log_debug 0 TIDEVEC_LOG_LEVEL=debug -- &&
    check eval '! printf "%s\n" "$out" | grep -q "^ARG:--quiet$"' && ok
run log_bad 1 TIDEVEC_LOG_LEVEL=verbose -- && ok
run quiet 0 TIDEVEC_QUIET=1 -- && check has "ARG:--quiet" && ok

printf 'cert' > "$tmp/c.pem"; printf 'key' > "$tmp/k.pem"
run tls 0 TIDEVEC_TLS_CERT="$tmp/c.pem" TIDEVEC_TLS_KEY="$tmp/k.pem" -- &&
    check has "ARG:--tls-cert" && check has "ARG:$tmp/c.pem" &&
    check has "ARG:--tls-key" && check has "ARG:$tmp/k.pem" && ok
run tls_cert_only 1 TIDEVEC_TLS_CERT="$tmp/c.pem" -- && ok
run tls_missing_file 1 TIDEVEC_TLS_CERT="$tmp/c.pem" TIDEVEC_TLS_KEY="$tmp/nope.pem" -- && ok

# Container arguments come last (so they override) and stay intact.
run user_args 0 TIDEVEC_THREADS=4 -- --threads 32 "a b" &&
    check eval '[ "$(printf "%s\n" "$out" | grep "^ARG:" | tail -3 | tr "\n" "|")" = "ARG:--threads|ARG:32|ARG:a b|" ]' && ok

# A value with spaces is one argument, not extra flags.
name=no_injection
out=$(env -i PATH="$PATH" "TIDEVEC_RATE_LIMIT=5 --no-require-auth" sh "$tmp/entrypoint.sh" 2>&1)
check has "ARG:5 --no-require-auth" && check has "ARG:--require-auth" &&
    check eval '! printf "%s\n" "$out" | grep -qx "ARG:--no-require-auth"' && ok

echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
