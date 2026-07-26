#!/bin/sh
#
# Container-local liveness probe for the official image. Keep this POSIX-sh:
# the runtime image deliberately contains no Bash.
set -eu

listen=${NAMIDB_LISTEN:-0.0.0.0:8080}
port=${listen##*:}
address=${listen%:*}

case "$port" in
    '' | *[!0-9]*)
        echo "invalid NAMIDB_LISTEN port for healthcheck: $listen" >&2
        exit 2
        ;;
esac

# Connect to the corresponding loopback address for wildcard listeners. Keep
# brackets around IPv6 literals so curl parses the URL unambiguously.
case "$address" in
    0.0.0.0)
        health_host=127.0.0.1
        ;;
    '[::]')
        health_host='[::1]'
        ;;
    '['*']')
        health_host=$address
        ;;
    *)
        health_host=$address
        ;;
esac

if [ -n "${NAMIDB_TLS_CERT:-}" ] || [ -n "${NAMIDB_TLS_KEY:-}" ]; then
    # Certificate verification is intentionally disabled only for this
    # container-local liveness request: deployment certificates commonly do
    # not contain the loopback address in their SANs.
    exec curl --fail --silent --show-error --insecure --noproxy '*' --max-time 3 \
        "https://${health_host}:${port}/v0/livez"
fi

exec curl --fail --silent --show-error --noproxy '*' --max-time 3 \
    "http://${health_host}:${port}/v0/livez"
