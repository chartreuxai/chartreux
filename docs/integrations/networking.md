# Networking

Chartreux-owned HTTP clients use proxy settings from the process environment.
You may set them in the launching shell or in `$CHARTREUX_HOME/.env`; the
terminal `/proxy-setup` interface validates the full batch of supported settings
before writing any values there; if a field is invalid, it keeps the editor open
and shows the error without partial writes.

## Proxy variables

| Variable | Purpose |
| --- | --- |
| `HTTP_PROXY` | Proxy URL for HTTP requests. |
| `HTTPS_PROXY` | Proxy URL for HTTPS requests. |
| `ALL_PROXY` | Fallback proxy URL when a scheme-specific proxy is not set. |
| `NO_PROXY` | Comma-separated proxy-bypass rules. |
| `SSL_CERT_FILE` | Additional certificate bundle or certificate file. |
| `SSL_CERT_DIR` | Additional certificate directory. |

Lowercase proxy spellings are also recognized by the platform environment proxy
reader. A proxy URL without a scheme is treated as an HTTP proxy URL.

`NO_PROXY` accepts `*`, host names, domain suffixes (for example `.example.com`),
IP literals, optional ports, and CIDR networks. CIDR rules match only IP-literal
request hosts: Chartreux does not resolve DNS names when deciding whether to use
a proxy. `HTTP_PROXY` and `HTTPS_PROXY` take precedence for their respective
schemes; `ALL_PROXY` is the fallback.

## TLS trust

By default, outbound HTTPS trusts the bundled Certifi root store. To use the
operating-system trust store instead, set this `config.toml` key:

```toml
enable_system_trust_store = true
```

`SSL_CERT_FILE` and `SSL_CERT_DIR` are additive in both modes: they add private
or corporate trust anchors without replacing public roots. If a custom path
cannot be loaded, Chartreux logs a warning and continues with the selected base
trust store.

## Applying changes

A newly started CLI process reads its environment and `.env` values at startup.
The ACP entrypoint also loads dotenv at startup, so changing `.env` requires
restarting the ACP process or having the ACP client launch a new one. This is not
a browser sign-in flow. See [ACP](acp.md) and the [configuration reference](../reference/configuration.md#environment-variables-and-env).
