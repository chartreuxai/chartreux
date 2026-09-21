# 0008 Feature Instrumentation

## Decision

Chartreux does not collect product analytics or export product telemetry. Application
code must not create analytics events, OpenTelemetry spans, tracing wrappers, or
telemetry exporters. This is a policy against Chartreux instrumentation and
analytics traffic, not a promise that every dependency is free of OTel packages or
that arbitrary user-loaded extensions cannot make their own network requests.

Provider SDKs may retain OTel APIs and semantic conventions transitively. Supported
provider instances must use their available per-instance opt-out configuration;
transitive packages are not evidence that Chartreux instrumentation is active.
Local diagnostic logs, session logs, and token/cost accounting remain supported.

## Rationale

The standalone harness needs predictable local behavior without product analytics
or a remote telemetry service. Retaining ordinary diagnostics and accounting keeps
failures, sessions, and usage understandable without turning them into analytics
streams.

## Agent Guidance

- Do not add analytics events, spans, exporters, or telemetry configuration.
- Do not load or invoke the feature-analytics skill for Chartreux instrumentation.
- Preserve local diagnostics, session logging, and token/cost accounting.
- Inspect the pinned provider SDK's actual construction options and test every
  supported construction path under hostile telemetry settings.
- Do not remove a transitive OTel dependency solely because its name remains in the
  lockfile, and do not claim to control arbitrary user extension traffic.

## Flag To User When

- A feature requires product analytics or remote telemetry to be considered complete.
- A provider SDK cannot be configured to opt out on a supported construction path.
- A proposed change removes local diagnostics, session logs, or usage accounting.
- A change claims that all OTel packages or all extension-generated traffic vanish.
