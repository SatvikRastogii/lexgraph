"""
LexGraph — Telemetry Exporter
Bridges telemetry.py's SQLite query/RAGAS log into Prometheus metrics so
Grafana can chart trends and alert on them. SQLite stays the source of
truth for ad-hoc drill-down; Prometheus only ever sees aggregates re-derived
from it on each poll, so nothing is double-recorded.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import telemetry
from prometheus_client import Gauge, Counter, start_http_server

POLL_INTERVAL_SECONDS = int(os.getenv("TELEMETRY_EXPORTER_INTERVAL", "15"))
EXPORTER_PORT = int(os.getenv("TELEMETRY_EXPORTER_PORT", "9200"))

queries_total = Counter("lexgraph_queries_total", "Total query events logged", ["route"])
query_failures_total = Counter("lexgraph_query_failures_total", "Total failed query events", ["route"])
query_latency_ms = Gauge("lexgraph_query_latency_ms", "Most recent pipeline latency sample (ms)", ["route"])
failure_rate = Gauge("lexgraph_failure_rate", "Failure rate over the last 50 query events")
ragas_score = Gauge("lexgraph_ragas_score", "Most recent RAGAS drift-check score", ["metric", "pipeline"])
ragas_drift_detected = Gauge("lexgraph_ragas_drift_detected", "1 if the last drift check flagged a regression", ["metric", "pipeline"])

# Running counts re-derived each poll; Counter only supports .inc(), so track
# the last-seen total per route and increment by the delta.
_last_counts = {}
_last_failures = {}


def _sync_counter(counter, label, current_total, last_map):
    previous = last_map.get(label, 0)
    delta = current_total - previous
    if delta > 0:
        counter.labels(route=label).inc(delta)
    last_map[label] = current_total


def poll_once():
    events = telemetry.fetch_recent_events(limit=1000)

    totals, failures, latest_latency = {}, {}, {}
    for e in events:
        route = e["route"]
        totals[route] = totals.get(route, 0) + 1
        if not e["success"]:
            failures[route] = failures.get(route, 0) + 1
        if route not in latest_latency and e.get("pipeline_latency_ms") is not None:
            latest_latency[route] = e["pipeline_latency_ms"]  # events are newest-first

    for route, total in totals.items():
        _sync_counter(queries_total, route, total, _last_counts)
    for route, count in failures.items():
        _sync_counter(query_failures_total, route, count, _last_failures)
    for route, latency in latest_latency.items():
        query_latency_ms.labels(route=route).set(latency)

    failure_rate.set(telemetry.fetch_failure_rate(window=50))

    for row in telemetry.fetch_ragas_history(limit=500):
        ragas_score.labels(metric=row["metric"], pipeline=row["pipeline"]).set(row["score"])
        ragas_drift_detected.labels(metric=row["metric"], pipeline=row["pipeline"]).set(
            1 if row["drift_detected"] else 0
        )


def main():
    telemetry.init_db()
    start_http_server(EXPORTER_PORT)
    print(f"telemetry_exporter listening on :{EXPORTER_PORT}, polling every {POLL_INTERVAL_SECONDS}s")
    while True:
        try:
            poll_once()
        except Exception as e:
            print(f"telemetry_exporter poll error: {e}")
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
