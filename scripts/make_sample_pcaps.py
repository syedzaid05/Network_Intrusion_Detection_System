"""Generate sample PCAP files for testing the PCAP Analysis tab / API.

Usage:
    python scripts/make_sample_pcaps.py [--out data/pcaps]

Creates six captures with different traffic patterns:
    normal_http.pcap   - clean TCP handshake + HTTP request/response
    port_scan.pcap     - one host scanning many TCP ports
    syn_flood.pcap     - many half-open TCP SYNs to a single service
    udp_dns.pcap       - UDP/DNS queries
    icmp_ping.pcap     - ICMP echo requests
    mixed_traffic.pcap - a blend of the above
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scapy.all import Ether, ICMP, IP, Raw, TCP, UDP, wrpcap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "pcaps"
T0 = 1_700_000_000.0


def _stamp(pkts: list, t0: float = T0, step: float = 0.001) -> list:
    for i, p in enumerate(pkts):
        p.time = t0 + i * step
    return pkts


def normal_http() -> list:
    src, dst, sport = "192.168.1.10", "93.184.216.34", 41000
    seq_c, seq_s = 1000, 5000
    req = b"GET / HTTP/1.1\r\nHost: example.com\r\nUser-Agent: demo\r\n\r\n"
    resp = b"HTTP/1.1 200 OK\r\nContent-Length: 13\r\n\r\nHello, world!"
    pkts = [
        Ether() / IP(src=src, dst=dst) / TCP(sport=sport, dport=80, flags="S", seq=seq_c, window=64240),
        Ether() / IP(src=dst, dst=src) / TCP(sport=80, dport=sport, flags="SA", seq=seq_s, ack=seq_c + 1, window=65535),
        Ether() / IP(src=src, dst=dst) / TCP(sport=sport, dport=80, flags="A", seq=seq_c + 1, ack=seq_s + 1, window=64240),
        Ether() / IP(src=src, dst=dst) / TCP(sport=sport, dport=80, flags="PA", seq=seq_c + 1, ack=seq_s + 1) / Raw(req),
        Ether() / IP(src=dst, dst=src) / TCP(sport=80, dport=sport, flags="PA", seq=seq_s + 1, ack=seq_c + 1 + len(req)) / Raw(resp),
        Ether() / IP(src=src, dst=dst) / TCP(sport=sport, dport=80, flags="A", seq=seq_c + 1 + len(req), ack=seq_s + 1 + len(resp)),
        Ether() / IP(src=src, dst=dst) / TCP(sport=sport, dport=80, flags="FA", seq=seq_c + 1 + len(req), ack=seq_s + 1 + len(resp)),
        Ether() / IP(src=dst, dst=src) / TCP(sport=80, dport=sport, flags="FA", seq=seq_s + 1 + len(resp), ack=seq_c + 2 + len(req)),
        Ether() / IP(src=src, dst=dst) / TCP(sport=sport, dport=80, flags="A", seq=seq_c + 2 + len(req), ack=seq_s + 2 + len(resp)),
    ]
    return _stamp(pkts, step=0.02)


def port_scan() -> list:
    rng = random.Random(1)
    src, dst = "10.0.0.66", "10.0.0.5"
    pkts = []
    for port in range(20, 140):
        sport = rng.randint(40000, 60000)
        pkts.append(Ether() / IP(src=src, dst=dst) / TCP(sport=sport, dport=port, flags="S", seq=rng.randint(1, 1 << 30), window=1024))
    return _stamp(pkts, step=0.0005)


def syn_flood() -> list:
    rng = random.Random(2)
    src, dst = "172.16.0.9", "10.0.0.5"
    pkts = []
    for _ in range(400):
        sport = rng.randint(1024, 65535)
        pkts.append(Ether() / IP(src=src, dst=dst) / TCP(sport=sport, dport=80, flags="S", seq=rng.randint(1, 1 << 30), window=512))
    return _stamp(pkts, step=0.0002)


def udp_dns() -> list:
    rng = random.Random(3)
    src, dst = "192.168.1.10", "8.8.8.8"
    names = ["example.com", "openai.com", "github.com", "wikipedia.org", "test.local"]
    pkts = []
    for name in names:
        sport = rng.randint(40000, 60000)
        query = bytes([0x12, 0x34, 0x01, 0x00, 0x00, 0x01, 0, 0, 0, 0, 0, 0]) + b"".join(
            bytes([len(l)]) + l.encode() for l in name.split(".")
        ) + b"\x00"
        pkts.append(Ether() / IP(src=src, dst=dst) / UDP(sport=sport, dport=53) / Raw(query))
        pkts.append(Ether() / IP(src=dst, dst=src) / UDP(sport=53, dport=sport) / Raw(query[:2] + b"\x81\x80" + b"\x00" * 20))
    return _stamp(pkts, step=0.03)


def icmp_ping() -> list:
    src, dst = "192.168.1.10", "1.1.1.1"
    pkts = []
    for i in range(8):
        pkts.append(Ether() / IP(src=src, dst=dst) / ICMP(type=8, id=0x1234, seq=i) / Raw(b"x" * 32))
        pkts.append(Ether() / IP(src=dst, dst=src) / ICMP(type=0, id=0x1234, seq=i) / Raw(b"x" * 32))
    return _stamp(pkts, step=0.5)


def mixed_traffic() -> list:
    pkts = normal_http() + udp_dns() + icmp_ping()
    return _stamp(pkts, step=0.01)


SCENARIOS = {
    "normal_http.pcap": normal_http,
    "port_scan.pcap": port_scan,
    "syn_flood.pcap": syn_flood,
    "udp_dns.pcap": udp_dns,
    "icmp_ping.pcap": icmp_ping,
    "mixed_traffic.pcap": mixed_traffic,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample PCAP files.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output directory")
    args = parser.parse_args()

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, builder in SCENARIOS.items():
        path = out_dir / name
        pkts = builder()
        wrpcap(str(path), pkts)
        print(f"[pcap] {len(pkts):>4} packets -> {path}")

    print(f"\nDone. {len(SCENARIOS)} captures written to {out_dir}")


if __name__ == "__main__":
    main()
