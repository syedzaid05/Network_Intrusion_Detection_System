"""Tests for the PCAP flow extractor."""
from __future__ import annotations

import pytest

from src import config, pcap_extractor

scapy = pytest.importorskip("scapy.all")


def _write_pcap(path):
    from scapy.all import Ether, IP, TCP, UDP, Raw, wrpcap

    packets = []
    t = 1000.0
    # one TCP session (SYN, SYN-ACK, ACK, PS, PA, FIN)
    chain = [
        (IP(src="192.168.1.10", dst="10.0.0.2", ttl=64) / TCP(sport=50000, dport=80,
         flags="S", seq=1000, window=29200)),
        (IP(src="10.0.0.2", dst="192.168.1.10", ttl=54) / TCP(sport=80, dport=50000,
         flags="SA", seq=5000, ack=1001, window=28960)),
        (IP(src="192.168.1.10", dst="10.0.0.2", ttl=64) / TCP(sport=50000, dport=80,
         flags="A", seq=1001, ack=5001, window=29200)),
        (IP(src="192.168.1.10", dst="10.0.0.2", ttl=64) / TCP(sport=50000, dport=80,
         flags="PA", seq=1001, ack=5001) / Raw(b"\x00" * 60)),
        (IP(src="10.0.0.2", dst="192.168.1.10", ttl=54) / TCP(sport=80, dport=50000,
         flags="FA", seq=5001, ack=1061)),
    ]
    for i, pkt in enumerate(chain):
        pkt.time = t + i * 0.05
        packets.append(Ether() / pkt)
    udp = Ether() / IP(src="192.168.1.10", dst="8.8.8.8", ttl=64) / UDP(sport=53000, dport=53) / Raw(b"\x00" * 30)
    udp.time = t + 1.0
    packets.append(udp)
    wrpcap(str(path), packets)


def test_extract_flows(tmp_path):
    pcap = tmp_path / "cap.pcap"
    _write_pcap(pcap)
    records = pcap_extractor.extract_flows(str(pcap))
    assert len(records) == 2  # one TCP flow + one UDP flow
    for rec in records:
        assert list(rec.keys()) == config.ALL_FEATURE_COLS

    http = next(r for r in records if r["service"] == "http")
    assert http["spkts"] >= 1 and http["dpkts"] >= 1
    assert http["dur"] > 0
    assert http["proto"] == "tcp"

    dns = next(r for r in records if r["service"] == "dns")
    assert dns["proto"] == "udp"


def test_pcap_to_dataframe(tmp_path):
    pcap = tmp_path / "cap2.pcap"
    _write_pcap(pcap)
    df = pcap_extractor.pcap_to_dataframe(str(pcap))
    assert list(df.columns) == config.ALL_FEATURE_COLS
    assert len(df) == 2
