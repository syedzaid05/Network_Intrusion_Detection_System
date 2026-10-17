"""Approximate UNSW-NB15 feature extraction from a PCAP file (scapy).

This is a lightweight, dependency-friendly flow exporter: it groups packets into
bidirectional flows and computes the UNSW-NB15 feature set using the fields
available from raw packets. A few UNSW features that require deep inspection
(HTTP transaction depth, FTP commands, retransmission analysis) are approximated
as 0, and the ``ct_*`` last-100-flows counters use a simplified window.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from . import config

try:  # scapy is optional at import time
    from scapy.layers.inet import IP, TCP, UDP, ICMP
    from scapy.layers.l2 import ARP
    from scapy.utils import PcapReader
except Exception:  # pragma: no cover
    PcapReader = None

HTTP_PORTS = {80, 8080, 8000, 8888}
PORT_SERVICE = {
    20: "ftp-data", 21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp", 53: "dns",
    67: "dhcp", 68: "dhcp", 69: "tftp", 80: "http", 110: "pop3", 123: "ntp",
    143: "imap", 161: "snmp", 443: "https", 465: "smtp", 993: "imap", 995: "pop3",
    3306: "mysql", 3389: "rdp", 5432: "postgres", 8080: "http",
}

IP_PROTO = {1: "icmp", 6: "tcp", 17: "udp", 89: "ospf", 132: "sctp", 2: "igmp"}
PROTO_OPTIONS = {"icmp", "igmp", "tcp", "udp", "ospf", "sctp", "arp", "any"}


def _service(port: int | None) -> str:
    if port is None:
        return "-"
    return PORT_SERVICE.get(port, "-")


@dataclass
class Flow:
    initiator: tuple
    first_ts: float
    last_ts: float
    # per-direction accumulators: forward = source->dest (initiator first)
    fwd_pkts: int = 0
    bwd_pkts: int = 0
    fwd_bytes: int = 0
    bwd_bytes: int = 0
    fwd_sizes: list = field(default_factory=list)
    bwd_sizes: list = field(default_factory=list)
    fwd_times: list = field(default_factory=list)
    bwd_times: list = field(default_factory=list)
    fwd_ttl: int = 0
    bwd_ttl: int = 0
    fwd_win: int = 0
    bwd_win: int = 0
    fwd_tcpb: int = 0
    bwd_tcpb: int = 0
    syn_ts: float | None = None
    synack_ts: float | None = None
    ack_ts: float | None = None
    flags: set = field(default_factory=set)
    proto: str = "any"
    sport: int | None = None
    dport: int | None = None
    src_ip: str = ""
    dst_ip: str = ""

    def add(self, ts: float, is_fwd: bool, size: int, ttl: int, win: int, tcpb: int):
        self.last_ts = ts
        if is_fwd:
            self.fwd_pkts += 1
            self.fwd_bytes += size
            self.fwd_sizes.append(size)
            self.fwd_times.append(ts)
            self.fwd_ttl = ttl or self.fwd_ttl
            self.fwd_win = win or self.fwd_win
            self.fwd_tcpb = tcpb or self.fwd_tcpb
        else:
            self.bwd_pkts += 1
            self.bwd_bytes += size
            self.bwd_sizes.append(size)
            self.bwd_times.append(ts)
            self.bwd_ttl = ttl or self.bwd_ttl
            self.bwd_win = win or self.bwd_win
            self.bwd_tcpb = tcpb or self.bwd_tcpb

    def state(self) -> str:
        f = self.flags
        if "R" in f and "S" in f:
            return "RST"
        if "R" in f:
            return "RST"
        if "F" in f:
            return "FIN"
        if "S" in f and "A" not in f:
            return "REQ"
        if "S" in f and "A" in f:
            return "CON"
        if "S" in f:
            return "CON"
        return "INT"


def _inter_arrival(times: list[float]) -> tuple[float, float]:
    if len(times) < 2:
        return 0.0, 0.0
    diffs = [b - a for a, b in zip(times, times[1:])]
    mean = sum(diffs) / len(diffs)
    if len(diffs) < 2:
        return mean, 0.0
    jitter = sum(abs(b - a) for a, b in zip(diffs, diffs[1:])) / (len(diffs) - 1)
    return mean, jitter


def _flow_to_record(flow: Flow, ct: dict) -> dict:
    dur = max(flow.last_ts - flow.first_ts, 1e-9)
    total_pkts = flow.fwd_pkts + flow.bwd_pkts
    rate = total_pkts / dur if dur > 0 else 0.0
    sinpkt, sjit = _inter_arrival(flow.fwd_times)
    dinpkt, djit = _inter_arrival(flow.bwd_times)
    smean = flow.fwd_bytes / flow.fwd_pkts if flow.fwd_pkts else 0
    dmean = flow.bwd_bytes / flow.bwd_pkts if flow.bwd_pkts else 0

    tcprtt = synack = ackdat = 0.0
    if flow.syn_ts is not None and flow.synack_ts is not None:
        synack = max(flow.synack_ts - flow.syn_ts, 0.0)
        if flow.ack_ts is not None:
            ackdat = max(flow.ack_ts - flow.synack_ts, 0.0)
            tcprtt = synack + ackdat

    record = {
        "dur": round(dur, 6),
        "spkts": flow.fwd_pkts,
        "dpkts": flow.bwd_pkts,
        "sbytes": flow.fwd_bytes,
        "dbytes": flow.bwd_bytes,
        "rate": round(rate, 4),
        "sttl": flow.fwd_ttl,
        "dttl": flow.bwd_ttl,
        "sload": round(flow.fwd_bytes * 8 / dur, 4),
        "dload": round(flow.bwd_bytes * 8 / dur, 4),
        "sloss": 0,
        "dloss": 0,
        "sinpkt": round(sinpkt * 1000, 6),
        "dinpkt": round(dinpkt * 1000, 6),
        "sjit": round(sjit * 1000, 6),
        "djit": round(djit * 1000, 6),
        "swin": flow.fwd_win,
        "stcpb": flow.fwd_tcpb,
        "dtcpb": flow.bwd_tcpb,
        "dwin": flow.bwd_win,
        "tcprtt": round(tcprtt, 6),
        "synack": round(synack, 6),
        "ackdat": round(ackdat, 6),
        "smean": round(smean, 3),
        "dmean": round(dmean, 3),
        "trans_depth": 0,
        "response_body_len": 0,
        "ct_srv_src": ct["ct_srv_src"],
        "ct_state_ttl": ct["ct_state_ttl"],
        "ct_dst_ltm": ct["ct_dst_ltm"],
        "ct_src_dport_ltm": ct["ct_src_dport_ltm"],
        "ct_dst_sport_ltm": ct["ct_dst_sport_ltm"],
        "ct_dst_src_ltm": ct["ct_dst_src_ltm"],
        "is_ftp_login": 0,
        "ct_ftp_cmd": 0,
        "ct_flw_http_mthd": 0,
        "ct_src_ltm": ct["ct_src_ltm"],
        "ct_srv_dst": ct["ct_srv_dst"],
        "is_sm_ips_ports": int(flow.src_ip == flow.dst_ip and flow.sport == flow.dport),
        "proto": flow.proto,
        "service": _service(flow.dport),
        "state": flow.state(),
    }
    return {k: record.get(k, 0) for k in config.ALL_FEATURE_COLS}


class _Windows:
    """Simplified last-100-flow connection counters."""

    def __init__(self, size: int = 100):
        self.n = size
        self.dst = deque(maxlen=size)
        self.src = deque(maxlen=size)
        self.srv_src = deque(maxlen=size)
        self.srv_dst = deque(maxlen=size)
        self.dst_src = deque(maxlen=size)
        self.src_dport = deque(maxlen=size)
        self.dst_sport = deque(maxlen=size)

    def counts(self, flow: Flow) -> dict:
        return {
            "ct_dst_ltm": self.dst.count(flow.dst_ip) + 1,
            "ct_src_ltm": self.src.count(flow.src_ip) + 1,
            "ct_srv_src": self.srv_src.count((flow.dport, flow.src_ip)) + 1,
            "ct_srv_dst": self.srv_dst.count((flow.dport, flow.dst_ip)) + 1,
            "ct_dst_src_ltm": self.dst_src.count((flow.dst_ip, flow.src_ip)) + 1,
            "ct_src_dport_ltm": self.src_dport.count((flow.src_ip, flow.dport)) + 1,
            "ct_dst_sport_ltm": self.dst_sport.count((flow.dst_ip, flow.sport)) + 1,
            "ct_state_ttl": 1,
        }

    def update(self, flow: Flow) -> None:
        self.dst.append(flow.dst_ip)
        self.src.append(flow.src_ip)
        self.srv_src.append((flow.dport, flow.src_ip))
        self.srv_dst.append((flow.dport, flow.dst_ip))
        self.dst_src.append((flow.dst_ip, flow.src_ip))
        self.src_dport.append((flow.src_ip, flow.dport))
        self.dst_sport.append((flow.dst_ip, flow.sport))


def _endpoints(pkt):
    ip = pkt[IP]
    sport = dport = None
    if pkt.haslayer(TCP):
        sport, dport = int(pkt[TCP].sport), int(pkt[TCP].dport)
    elif pkt.haslayer(UDP):
        sport, dport = int(pkt[UDP].sport), int(pkt[UDP].dport)
    return (ip.src, sport), (ip.dst, dport)


def extract_flows(pcap_path: str, max_flows: int | None = None) -> list[dict]:
    """Read a PCAP and return a list of UNSW-NB15-style flow records."""
    if PcapReader is None:  # pragma: no cover
        raise RuntimeError("scapy is required for PCAP extraction: pip install scapy")

    flows: dict[tuple, Flow] = {}
    order: list[tuple] = []
    windows = _Windows()

    def finalise(flow: Flow) -> dict:
        ct = windows.counts(flow)
        windows.update(flow)
        return _flow_to_record(flow, ct)

    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            ts = float(pkt.time)
            if pkt.haslayer(ARP):
                key = ("arp", pkt[ARP].psrc, pkt[ARP].pdst)
                if key not in flows:
                    flows[key] = Flow(initiator=key, first_ts=ts, last_ts=ts, proto="arp")
                    order.append(key)
                continue
            if not pkt.haslayer(IP):
                continue

            (src, sport), (dst, dport) = _endpoints(pkt)
            a, b = (src, sport), (dst, dport)
            key = (a, b) if a <= b else (b, a)

            flow = flows.get(key)
            if flow is None:
                flow = Flow(
                    initiator=a, first_ts=ts, last_ts=ts,
                    proto=IP_PROTO.get(int(pkt[IP].proto), "any"),
                    src_ip=src, dst_ip=dst, sport=sport, dport=dport,
                )
                flows[key] = flow
                order.append(key)

            is_fwd = flow.initiator == a
            size = int(getattr(pkt[IP], "len", len(pkt)))
            ttl = int(pkt[IP].ttl)
            win = tcpb = 0
            if pkt.haslayer(TCP):
                win = int(pkt[TCP].window)
                tcpb = int(pkt[TCP].seq)
                flags = str(pkt[TCP].flags)
                flow.flags.update(flags)
                if "S" in flags and "A" not in flags and flow.syn_ts is None:
                    flow.syn_ts = ts
                elif "S" in flags and "A" in flags and flow.synack_ts is None:
                    flow.synack_ts = ts
                elif "A" in flags and "S" not in flags and flow.ack_ts is None:
                    flow.ack_ts = ts
            flow.add(ts, is_fwd, size, ttl, win, tcpb)

            if max_flows is not None and len(order) >= max_flows * 4:
                # safety valve for very large captures
                break

    return [finalise(flows[k]) for k in order]


def pcap_to_dataframe(pcap_path: str, max_flows: int | None = None):
    import pandas as pd

    records = extract_flows(pcap_path, max_flows=max_flows)
    if max_flows is not None:
        records = records[:max_flows]
    return pd.DataFrame(records, columns=config.ALL_FEATURE_COLS)
