"""Pydantic request/response schemas for the prediction API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TrafficRecord(BaseModel):
    """A single network flow described by UNSW-NB15 raw features."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "dur": 0.121478, "spkts": 6, "dpkts": 4, "sbytes": 258,
                "dbytes": 172, "rate": 74.087, "sttl": 252, "dttl": 254,
                "sload": 14158.94, "dload": 8495.36, "sloss": 0, "dloss": 0,
                "sinpkt": 24.2956, "dinpkt": 8.375, "sjit": 30.177, "djit": 11.83,
                "swin": 255, "stcpb": 621772692, "dtcpb": 2202533631, "dwin": 255,
                "tcprtt": 0.0, "synack": 0.0, "ackdat": 0.0, "smean": 43, "dmean": 43,
                "trans_depth": 0, "response_body_len": 0, "ct_srv_src": 1,
                "ct_state_ttl": 0, "ct_dst_ltm": 1, "ct_src_dport_ltm": 1,
                "ct_dst_sport_ltm": 1, "ct_dst_src_ltm": 1, "is_ftp_login": 0,
                "ct_ftp_cmd": 0, "ct_flw_http_mthd": 0, "ct_src_ltm": 1,
                "ct_srv_dst": 1, "is_sm_ips_ports": 0, "proto": "tcp",
                "service": "-", "state": "FIN",
            }
        },
    )

    dur: float = Field(0.0, ge=0, description="Flow duration (s)")
    spkts: float = Field(0.0, ge=0, description="Source->dest packet count")
    dpkts: float = Field(0.0, ge=0, description="Dest->source packet count")
    sbytes: float = Field(0.0, ge=0, description="Source->dest bytes")
    dbytes: float = Field(0.0, ge=0, description="Dest->source bytes")
    rate: float = Field(0.0, ge=0, description="Flow rate (pkt/s)")
    sttl: float = Field(0.0, ge=0, description="Source TTL")
    dttl: float = Field(0.0, ge=0, description="Destination TTL")
    sload: float = Field(0.0, ge=0, description="Source bits/s")
    dload: float = Field(0.0, ge=0, description="Destination bits/s")
    sloss: float = Field(0.0, ge=0, description="Source retransmitted packets")
    dloss: float = Field(0.0, ge=0, description="Destination retransmitted packets")
    sinpkt: float = Field(0.0, ge=0, description="Source inter-packet arrival time")
    dinpkt: float = Field(0.0, ge=0, description="Destination inter-packet arrival time")
    sjit: float = Field(0.0, ge=0, description="Source jitter")
    djit: float = Field(0.0, ge=0, description="Destination jitter")
    swin: float = Field(0.0, ge=0, description="Source TCP window")
    stcpb: float = Field(0.0, ge=0, description="Source TCP base sequence")
    dtcpb: float = Field(0.0, ge=0, description="Destination TCP base sequence")
    dwin: float = Field(0.0, ge=0, description="Destination TCP window")
    tcprtt: float = Field(0.0, ge=0, description="TCP round-trip time")
    synack: float = Field(0.0, ge=0, description="SYN->SYN_ACK time")
    ackdat: float = Field(0.0, ge=0, description="SYN_ACK->ACK time")
    smean: float = Field(0.0, ge=0, description="Mean source packet size")
    dmean: float = Field(0.0, ge=0, description="Mean destination packet size")
    trans_depth: float = Field(0.0, ge=0, description="HTTP transaction depth")
    response_body_len: float = Field(0.0, ge=0, description="HTTP response body length")
    ct_srv_src: float = Field(0.0, ge=0, description="Flows to same service/src in 100 flows")
    ct_state_ttl: float = Field(0.0, ge=0, description="Flows with same state/TTL")
    ct_dst_ltm: float = Field(0.0, ge=0, description="Flows to same dest in last 100")
    ct_src_dport_ltm: float = Field(0.0, ge=0, description="Flows same src/dest port last 100")
    ct_dst_sport_ltm: float = Field(0.0, ge=0, description="Flows same dest/sport last 100")
    ct_dst_src_ltm: float = Field(0.0, ge=0, description="Flows same src/dest last 100")
    is_ftp_login: float = Field(0.0, ge=0, description="FTP login flag")
    ct_ftp_cmd: float = Field(0.0, ge=0, description="FTP command count")
    ct_flw_http_mthd: float = Field(0.0, ge=0, description="HTTP methods count")
    ct_src_ltm: float = Field(0.0, ge=0, description="Flows same source last 100")
    ct_srv_dst: float = Field(0.0, ge=0, description="Flows same service/dest last 100")
    is_sm_ips_ports: float = Field(0.0, ge=0, description="Same src/dest IP+port flag")

    proto: str = Field("tcp", description="Protocol (tcp/udp/icmp/...)")
    service: str = Field("-", description="Service (http/ftp/dns/... or -)")
    state: str = Field("FIN", description="Connection state (FIN/CON/INT/...)")


class PredictionResponse(BaseModel):
    label: int = Field(..., description="0 = Normal, 1 = Attack")
    verdict: Literal["Normal", "Attack"]
    severity: Literal["Normal", "Suspicious", "Malicious"]
    attack_probability: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    attack_category: str | None = Field(
        None, description="Predicted attack type (e.g. DoS, Exploits) or Normal"
    )


class BatchRequest(BaseModel):
    records: list[TrafficRecord] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    count: int
    attacks: int
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_ready: bool


class ModelInfoResponse(BaseModel):
    model_ready: bool
    best_model: str | None = None
    trained_at: str | None = None
    dataset: str | None = None
    n_features_input: int | None = None
    n_selected: int | None = None
    decision_threshold: float | None = None
    multiclass_ready: bool = False
    attack_categories: list[str] = Field(default_factory=list)
    selected_features: list[str] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)


class PcapSummary(BaseModel):
    pcap: str
    flows: int
    attacks: int
    attack_rate: float
    severity_counts: dict
    category_counts: dict
    top_flows: list[dict]
