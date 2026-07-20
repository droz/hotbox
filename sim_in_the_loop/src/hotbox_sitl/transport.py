from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TransportMailbox:
    sent_frames: list[bytes] = field(default_factory=list)
    received_frames: list[bytes] = field(default_factory=list)

    def send(self, frame: bytes) -> None:
        self.sent_frames.append(frame)

    def inject(self, frame: bytes) -> None:
        self.received_frames.append(frame)
