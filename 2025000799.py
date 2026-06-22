"""
Implementation of US20250007996A1 - Claim 1
"Distributing Messages in a Network Environment"

Core concept: A message distribution system where:
1. Followers select MULTIPLE communication endpoints (email, SMS, push, etc.)
   through which they want to receive messages from a specific publisher.
2. When the publisher sends a message to "unspecified recipients", the system:
   a. Identifies all followers of that publisher
   b. For each follower, retrieves their chosen endpoints
   c. Delivers the message to every endpoint that follower selected
"""

from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable


# ---------------------------------------------------------------------------
# Endpoint types (the "plurality of communications endpoints" in the claim)
# ---------------------------------------------------------------------------

class EndpointType(str, Enum):
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push_notification"
    WEBHOOK = "webhook"
    IN_APP = "in_app"


@dataclass
class Endpoint:
    """A single communication endpoint belonging to a user."""
    type: EndpointType
    address: str          # e.g. email address, phone number, device token, URL

    def __hash__(self):
        return hash((self.type, self.address))


# ---------------------------------------------------------------------------
# Core domain objects
# ---------------------------------------------------------------------------

@dataclass
class User:
    user_id: str
    name: str

    def __repr__(self):
        return f"User({self.name!r})"


@dataclass
class Message:
    message_id: str
    sender_id: str
    content: str
    sent_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DeliveryRecord:
    """Represents one delivery attempt to one endpoint."""
    message_id: str
    recipient_id: str
    endpoint: Endpoint
    delivered_at: datetime = field(default_factory=datetime.utcnow)
    success: bool = True


# ---------------------------------------------------------------------------
# Storage layer  (in-memory; swap for a real DB in production)
# ---------------------------------------------------------------------------

class Storage:
    """
    Stores:
    - users
    - follow relationships  (follower_id -> set of followed publisher_ids)
    - endpoint preferences  (follower_id -> publisher_id -> set of Endpoints)
    - delivery log
    """

    def __init__(self):
        self._users: dict[str, User] = {}
        # follower_id -> set of publisher_ids they follow
        self._follows: dict[str, set[str]] = {}
        # (follower_id, publisher_id) -> set of Endpoints
        self._endpoint_prefs: dict[tuple[str, str], set[Endpoint]] = {}
        self._delivery_log: list[DeliveryRecord] = []

    # --- User management ---

    def add_user(self, user: User) -> None:
        self._users[user.user_id] = user

    def get_user(self, user_id: str) -> User | None:
        return self._users.get(user_id)

    # --- Follow relationships ---

    def add_follow(self, follower_id: str, publisher_id: str) -> None:
        self._follows.setdefault(follower_id, set()).add(publisher_id)

    def get_followers(self, publisher_id: str) -> list[str]:
        """Return all user_ids who follow publisher_id."""
        return [uid for uid, followed in self._follows.items()
                if publisher_id in followed]

    # --- Endpoint preferences (the key patent feature) ---

    def set_endpoint_selection(
        self,
        follower_id: str,
        publisher_id: str,
        endpoints: set[Endpoint],
    ) -> None:
        """
        Store a follower's chosen endpoints for a specific publisher.
        This is the 'receiving, from a device of a first user, a selection of
        a plurality of communications endpoints' step from the claim.
        """
        if len(endpoints) < 1:
            raise ValueError("At least one endpoint must be selected.")
        self._endpoint_prefs[(follower_id, publisher_id)] = set(endpoints)

    def get_endpoint_selection(
        self, follower_id: str, publisher_id: str
    ) -> set[Endpoint]:
        return self._endpoint_prefs.get((follower_id, publisher_id), set())

    # --- Delivery log ---

    def log_delivery(self, record: DeliveryRecord) -> None:
        self._delivery_log.append(record)

    def get_delivery_log(self) -> list[DeliveryRecord]:
        return list(self._delivery_log)


# ---------------------------------------------------------------------------
# Delivery adapters  (simulate actual channel delivery)
# ---------------------------------------------------------------------------

DeliveryFn = Callable[[Endpoint, Message, User], bool]


def _default_email_deliver(ep: Endpoint, msg: Message, recipient: User) -> bool:
    print(f"  [EMAIL  → {ep.address}] '{msg.content}'")
    return True


def _default_sms_deliver(ep: Endpoint, msg: Message, recipient: User) -> bool:
    print(f"  [SMS    → {ep.address}] '{msg.content}'")
    return True


def _default_push_deliver(ep: Endpoint, msg: Message, recipient: User) -> bool:
    print(f"  [PUSH   → {ep.address}] '{msg.content}'")
    return True


def _default_webhook_deliver(ep: Endpoint, msg: Message, recipient: User) -> bool:
    print(f"  [WEBHOOK→ {ep.address}] '{msg.content}'")
    return True


def _default_inapp_deliver(ep: Endpoint, msg: Message, recipient: User) -> bool:
    print(f"  [IN-APP → {ep.address}] '{msg.content}'")
    return True


DEFAULT_ADAPTERS: dict[EndpointType, DeliveryFn] = {
    EndpointType.EMAIL:   _default_email_deliver,
    EndpointType.SMS:     _default_sms_deliver,
    EndpointType.PUSH:    _default_push_deliver,
    EndpointType.WEBHOOK: _default_webhook_deliver,
    EndpointType.IN_APP:  _default_inapp_deliver,
}


# ---------------------------------------------------------------------------
# MessageDistributionSystem  (the main claimed system)
# ---------------------------------------------------------------------------

class MessageDistributionSystem:
    """
    Implements the method claimed in US20250007996A1, Claim 1:

    1. receive_endpoint_selection(follower, publisher, endpoints)
       → "receives, from a device of a first user who is a follower of the
          second user, a selection of a plurality of communications endpoints"

    2. distribute_message(publisher, content)
       → "receives, from a device of the second user, a message for
          distribution to one or more unspecified recipients"
       → identifies followers, identifies each follower's chosen endpoints,
          sends the message to each of those endpoints.
    """

    def __init__(
        self,
        storage: Storage | None = None,
        adapters: dict[EndpointType, DeliveryFn] | None = None,
    ):
        self.storage = storage or Storage()
        self.adapters = adapters or DEFAULT_ADAPTERS

    # ---- Step 1 (claim): follower registers endpoint preferences ----------

    def register_user(self, name: str) -> User:
        user = User(user_id=str(uuid.uuid4()), name=name)
        self.storage.add_user(user)
        return user

    def follow(self, follower: User, publisher: User) -> None:
        self.storage.add_follow(follower.user_id, publisher.user_id)

    def select_endpoints(
        self,
        follower: User,
        publisher: User,
        endpoints: set[Endpoint],
    ) -> None:
        """
        Claim step: "receives … a selection of a plurality of communications
        endpoints for receiving messages from the second user" and
        "stores the selection in a storage."
        """
        self.storage.set_endpoint_selection(
            follower.user_id, publisher.user_id, endpoints
        )
        print(
            f"{follower.name} selected {len(endpoints)} endpoint(s) "
            f"for messages from {publisher.name}:"
        )
        for ep in endpoints:
            print(f"  • {ep.type.value}: {ep.address}")

    # ---- Step 2 (claim): publisher sends, system fans out -----------------

    def distribute_message(self, publisher: User, content: str) -> list[DeliveryRecord]:
        """
        Claim steps:
        - "receives, from a device of the second user, a message for
           distribution to one or more unspecified recipients"
        - "identifies the followers of the second user … as recipients"
        - "identifies the plurality of communications endpoints selected
           by the first user"
        - "sends the message to … each of the plurality of communications
           endpoints selected by the first user"
        """
        msg = Message(
            message_id=str(uuid.uuid4()),
            sender_id=publisher.user_id,
            content=content,
        )

        print(f"\n📨  {publisher.name} sent: \"{content}\"")

        # Identify followers  (claim: "identifies the followers of the second user")
        follower_ids = self.storage.get_followers(publisher.user_id)
        print(f"    → {len(follower_ids)} follower(s) identified")

        records: list[DeliveryRecord] = []

        for follower_id in follower_ids:
            follower = self.storage.get_user(follower_id)
            if follower is None:
                continue

            # Identify that follower's chosen endpoints
            endpoints = self.storage.get_endpoint_selection(
                follower_id, publisher.user_id
            )
            if not endpoints:
                print(f"  ⚠  {follower.name} has no endpoint preference set; skipping.")
                continue

            print(f"  ➤ Delivering to {follower.name} via {len(endpoints)} endpoint(s):")

            for ep in endpoints:
                adapter = self.adapters.get(ep.type)
                success = adapter(ep, msg, follower) if adapter else False

                record = DeliveryRecord(
                    message_id=msg.message_id,
                    recipient_id=follower_id,
                    endpoint=ep,
                    success=success,
                )
                self.storage.log_delivery(record)
                records.append(record)

        return records

    # ---- Convenience: delivery summary ------------------------------------

    def delivery_summary(self) -> None:
        log = self.storage.get_delivery_log()
        print(f"\n── Delivery log: {len(log)} total delivery(s) ──")
        for r in log:
            status = "✓" if r.success else "✗"
            user = self.storage.get_user(r.recipient_id)
            name = user.name if user else r.recipient_id
            print(f"  {status} msg:{r.message_id[:8]}… → {name} via "
                  f"{r.endpoint.type.value}:{r.endpoint.address}")


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    system = MessageDistributionSystem()

    # Create users
    alice   = system.register_user("Alice (publisher)")
    bob     = system.register_user("Bob")
    carol   = system.register_user("Carol")
    dave    = system.register_user("Dave")

    # Bob and Carol follow Alice; Dave follows Alice too but won't pick endpoints
    system.follow(bob,   alice)
    system.follow(carol, alice)
    system.follow(dave,  alice)

    print("\n── Endpoint selection phase ──")

    # Bob wants Alice's messages via email AND push notification
    system.select_endpoints(bob, alice, {
        Endpoint(EndpointType.EMAIL, "bob@example.com"),
        Endpoint(EndpointType.PUSH,  "device-token-bob-xyz"),
    })

    # Carol wants them via SMS, email, AND in-app
    system.select_endpoints(carol, alice, {
        Endpoint(EndpointType.SMS,    "+1-555-0100"),
        Endpoint(EndpointType.EMAIL,  "carol@example.com"),
        Endpoint(EndpointType.IN_APP, "carol-user-id"),
    })

    # Dave follows but never calls select_endpoints → will be skipped

    print("\n── Message distribution phase ──")

    # Alice posts a message "to unspecified recipients"
    system.distribute_message(alice, "Hello everyone! Big announcement today.")

    # Alice posts another
    system.distribute_message(alice, "Follow-up: details are live on our site.")

    system.delivery_summary()
    
