
"""
Implementation of Claim 1 of US 2020/0202579 A1 (Dynamic Mask Application)

Claim 1 — A method comprising, by one or more computing devices:
  1. identifying a first user in an input image;
  2. accessing social data of the first user, where social data comprises
     information from a social graph of an online social network;
  3. selecting, based on the social data, a mask from a plurality of masks,
     where the plurality of masks comprise masks previously selected by
     friends of the first user, and the mask specifies one or more mask
     effects; and
  4. applying the mask to the first user in the input image.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import Counter
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Mask representation (corresponds to FIG. 3 "Dynamic Mask" / FIG. 4C table)
# ---------------------------------------------------------------------------

@dataclass
class MaskEffect:
    """A single mask effect: a mask image plus geometry/instructions."""
    mask_image_path: Optional[str] = None
    geometry: Dict[str, Tuple[int, int, int, int]] = field(default_factory=dict)
    # geometry maps feature name -> (left, right, top, bottom) coordinates


@dataclass
class DynamicMask:
    """A dynamic mask (360 in FIG. 3) with a name and set of effects."""
    name: str
    effects: List[MaskEffect]


class MaskDatabase:
    """Corresponds to mask database 322 / table 404 in FIG. 4C."""

    def __init__(self):
        self._masks: Dict[str, DynamicMask] = {}

    def add_mask(self, mask: DynamicMask) -> None:
        self._masks[mask.name] = mask

    def get_mask(self, name: str) -> Optional[DynamicMask]:
        return self._masks.get(name)

    def all_mask_names(self) -> List[str]:
        return list(self._masks.keys())


# ---------------------------------------------------------------------------
# Social graph (corresponds to FIG. 2)
# ---------------------------------------------------------------------------

class SocialGraph:
    """
    A minimal social graph: user nodes connected by "friend" edges, and
    user nodes connected to "selected_mask" concept nodes (representing
    masks the user has previously chosen).
    """

    def __init__(self):
        self.friends: Dict[str, set] = {}              # user -> set of friend user_ids
        self.selected_masks: Dict[str, List[str]] = {}  # user -> list of mask names chosen

    def add_user(self, user_id: str) -> None:
        self.friends.setdefault(user_id, set())
        self.selected_masks.setdefault(user_id, [])

    def add_friendship(self, user_a: str, user_b: str) -> None:
        self.add_user(user_a)
        self.add_user(user_b)
        self.friends[user_a].add(user_b)
        self.friends[user_b].add(user_a)

    def record_mask_selection(self, user_id: str, mask_name: str) -> None:
        self.add_user(user_id)
        self.selected_masks[user_id].append(mask_name)

    def get_friends(self, user_id: str) -> set:
        return self.friends.get(user_id, set())

    def get_social_data(self, user_id: str) -> dict:
        """
        Access social data of a user: includes information from the social
        graph such as the user's friends and the masks those friends have
        previously selected (as referenced in claim 1).
        """
        friends = self.get_friends(user_id)
        friends_masks: List[str] = []
        for friend_id in friends:
            friends_masks.extend(self.selected_masks.get(friend_id, []))

        return {
            "user_id": user_id,
            "friends": list(friends),
            "friends_selected_masks": friends_masks,
        }


# ---------------------------------------------------------------------------
# Face / user identification (step 1 of claim 1)
# ---------------------------------------------------------------------------

class UserIdentifier:
    """
    Identifies a first user in an input image.

    In a real system this would use facial recognition against a database
    of known users' faces. Here we provide a pluggable interface with a
    simple stub implementation.
    """

    def __init__(self, face_db: Optional[Dict[str, Tuple[int, int, int, int]]] = None):
        # face_db maps user_id -> a bounding box (left, top, right, bottom)
        # representing where that user's face is expected/found in the image
        self.face_db = face_db or {}

    def identify_first_user(self, input_image: Image.Image) -> Tuple[str, Tuple[int, int, int, int]]:
        """
        Identify the first user in the input image and return their user_id
        and the bounding box of their face.
        """
        if not self.face_db:
            raise ValueError("No known users registered for identification")

        # In a real implementation: run face detection on input_image,
        # extract embeddings, and match against stored embeddings via
        # facial recognition. Here we simply return the first registered user.
        user_id, bbox = next(iter(self.face_db.items()))
        return user_id, bbox


# ---------------------------------------------------------------------------
# Mask selection (step 3 of claim 1)
# ---------------------------------------------------------------------------

class MaskSelector:
    """
    Selects, based on the social data of the first user, a mask from a
    plurality of masks. The plurality of masks may comprise masks
    previously selected by friends of the first user (claim 1).
    """

    def __init__(self, mask_db: MaskDatabase, lookup_table: Optional[Dict[str, str]] = None):
        self.mask_db = mask_db
        # lookup_table optionally maps a social-data key (e.g. a theme,
        # or a friends'-mask name) directly to a selected mask name,
        # corresponding to claim 2's "lookup table that maps the social
        # data to the selected mask."
        self.lookup_table = lookup_table or {}

    def select_mask(self, social_data: dict) -> Optional[DynamicMask]:
        # 1. Try the explicit lookup table first (claim 2 mechanism)
        for key in social_data.get("friends_selected_masks", []):
            if key in self.lookup_table:
                mask_name = self.lookup_table[key]
                mask = self.mask_db.get_mask(mask_name)
                if mask:
                    return mask

        # 2. Otherwise, select the mask most frequently chosen among friends
        friends_masks = social_data.get("friends_selected_masks", [])
        if friends_masks:
            most_common_name, _ = Counter(friends_masks).most_common(1)[0]
            mask = self.mask_db.get_mask(most_common_name)
            if mask:
                return mask

        # 3. Fallback: no friend-selected masks available
        return None


# ---------------------------------------------------------------------------
# Mask application (step 4 of claim 1)
# ---------------------------------------------------------------------------

class MaskApplier:
    """
    Applies the selected mask to the first user in the input image,
    producing an output image (corresponds to "apply mask" 330 in FIG. 3).
    """

    def apply_mask(
        self,
        input_image: Image.Image,
        user_bbox: Tuple[int, int, int, int],
        mask: DynamicMask,
    ) -> Image.Image:
        output_image = input_image.copy()
        draw = ImageDraw.Draw(output_image)

        left, top, right, bottom = user_bbox
        width = right - left
        height = bottom - top

        for effect in mask.effects:
            if effect.mask_image_path:
                # Overlay an actual mask image (e.g., panda.jpg), scaled
                # and positioned to fit the user's face bounding box,
                # per the mask image geometry (FIG. 4C / 366).
                mask_img = Image.open(effect.mask_image_path).convert("RGBA")
                mask_img = mask_img.resize((width, height))
                output_image.paste(mask_img, (left, top), mask_img)
            else:
                # Fallback: simple rendering-code-style effect, e.g. draw
                # placeholder graphical features at locations derived from
                # the mask geometry, relative to the user's bounding box.
                for feature_name, (l, r, t, b) in effect.geometry.items():
                    fl = left + l
                    ft = top + t
                    fr = left + r
                    fb = top + b
                    draw.ellipse([fl, ft, fr, fb], outline="black", width=3)

        return output_image


# ---------------------------------------------------------------------------
# Top-level method implementing Claim 1
# ---------------------------------------------------------------------------

class DynamicMaskSystem:
    """
    Ties together identification, social-data access, mask selection, and
    mask application, implementing the method of Claim 1.
    """

    def __init__(
        self,
        social_graph: SocialGraph,
        mask_db: MaskDatabase,
        user_identifier: UserIdentifier,
        lookup_table: Optional[Dict[str, str]] = None,
    ):
        self.social_graph = social_graph
        self.mask_db = mask_db
        self.user_identifier = user_identifier
        self.mask_selector = MaskSelector(mask_db, lookup_table)
        self.mask_applier = MaskApplier()

    def process_image(self, input_image: Image.Image) -> Tuple[Image.Image, Optional[DynamicMask], str]:
        """
        Performs the steps of Claim 1 on a single input image.

        Returns:
            output_image: the image with the mask applied
            selected_mask: the DynamicMask that was selected (or None)
            user_id: the identified first user's id
        """
        # Step 1: identify a first user in an input image
        user_id, bbox = self.user_identifier.identify_first_user(input_image)

        # Step 2: access social data of the first user (from social graph)
        social_data = self.social_graph.get_social_data(user_id)

        # Step 3: select a mask based on the social data, where the
        # plurality of masks comprise masks previously selected by friends
        selected_mask = self.mask_selector.select_mask(social_data)

        # Step 4: apply the mask to the first user in the input image
        if selected_mask is not None:
            output_image = self.mask_applier.apply_mask(input_image, bbox, selected_mask)
        else:
            output_image = input_image.copy()

        return output_image, selected_mask, user_id


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Build a social graph (FIG. 2 style) -------------------------------
    graph = SocialGraph()
    graph.add_friendship("User_A", "User_B")
    graph.add_friendship("User_A", "User_C")

    # Friends of "User_A" have previously selected these masks
    graph.record_mask_selection("User_B", "happy_panda")
    graph.record_mask_selection("User_C", "happy_panda")
    graph.record_mask_selection("User_C", "angry_bird")

    # --- Build a mask database (FIG. 4C style) ------------------------------
    mask_db = MaskDatabase()
    panda_effect = MaskEffect(
        geometry={
            "left_eye": (40, 60, 30, 50),
            "right_eye": (90, 110, 30, 50),
            "nose": (60, 90, 70, 100),
        }
    )
    bird_effect = MaskEffect(
        geometry={
            "left_eye": (35, 65, 20, 50),
            "right_eye": (85, 115, 20, 50),
            "beak": (55, 95, 60, 110),
        }
    )
    mask_db.add_mask(DynamicMask(name="happy_panda", effects=[panda_effect]))
    mask_db.add_mask(DynamicMask(name="angry_bird", effects=[bird_effect]))

    # --- Set up user identification (stub face DB) -------------------------
    identifier = UserIdentifier(face_db={"User_A": (50, 50, 200, 200)})

    # --- Run the Claim 1 pipeline -------------------------------------------
    system = DynamicMaskSystem(social_graph=graph, mask_db=mask_db, user_identifier=identifier)

    input_image = Image.new("RGB", (300, 300), color="white")
    output_image, mask, user_id = system.process_image(input_image)

    print(f"Identified user: {user_id}")
    print(f"Selected mask (based on social data / friends' selections): "
          f"{mask.name if mask else None}")
    output_image.save("output_image.png")
    print("Output image saved to output_image.png")
