"""Turning a character's attributes into a portrait prompt.

Two rules do most of the work here.

**Skin tone is named as a hex, never as a word.** "Medium brown skin" drifts lighter on every
generation; `#a07e56` does not. Since the whole point of the panel is catching colour advice that
degrades at deeper skin tones, a portrait set that quietly lightens everyone would defeat it
before a stylist ever looked. This matters more, not less, now that the portraits are
photorealistic: the first faceless set already rendered a declared Monk 5 nearer Monk 3, and a
photographic face makes that drift read as a real person rather than an obvious artefact.

**The figures are photorealistic and have faces**, at the product owner's direction, replacing an
earlier stylised faceless figure. They depict no real individual: the prompt describes a
composite from declared attributes only, names no person, and asks for an ordinary neutral
expression rather than a recognisable likeness.

**Body shape is described geometrically, never by its label.** Image models read "rectangle" and
"triangle" as literal shapes. "Shoulders and hips of similar width with little waist definition"
is what the label actually means.

Everything except the character's own body, colouring and hair is held constant — camera, pose,
lighting, backdrop, and a plain mid-grey base garment. With the clothing fixed, two figures side
by side differ only in the things being evaluated, which makes the set a controlled comparison
rather than twenty-two pieces of unaudited styling advice from an image model.
"""

from typing import Any, Dict, Optional

# Monk Skin Tone scale, the scale production's own `skin_tone_monk` column already uses.
MONK_HEX = {
    1: "#f6ede4", 2: "#f3e7db", 3: "#f7ead0", 4: "#eadaba", 5: "#d7bd96",
    6: "#a07e56", 7: "#825c43", 8: "#604134", 9: "#3a312a", 10: "#292420",
}

# What each shape label means as geometry.
BODY_SHAPE_GEOMETRY = {
    "rectangle": "shoulders and hips of similar width with little waist definition",
    "triangle": "hips noticeably wider than the shoulders, weight carried low",
    "inverted_triangle": "shoulders noticeably broader than the hips",
    "hourglass": "shoulders and hips of similar width with a clearly narrower waist",
    "oval": "fullness carried at the midsection, with narrower shoulders and hips",
    "trapezoid": "shoulders broader than the waist, tapering evenly, athletic",
}

HEIGHT_BAND_NOTE = {
    "petite": "short in stature, compact proportions",
    "average": "average stature",
    "tall": "tall, long-limbed proportions",
}

BUILD_NOTE = {
    "slight": "very slim build", "lean": "lean build", "athletic": "athletic build",
    "average": "average build", "solid": "solid, broad build", "full": "full figure",
    "plus": "plus-size figure",
}

HAIR_LENGTH_NOTE = {
    "shaved": "shaved head", "cropped": "very short cropped hair", "short": "short hair",
    "chin": "chin-length hair", "shoulder": "shoulder-length hair",
    "mid_back": "hair to the middle of the back", "waist": "waist-length hair",
    "hip": "hip-length hair",
}

HEAD_COVERING_NOTE = {
    "turban": "wearing a neatly tied turban",
    "hijab": "wearing a hijab covering the hair and neck",
    "dupatta_over_head": "a dupatta draped over the head",
    "cap": "wearing a fitted cap",
    "stole": "a stole draped over the head and shoulders",
}

# Held identical across the roster so the set is comparable.
FIXED_FRAMING = (
    "Full-body figure, head to ankles, centred, shot straight on at eye level with an "
    "85mm-equivalent lens at f/5.6, sharp focus throughout. Neutral A-pose: standing upright, "
    "arms slightly away from the body, feet hip-width apart. Soft large key light at 45 degrees "
    "with gentle fill, no coloured gels. Plain solid #2B2B2B charcoal backdrop, no props, no "
    "furniture, no text."
)
FIXED_GARMENT = (
    "The figure wears only a plain mid-grey #8A8A8A fitted short-sleeved top with matching "
    "mid-thigh shorts. No pattern, no logo, no other clothing, no footwear, no jewellery."
)
FACE_RULE = (
    "Photorealistic documentary portrait photography of a real-looking adult person, with a "
    "natural, fully visible face: ordinary everyday features, relaxed neutral expression, "
    "direct gaze to camera, natural skin texture with visible pores and fine lines, no makeup "
    "beyond bare skin, no retouching, no beautification, no smoothing. Not a 3D render, not CGI, "
    "not a video-game character, not an illustration, not a mannequin or doll. "
    "The person is a fictional composite and resembles no specific real individual; give them an "
    "ordinary, unremarkable appearance rather than a model's."
)
NEGATIVE = (
    "Avoid: 3D render, CGI, video game character, cartoon, illustration, anime, plastic or waxy "
    "skin, airbrushed or smoothed skin, mannequin, doll, blank or featureless face, beauty "
    "retouching, glamour lighting, fashion-model proportions, skin lightened or whitened away "
    "from the stated hex."
)


def _monk_hex(monk: Optional[int]) -> str:
    if not monk:
        return MONK_HEX[6]
    return MONK_HEX.get(max(1, min(10, int(monk))), MONK_HEX[6])


def build_portrait_prompt(persona: Any) -> str:
    """Builds the prompt from a `Persona` row (or anything with the same attributes)."""
    hair: Dict[str, Any] = getattr(persona, "hair", None) or {}

    shape = (getattr(persona, "body_shape", "") or "rectangle").lower()
    geometry = BODY_SHAPE_GEOMETRY.get(shape, BODY_SHAPE_GEOMETRY["rectangle"])
    height_note = HEIGHT_BAND_NOTE.get(getattr(persona, "height_band", "") or "average", "average stature")
    build_note = BUILD_NOTE.get(getattr(persona, "build", "") or "average", "average build")
    height_cm = getattr(persona, "height_cm", None)
    height_phrase = f"approximately {height_cm}cm tall, " if height_cm else ""

    hair_bits = [HAIR_LENGTH_NOTE.get(hair.get("length", ""), "short hair")]
    if hair.get("texture"):
        hair_bits.append(f"{hair['texture']} texture")
    if hair.get("colour"):
        hair_bits.append(str(hair["colour"]).replace("_", " "))
    hair_phrase = ", ".join(hair_bits)

    extras = []
    if hair.get("facial_hair") and hair["facial_hair"] != "clean_shaven":
        extras.append(str(hair["facial_hair"]).replace("_", " "))
    covering = HEAD_COVERING_NOTE.get(hair.get("head_covering", "none"))
    if covering:
        extras.append(covering)
    extras_phrase = (" " + ". ".join(e.capitalize() for e in extras) + ".") if extras else ""

    age = getattr(persona, "age", None)
    gender_word = {"women": "woman", "men": "man"}.get(
        getattr(persona, "gender", "") or "", "person")
    age_phrase = f"{age}-year-old " if age else ""

    return (
        f"A full-body photograph of a {age_phrase}South Asian (Indian) {gender_word}. "
        f"Skin tone exactly matching the colour {_monk_hex(getattr(persona, 'skin_tone_monk', None))} "
        f"— reproduce this tone faithfully and do not lighten it. "
        f"{height_phrase}{height_note}, {build_note}, with {geometry}. "
        f"Hair: {hair_phrase}.{extras_phrase} "
        f"{FACE_RULE} {FIXED_GARMENT} {FIXED_FRAMING} {NEGATIVE}"
    )
