from pathlib import Path

MOCK_CLIPS = [
    {
        "id": "1089-134686-0000",
        "url": "https://example.com/stt/clips/1089-134686-0000.flac",
        "sha256": "sha256:aabbcc",
        "duration_s": 11.4,
        "reference": "HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS",
    },
    {
        "id": "1221-135766-0001",
        "url": "https://example.com/stt/clips/1221-135766-0001.flac",
        "sha256": "sha256:ddeeff",
        "duration_s": 6.3,
        "reference": "AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP",
    },
]

MOCK_CLIP_PATHS: dict[str, Path] = {
    "1089-134686-0000": Path("/tmp/1089-134686-0000.flac"),
    "1221-135766-0001": Path("/tmp/1221-135766-0001.flac"),
}
