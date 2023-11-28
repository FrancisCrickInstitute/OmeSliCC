

def int_to_rgba(intrgba: int) -> list:
    rgba = [x / 255 for x in intrgba.to_bytes(4, signed=True, byteorder="big")]
    return rgba


def rgba_to_int(rgba: list) -> int:
    intrgba = int.from_bytes([int(x * 255) for x in rgba], byteorder="big", signed=True)
    return intrgba


def rgba_to_hexrgb(rgba: list) -> str:
    hexrgb = ''.join([hex(int(x * 255))[2:].upper().zfill(2) for x in rgba[:3]])
    return hexrgb
